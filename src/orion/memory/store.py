"""SQLite persistence for selected episodic and durable memories."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    kind: str
    content: str
    topic: str
    created_at: str
    source: str


class SQLiteMemoryStore:
    """Store only selected memories. Conversation turns are never stored here."""

    _KINDS = {"episode", "fact", "preference", "decision"}

    def __init__(self, database_path: str | Path, *, connection: sqlite3.Connection | None = None):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._connection = connection or sqlite3.connect(
            str(Path(self.database_path).expanduser())
            if self.database_path != ":memory:"
            else ":memory:",
            check_same_thread=False,
            timeout=5.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if self.database_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('episode', 'fact', 'preference', 'decision')),
                content TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                source TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                topic,
                content='memories',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, topic)
                VALUES (new.rowid, new.content, new.topic);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, topic)
                VALUES ('delete', old.rowid, old.content, old.topic);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, topic)
                VALUES ('delete', old.rowid, old.content, old.topic);
                INSERT INTO memories_fts(rowid, content, topic)
                VALUES (new.rowid, new.content, new.topic);
            END;
            """
        )
        self._connection.commit()

    def save(
        self,
        kind: str,
        content: str,
        *,
        topic: str = "",
        source: str,
        memory_id: str | None = None,
        created_at: datetime | None = None,
    ) -> MemoryRecord:
        if kind not in self._KINDS:
            raise ValueError(f"Unsupported memory kind: {kind}")
        normalized_content = " ".join(content.split())
        normalized_topic = " ".join(topic.split())[:160]
        if not normalized_content:
            raise ValueError("Memory content must not be empty")
        record = MemoryRecord(
            id=memory_id or str(uuid.uuid4()),
            kind=kind,
            content=normalized_content,
            topic=normalized_topic,
            created_at=(created_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
            source=source,
        )
        with self._lock:
            self._ensure_open()
            self._connection.execute(
                """INSERT INTO memories (id, kind, content, topic, created_at, source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.kind,
                    record.content,
                    record.topic,
                    record.created_at,
                    record.source,
                ),
            )
            self._connection.commit()
        return record

    def search(self, query: str, *, limit: int = 5, kind: str | None = None) -> list[MemoryRecord]:
        if limit < 1:
            return []
        match_query = self._fts_query(query)
        if not match_query:
            return []
        with self._lock:
            self._ensure_open()
            if kind is not None:
                if kind not in self._KINDS:
                    raise ValueError(f"Unsupported memory kind: {kind}")
                rows = self._connection.execute(
                    """SELECT m.* FROM memories_fts f
                       JOIN memories m ON m.rowid = f.rowid
                       WHERE memories_fts MATCH ? AND m.kind = ?
                       ORDER BY bm25(memories_fts), m.created_at DESC LIMIT ?""",
                    (match_query, kind, limit),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """SELECT m.* FROM memories_fts f
                       JOIN memories m ON m.rowid = f.rowid
                       WHERE memories_fts MATCH ?
                       ORDER BY bm25(memories_fts), m.created_at DESC LIMIT ?""",
                    (match_query, limit),
                ).fetchall()
        return [self._record(row) for row in rows]

    def list_memories(self, *, kind: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        if limit < 1:
            return []
        with self._lock:
            self._ensure_open()
            if kind is not None and kind not in self._KINDS:
                raise ValueError(f"Unsupported memory kind: {kind}")
            if kind is None:
                rows = self._connection.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """SELECT * FROM memories WHERE kind = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (kind, limit),
                ).fetchall()
        return [self._record(row) for row in rows]

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._record(row) if row else None

    def delete_episode(self, memory_id: str) -> bool:
        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE id = ? AND kind = 'episode'", (memory_id,)
            )
            self._connection.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteMemoryStore:
        self._ensure_open()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = query.split()
        quoted = ['"' + term.replace('"', '""') + '"' for term in terms if term]
        return " OR ".join(quoted)

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            topic=row["topic"],
            created_at=row["created_at"],
            source=row["source"],
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Memory store is closed")