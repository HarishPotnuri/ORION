"""SQLite reminder table and durable reminder state transitions."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import Reminder, ReminderStatus


class SQLiteReminderStore:
    _RECURRENCES = {"none", "daily", "weekly"}

    def __init__(self, database_path: str | Path, *, connection: sqlite3.Connection | None = None):
        self.database_path = str(database_path)
        if connection is None:
            if self.database_path != ":memory:":
                Path(self.database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                ":memory:" if self.database_path == ":memory:" else str(Path(self.database_path).expanduser()),
                check_same_thread=False,
                timeout=5.0,
            )
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        with self._lock:
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if self.database_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    due_at_utc TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    original_request TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'delivering', 'completed', 'cancelled')),
                    recurrence TEXT NOT NULL CHECK (recurrence IN ('none', 'daily', 'weekly')),
                    recurrence_weekday INTEGER,
                    created_at_utc TEXT NOT NULL,
                    last_notified_at_utc TEXT
                )"""
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS reminders_due_idx ON reminders(status, due_at_utc)"
            )
            self._connection.execute(
                "UPDATE reminders SET status = 'pending' WHERE status = 'delivering'"
            )
            self._connection.commit()

    def create(
        self,
        *,
        title: str,
        due_at: datetime,
        timezone: str,
        notes: str = "",
        original_request: str,
        recurrence: str = "none",
        recurrence_weekday: int | None = None,
        reminder_id: str | None = None,
    ) -> Reminder:
        if due_at.tzinfo is None or due_at.utcoffset() is None:
            raise ValueError("Reminder due_at must be timezone-aware")
        if recurrence not in self._RECURRENCES:
            raise ValueError("recurrence must be none, daily, or weekly")
        if recurrence == "weekly" and recurrence_weekday not in range(7):
            raise ValueError("weekly recurrence requires weekday 0 through 6")
        if recurrence != "weekly":
            recurrence_weekday = None
        normalized_title = " ".join(title.split())
        normalized_request = " ".join(original_request.split())
        if not normalized_title or not normalized_request:
            raise ValueError("Reminder title and original request must not be empty")
        now = datetime.now(UTC)
        identifier = reminder_id or str(uuid.uuid4())
        due_utc = due_at.astimezone(UTC)
        with self._lock:
            self._ensure_open()
            self._connection.execute(
                """INSERT INTO reminders
                   (id, title, due_at_utc, timezone, notes, original_request, status,
                    recurrence, recurrence_weekday, created_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (
                    identifier,
                    normalized_title,
                    due_utc.isoformat(),
                    timezone,
                    notes.strip(),
                    normalized_request,
                    recurrence,
                    recurrence_weekday,
                    now.isoformat(),
                ),
            )
            self._connection.commit()
        return Reminder(
            identifier,
            normalized_title,
            due_utc,
            timezone,
            notes.strip(),
            normalized_request,
            ReminderStatus.PENDING,
            recurrence,
            recurrence_weekday,
            now,
        )

    def get(self, reminder_id: str) -> Reminder | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
        return self._to_reminder(row) if row else None

    def list(self, *, status: ReminderStatus | str | None = None, limit: int = 100) -> list[Reminder]:
        if limit < 1:
            return []
        with self._lock:
            self._ensure_open()
            if status is None:
                rows = self._connection.execute(
                    "SELECT * FROM reminders ORDER BY due_at_utc LIMIT ?", (limit,)
                ).fetchall()
            else:
                status_value = ReminderStatus(status).value
                rows = self._connection.execute(
                    "SELECT * FROM reminders WHERE status = ? ORDER BY due_at_utc LIMIT ?",
                    (status_value, limit),
                ).fetchall()
        return [self._to_reminder(row) for row in rows]

    def find(self, query: str, *, status: ReminderStatus | str | None = ReminderStatus.PENDING, limit: int = 20) -> list[Reminder]:
        terms = [term for term in query.casefold().split() if term]
        if not terms:
            return []
        clauses = ["lower(title || ' ' || notes || ' ' || original_request) LIKE ?" for _ in terms]
        parameters: list[object] = [f"%{term}%" for term in terms]
        where = " AND ".join(clauses)
        if status is not None:
            where += " AND status = ?"
            parameters.append(ReminderStatus(status).value)
        parameters.append(limit)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                f"SELECT * FROM reminders WHERE {where} ORDER BY due_at_utc LIMIT ?",
                parameters,
            ).fetchall()
        return [self._to_reminder(row) for row in rows]

    def due(self, now: datetime, *, limit: int | None = 100) -> list[Reminder]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        with self._lock:
            self._ensure_open()
            query = """SELECT * FROM reminders
                       WHERE status = 'pending' AND due_at_utc <= ?
                       ORDER BY due_at_utc"""
            parameters: tuple[object, ...] = (now.astimezone(UTC).isoformat(),)
            if limit is not None:
                query += " LIMIT ?"
                parameters += (limit,)
            rows = self._connection.execute(query, parameters).fetchall()
        return [self._to_reminder(row) for row in rows]

    def cancel(self, reminder_id: str) -> bool:
        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                (reminder_id,),
            )
            self._connection.commit()
        return cursor.rowcount > 0

    def update(
        self,
        reminder_id: str,
        *,
        title: str | None = None,
        due_at: datetime | None = None,
        timezone: str | None = None,
        notes: str | None = None,
        recurrence: str | None = None,
        recurrence_weekday: int | None = None,
    ) -> Reminder | None:
        existing = self.get(reminder_id)
        if existing is None or existing.status != ReminderStatus.PENDING:
            return None
        updated_recurrence = recurrence or existing.recurrence
        if updated_recurrence not in self._RECURRENCES:
            raise ValueError("recurrence must be none, daily, or weekly")
        updated_weekday = recurrence_weekday
        if updated_recurrence == "weekly":
            if updated_weekday is None:
                updated_weekday = existing.recurrence_weekday
            if updated_weekday not in range(7):
                raise ValueError("weekly recurrence requires weekday 0 through 6")
        else:
            updated_weekday = None
        if due_at is not None and (due_at.tzinfo is None or due_at.utcoffset() is None):
            raise ValueError("Reminder due_at must be timezone-aware")
        values = {
            "title": " ".join(title.split()) if title is not None else existing.title,
            "due_at_utc": (due_at or existing.due_at).astimezone(UTC).isoformat(),
            "timezone": timezone or existing.timezone,
            "notes": notes if notes is not None else existing.notes,
            "recurrence": updated_recurrence,
            "recurrence_weekday": updated_weekday,
        }
        with self._lock:
            self._ensure_open()
            self._connection.execute(
                """UPDATE reminders SET title = :title, due_at_utc = :due_at_utc,
                   timezone = :timezone, notes = :notes, recurrence = :recurrence,
                   recurrence_weekday = :recurrence_weekday
                   WHERE id = :id AND status = 'pending'""",
                {**values, "id": reminder_id},
            )
            self._connection.commit()
        return self.get(reminder_id)

    def claim_due(self, reminder_id: str, now: datetime) -> Reminder | None:
        """Atomically claim a due reminder so concurrent pollers do not duplicate it."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM reminders WHERE id = ? AND status = 'pending' AND due_at_utc <= ?",
                    (reminder_id, now.astimezone(UTC).isoformat()),
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                reminder = self._to_reminder(row)
                self._connection.execute(
                    "UPDATE reminders SET status = 'delivering' WHERE id = ? AND status = 'pending'",
                    (reminder_id,),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return reminder

    def finish_delivery(self, reminder: Reminder, now: datetime, *, delivered: bool) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        with self._lock:
            self._ensure_open()
            if not delivered:
                self._connection.execute(
                    "UPDATE reminders SET status = 'pending' WHERE id = ? AND status = 'delivering'",
                    (reminder.id,),
                )
            elif reminder.recurrence == "none":
                self._connection.execute(
                    """UPDATE reminders SET status = 'completed', last_notified_at_utc = ?
                       WHERE id = ? AND status = 'delivering'""",
                    (now.astimezone(UTC).isoformat(), reminder.id),
                )
            else:
                next_due = self._next_due(reminder, now)
                self._connection.execute(
                    """UPDATE reminders SET status = 'pending', due_at_utc = ?,
                       last_notified_at_utc = ? WHERE id = ? AND status = 'delivering'""",
                    (
                        next_due.astimezone(UTC).isoformat(),
                        now.astimezone(UTC).isoformat(),
                        reminder.id,
                    ),
                )
            self._connection.commit()

    def _next_due(self, reminder: Reminder, now: datetime) -> datetime:
        from datetime import timedelta
        from zoneinfo import ZoneInfo

        local_due = reminder.due_at.astimezone(ZoneInfo(reminder.timezone))
        local_now = now.astimezone(ZoneInfo(reminder.timezone))
        if reminder.recurrence == "daily":
            next_due = local_due + timedelta(days=1)
        else:
            days = (reminder.recurrence_weekday - local_due.weekday()) % 7
            if days == 0:
                days = 7
            next_due = local_due + timedelta(days=days)
        while next_due <= local_now:
            next_due += timedelta(days=1 if reminder.recurrence == "daily" else 7)
        return next_due

    @staticmethod
    def _to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"],
            title=row["title"],
            due_at=datetime.fromisoformat(row["due_at_utc"]),
            timezone=row["timezone"],
            notes=row["notes"],
            original_request=row["original_request"],
            status=ReminderStatus(row["status"]),
            recurrence=row["recurrence"],
            recurrence_weekday=row["recurrence_weekday"],
            created_at=datetime.fromisoformat(row["created_at_utc"]),
        )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Reminder store is closed")
