"""Durable microphone privacy state and generation-based input cancellation."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from threading import Event, RLock

logger = logging.getLogger(__name__)


class MicrophoneControl:
    """Serialize privacy changes; notify capture listeners outside the state lock.

    ``claim(token)`` marks the dispatch boundary, not completion of a request.
    Call it immediately before orchestration, which must run outside this lock.
    Later mute cancels input but does not undo an already claimed request.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._lock = RLock()
        self._muted = True
        self._generation = 0
        self._cancel_event = Event()
        self._cancel_event.set()
        self._on_change = on_change
        self._closed = False
        self.store: sqlite3.Connection | None = None
        try:
            self.store = sqlite3.connect(
                database_path, timeout=1.0, check_same_thread=False
            )
            self.store.execute("PRAGMA busy_timeout = 1000")
            self.store.execute("PRAGMA journal_mode = WAL")
            self.store.execute(
                "CREATE TABLE IF NOT EXISTS microphone_control ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "muted INTEGER NOT NULL CHECK (muted IN (0, 1)))"
            )
            self.store.commit()
        except Exception as error:
            logger.error("Microphone control storage failed: %s", type(error).__name__)
            if self.store is not None:
                try:
                    self.store.close()
                except Exception as close_error:
                    logger.error(
                        "Microphone control close failed: %s",
                        type(close_error).__name__,
                    )
            self.store = None
            return
        try:
            rows = self.store.execute(
                "SELECT id, muted FROM microphone_control"
            ).fetchall()
            if not rows:
                self._muted = False
            elif (
                len(rows) == 1
                and type(rows[0][0]) is int
                and rows[0][0] == 1
                and type(rows[0][1]) is int
                and rows[0][1] in (0, 1)
            ):
                self._muted = bool(rows[0][1])
            else:
                raise ValueError("Invalid microphone control record")
            if not self._muted:
                self._cancel_event = Event()
        except Exception as error:
            logger.error("Microphone control read failed: %s", type(error).__name__)

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._muted

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def cancel_event(self) -> Event:
        with self._lock:
            return self._cancel_event

    @property
    def on_change(self) -> Callable[[], None] | None:
        with self._lock:
            return self._on_change

    @on_change.setter
    def on_change(self, callback: Callable[[], None] | None) -> None:
        with self._lock:
            self._on_change = callback

    def allows(self, token: int) -> bool:
        with self._lock:
            return not self._muted and not self._closed and token == self._generation

    def claim(self, token: int) -> bool:
        """Atomically check permission at dispatch; never run orchestration here."""
        return self.allows(token)

    def _cancel_locked(self, muted: bool) -> None:
        self._cancel_event.set()
        self._generation += 1
        self._muted = muted
        self._cancel_event = Event()
        if muted:
            self._cancel_event.set()

    def _set_muted_locked(self, muted: bool) -> Callable[[], None] | None:
        if self._closed:
            return None
        changed = muted != self._muted
        if muted and changed:
            self._cancel_locked(True)
        try:
            if self.store is None:
                raise RuntimeError("Microphone control storage is unavailable")
            self.store.execute(
                "INSERT INTO microphone_control (id, muted) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET muted = excluded.muted",
                (int(muted),),
            )
            self.store.commit()
        except Exception as error:
            logger.error("Microphone control write failed: %s", type(error).__name__)
            if self.store is not None:
                try:
                    self.store.rollback()
                except Exception as rollback_error:
                    logger.error(
                        "Microphone control rollback failed: %s",
                        type(rollback_error).__name__,
                    )
            if not (muted and changed):
                self._cancel_locked(True)
            return self._on_change
        if not muted and changed:
            self._cancel_locked(False)
        return self._on_change if changed else None

    @staticmethod
    def _notify(callback: Callable[[], None] | None) -> None:
        if callback is not None:
            try:
                callback()
            except Exception as error:
                logger.error(
                    "Microphone control callback failed: %s", type(error).__name__
                )

    def toggle(self) -> bool:
        """Return actual muted state, including when persistence fails."""
        with self._lock:
            callback = self._set_muted_locked(not self._muted)
            actual = self._muted
        self._notify(callback)
        return actual

    def set_muted(self, muted: bool) -> bool:
        """Return actual muted state; unmute requires a successful commit."""
        if type(muted) is not bool:
            raise TypeError("muted must be a bool")
        with self._lock:
            callback = self._set_muted_locked(muted)
            actual = self._muted
        self._notify(callback)
        return actual

    def close(self) -> None:
        """Cancel input and close storage without changing the persisted state."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_locked(True)
            store, self.store = self.store, None
            if store is not None:
                try:
                    store.close()
                except Exception as error:
                    logger.error(
                        "Microphone control close failed: %s", type(error).__name__
                    )
            callback = self._on_change
        self._notify(callback)