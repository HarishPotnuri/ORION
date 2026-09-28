"""Lightweight in-process polling scheduler for persistent reminders."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from .models import Reminder
from .store import SQLiteReminderStore

logger = logging.getLogger(__name__)


class ReminderNotifier:
    def __init__(
        self,
        speak: Callable[[str], None],
        set_status: Callable[[str], None],
    ):
        self.speak = speak
        self.set_status = set_status

    def __call__(self, reminder: Reminder) -> None:
        self.set_status("SPEAKING")
        message = f"Reminder: {reminder.title}."
        if reminder.notes:
            message = f"{message} {reminder.notes}"
        try:
            self.speak(message)
        finally:
            self.set_status("IDLE")


class ReminderScheduler:
    def __init__(
        self,
        store: SQLiteReminderStore,
        notify: Callable[[Reminder], None],
        *,
        poll_interval: float = 5.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        wait: Callable[[float], None] | None = None,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.store = store
        self.notify = notify
        self.poll_interval = poll_interval
        self.clock = clock
        self._stop_event = threading.Event()
        self._wait = wait or self._stop_event.wait
        self._thread: threading.Thread | None = None
        self._run_lock = threading.Lock()

    def process_due(self) -> int:
        """Deliver every currently due reminder, including all reminders missed at boot."""
        delivered_count = 0
        with self._run_lock:
            now = self.clock()
            due_reminders = self.store.due(now, limit=None)
            for due in due_reminders:
                claimed = self.store.claim_due(due.id, now)
                if claimed is None:
                    continue
                delivered = False
                try:
                    self.notify(claimed)
                    delivered = True
                    delivered_count += 1
                except Exception as error:
                    logger.warning(
                        "Reminder delivery failed (%s)", type(error).__name__
                    )
                finally:
                    self.store.finish_delivery(claimed, self.clock(), delivered=delivered)
        return delivered_count

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.process_due()
        self._thread = threading.Thread(
            target=self._poll,
            name="orion-reminder-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        self._thread = None

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.process_due()
            except Exception as error:
                logger.warning("Reminder scheduler poll failed (%s)", type(error).__name__)
            self._wait(self.poll_interval)

    def __enter__(self) -> ReminderScheduler:
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.stop()
