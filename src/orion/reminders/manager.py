"""Reminder CRUD and natural-language parsing facade."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ..config import Settings
from .models import ParsedReminder, Reminder, ReminderStatus
from .parser import ReminderParser
from .store import SQLiteReminderStore


class ReminderManager:
    def __init__(
        self,
        store: SQLiteReminderStore,
        *,
        default_timezone: str = "Asia/Kolkata",
        parser: ReminderParser | None = None,
    ):
        self.store = store
        self.parser = parser or ReminderParser(default_timezone)
        self.default_timezone = default_timezone

    @classmethod
    def from_settings(cls, settings: Settings) -> ReminderManager:
        return cls(
            SQLiteReminderStore(settings.database_path),
            default_timezone=settings.reminder_timezone,
        )

    def create_from_request(
        self, request: str, *, now: datetime | None = None
    ) -> Reminder:
        parsed = self.parser.parse(request, now=now)
        return self.create(parsed, original_request=request)

    def create(self, parsed: ParsedReminder, *, original_request: str) -> Reminder:
        return self.store.create(
            title=parsed.title,
            due_at=parsed.due_at,
            timezone=parsed.timezone,
            notes=parsed.notes,
            original_request=original_request,
            recurrence=parsed.recurrence,
            recurrence_weekday=parsed.recurrence_weekday,
        )

    def list(self, *, status: ReminderStatus | str | None = None) -> Sequence[Reminder]:
        return self.store.list(status=status)

    def due(self, now: datetime) -> Sequence[Reminder]:
        return self.store.due(now, limit=None)

    def find(self, query: str, *, status: ReminderStatus | str | None = ReminderStatus.PENDING) -> Sequence[Reminder]:
        return self.store.find(query, status=status)

    def get(self, reminder_id: str) -> Reminder | None:
        return self.store.get(reminder_id)

    def cancel(self, reminder_id: str) -> bool:
        return self.store.cancel(reminder_id)

    def update(self, reminder_id: str, **changes) -> Reminder | None:
        return self.store.update(reminder_id, **changes)

    def reschedule_from_request(
        self, reminder_id: str, request: str, *, now: datetime | None = None
    ) -> Reminder | None:
        due_at, recurrence, recurrence_weekday = self.parser.parse_schedule(
            request, now=now
        )
        return self.store.update(
            reminder_id,
            due_at=due_at,
            timezone=self.default_timezone,
            recurrence=recurrence,
            recurrence_weekday=recurrence_weekday,
        )

    def close(self) -> None:
        self.store.close()
