"""Reminder data types with timezone-aware due dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ReminderStatus(StrEnum):
    PENDING = "pending"
    DELIVERING = "delivering"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Reminder:
    id: str
    title: str
    due_at: datetime
    timezone: str
    notes: str
    original_request: str
    status: ReminderStatus
    recurrence: str
    recurrence_weekday: int | None
    created_at: datetime

    def __post_init__(self) -> None:
        if self.due_at.tzinfo is None or self.due_at.utcoffset() is None:
            raise ValueError("Reminder due_at must be timezone-aware")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("Reminder created_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ParsedReminder:
    title: str
    due_at: datetime
    timezone: str
    recurrence: str = "none"
    recurrence_weekday: int | None = None
    notes: str = ""
