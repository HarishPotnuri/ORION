"""Deterministic parsing for common reminder time phrases."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import ParsedReminder


class ReminderParseError(ValueError):
    """Raised when a reminder request lacks a safe, unambiguous schedule."""


class ReminderParser:
    _weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    _time_pattern = re.compile(
        r"\b(?:at\s*)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<period>a\.?m\.?|p\.?m\.?)\b",
        re.IGNORECASE,
    )
    _relative_pattern = re.compile(
        r"\bin\s+(?P<amount>\d{1,4})\s+(?P<unit>minutes?|hours?|days?)\b",
        re.IGNORECASE,
    )

    def __init__(self, default_timezone: str = "Asia/Kolkata"):
        try:
            self.timezone = ZoneInfo(default_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown timezone: {default_timezone}") from error
        self.timezone_name = default_timezone

    def parse(self, request: str, *, now: datetime | None = None) -> ParsedReminder:
        original = " ".join(request.split())
        if not original:
            raise ReminderParseError("Reminder request is empty")
        current = now or datetime.now(self.timezone)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        current = current.astimezone(self.timezone)

        due_at, recurrence, weekday = self.parse_schedule(original, now=current)
        title, notes = self._extract_title_notes(original, self._schedule_match(original))
        if not title:
            raise ReminderParseError("Add reminder details after the time")
        return ParsedReminder(
            title=title,
            due_at=due_at,
            timezone=self.timezone_name,
            recurrence=recurrence,
            recurrence_weekday=weekday,
            notes=notes,
        )

    def parse_schedule(
        self, request: str, *, now: datetime | None = None
    ) -> tuple[datetime, str, int | None]:
        original = " ".join(request.split())
        if not original:
            raise ReminderParseError("Reminder request is empty")
        current = now or datetime.now(self.timezone)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        current = current.astimezone(self.timezone)

        relative = self._relative_pattern.search(original)
        if relative:
            due_at = self._relative_due(relative, current)
            recurrence = "none"
            weekday = None
        else:
            due_at, recurrence, weekday, _schedule_match = self._calendar_due(
                original, current
            )
        return due_at, recurrence, weekday

    def _schedule_match(self, request: str) -> re.Match:
        relative = self._relative_pattern.search(request)
        if relative:
            return relative
        recurrence = re.search(
            r"\bevery\s+(?:day|daily|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            request,
            re.IGNORECASE,
        )
        if recurrence:
            return recurrence
        day = re.search(
            r"\b(?:today|tomorrow|tmrw|tmr|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            request,
            re.IGNORECASE,
        )
        if day is None:
            raise ReminderParseError("Specify when, such as 'tomorrow' or 'every Monday'")
        return day

    @staticmethod
    def _relative_due(match: re.Match, current: datetime) -> datetime:
        amount = int(match.group("amount"))
        unit = match.group("unit").casefold()
        if unit.startswith("minute"):
            delta = timedelta(minutes=amount)
        elif unit.startswith("hour"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return current + delta

    def _calendar_due(
        self, request: str, current: datetime
    ) -> tuple[datetime, str, int | None, re.Match]:
        recurrence = "none"
        weekday = None
        recurring = re.search(
            r"\bevery\s+(?P<period>day|daily|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            request,
            re.IGNORECASE,
        )
        if recurring:
            period = recurring.group("period").casefold()
            if period in {"day", "daily"}:
                recurrence = "daily"
            else:
                recurrence = "weekly"
                weekday = self._weekdays.get(period)
                if weekday is None:
                    weekday = current.weekday()
        else:
            recurring = None

        time_match = self._time_pattern.search(request)
        if time_match is None:
            raise ReminderParseError("Specify a time, such as 'at 9 AM'")
        hour = int(time_match.group("hour"))
        minute = int(time_match.group("minute") or 0)
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            raise ReminderParseError("Reminder time is not valid")
        period = time_match.group("period").casefold().replace(".", "")
        hour = (hour % 12) + (12 if period == "pm" else 0)

        if recurring:
            if recurrence == "daily":
                day = current.date()
            else:
                days_ahead = (weekday - current.weekday()) % 7
                day = current.date() + timedelta(days=days_ahead)
            due_at = datetime.combine(day, time(hour, minute), self.timezone)
            if due_at <= current:
                due_at += timedelta(days=1 if recurrence == "daily" else 7)
            return due_at, recurrence, weekday, recurring

        day_match = re.search(
            r"\b(today|tomorrow|tmrw|tmr|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            request,
            re.IGNORECASE,
        )
        if day_match is None:
            raise ReminderParseError("Specify when, such as 'tomorrow' or 'every Monday'")
        day_word = day_match.group(1).casefold()
        if day_word in {"tmrw", "tmr"}:
            day_word = "tomorrow"
        if day_word == "today":
            day = current.date()
        elif day_word == "tomorrow":
            day = current.date() + timedelta(days=1)
        else:
            days_ahead = (self._weekdays[day_word] - current.weekday()) % 7
            day = current.date() + timedelta(days=days_ahead)
        due_at = datetime.combine(day, time(hour, minute), self.timezone)
        if due_at <= current:
            raise ReminderParseError("That time has passed; specify a future day")
        return due_at, "none", None, day_match

    @staticmethod
    def _extract_title_notes(request: str, schedule_match: re.Match) -> tuple[str, str]:
        tail = request[schedule_match.end() :]
        prefix = request[: schedule_match.start()]
        title_before_schedule = ReminderParser._clean_title_fragment(prefix)
        title_after_schedule = ReminderParser._clean_title_fragment(tail)
        if title_before_schedule:
            content = title_before_schedule
            notes_source = title_after_schedule
        else:
            content = title_after_schedule
            notes_source = ""
        sentences = re.split(r"[.!?]\s+", content, maxsplit=1)
        title = sentences[0].strip(" .,:;-\t") if sentences else ""
        notes = sentences[1].strip() if len(sentences) > 1 else notes_source
        title = re.sub(r"^(?:me\s+)?(?:about|to|for)\s+", "", title, flags=re.IGNORECASE)
        title = re.sub(r"^(?:the|a|an)\s+", "", title, flags=re.IGNORECASE)
        if title.casefold() in {"same", "that", "it", "there"}:
            title = ""
        return title[:240], notes[:2000]

    @staticmethod
    def _clean_title_fragment(fragment: str) -> str:
        content = re.sub(
            r"\b(?:at\s*)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:ist|utc)?\b",
            " ",
            fragment,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r"\bin\s+\d{1,4}\s+(?:minutes?|hours?|days?)\b",
            " ",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r"\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            " ",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r"^.*?\bremind(?:er)?(?:\s+me)?\b", " ", content, flags=re.IGNORECASE
        )
        content = re.sub(
            r"\bevery\s+(?:day|daily|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            " ",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"^\s*(?:about|to|for|that)\s+", "", content, flags=re.IGNORECASE)
        return content.strip(" .,:;-\t")
