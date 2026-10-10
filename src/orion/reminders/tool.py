"""Validated reminder operations for bounded local agent planning."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from zoneinfo import ZoneInfo

from ..core.contracts import ToolResult
from .manager import ReminderManager
from .models import ParsedReminder, ReminderStatus


class ReminderTool:
    name = "reminder"
    description = (
        "Manage reminders with action list, find, create, reschedule, add_notes, or cancel. "
        "Arguments: action; query for find/update/cancel; title, when, optional notes for create; "
        "when for reschedule; notes for add_notes. Use only details stated by the user."
    )
    _actions = {"list", "find", "create", "reschedule", "add_notes", "cancel"}
    _LIMITS = {"action": 20, "query": 160, "title": 240, "when": 160, "notes": 2000}

    def __init__(
        self,
        manager: ReminderManager,
        request: str,
        *,
        confirm_action: Callable[[str], bool] | None = None,
    ):
        self.manager = manager
        self.confirm_action = confirm_action
        self.original_request = " ".join(request.split())
        self.request = " ".join(request.casefold().split())

    def execute(self, arguments, *, timeout: float, cancel_event: threading.Event) -> ToolResult:
        deadline = time.monotonic() + timeout
        check_active = lambda: self._check_active(deadline, cancel_event)
        inactive = check_active()
        if inactive is not None:
            return inactive
        try:
            values = self._validate_arguments(arguments)
            action = values["action"]
            if not self._authorized(action):
                return ToolResult("The request does not authorize that reminder action.", success=False)
            if action == "list":
                reminders = self.manager.list(status=ReminderStatus.PENDING)[:10]
                return ToolResult(self._serialize(reminders))
            if action == "create":
                return self._create(values, check_active)
            query = values.get("query", "")
            reminder = self._resolve(query)
            if isinstance(reminder, ToolResult):
                return reminder
            if action == "find":
                return ToolResult(self._serialize([reminder]))
            if action == "cancel":
                if self.confirm_action is not None:
                    local_due = reminder.due_at.astimezone(ZoneInfo(reminder.timezone))
                    prompt = (
                        f"Cancel reminder {reminder.title}, due "
                        f"{local_due.strftime('%A %d %B at %I:%M %p')}? "
                        "Say yes to confirm or no to cancel."
                    )
                    if not self.confirm_action(prompt):
                        return ToolResult("Reminder cancellation was not confirmed.", success=False)
                inactive = check_active()
                if inactive is not None:
                    return inactive
                cancelled = self.manager.cancel(reminder.id)
                return ToolResult(
                    "Reminder cancelled." if cancelled else "Reminder is no longer pending.",
                    success=cancelled,
                )
            if action == "add_notes":
                notes = values.get("notes", "")
                if not notes:
                    return ToolResult("Specify reminder notes to add.", success=False)
                if not self._appears_in_request(notes):
                    return ToolResult("Reminder notes must come from the user's request.", success=False)
                inactive = check_active()
                if inactive is not None:
                    return inactive
                updated = self.manager.update(reminder.id, notes=notes)
                return ToolResult("Reminder notes updated." if updated else "Reminder update failed.", success=updated is not None)
            return self._reschedule(reminder.id, values.get("when", ""), check_active)
        except ValueError as error:
            return ToolResult(str(error), success=False)
        except (sqlite3.Error, RuntimeError):
            return ToolResult("Reminder service is unavailable. Check the reminder database and retry; the action was not confirmed successful.", success=False)

    @staticmethod
    def _check_active(deadline: float, cancel_event: threading.Event) -> ToolResult | None:
        if cancel_event.is_set():
            return ToolResult("Reminder action cancelled.", success=False)
        if time.monotonic() >= deadline:
            return ToolResult("Reminder action timed out. List reminders before retrying a change.", success=False)
        return None

    def _create(self, values: dict[str, str], check_active: Callable[[], ToolResult | None]) -> ToolResult:
        title = values.get("title", "").strip()
        when = values.get("when", "").strip()
        if not title or not when:
            return ToolResult("A reminder needs a title and a specific future time.", success=False)
        if not self._appears_in_request(when) or not self._appears_in_request(title):
            return ToolResult("Reminder title and time must come from the user's request.", success=False)
        due_at, recurrence, weekday = self.manager.parser.parse_schedule(when)
        notes = values.get("notes", "")
        if notes and not self._appears_in_request(notes):
            return ToolResult("Reminder notes must come from the user's request.", success=False)
        parsed = ParsedReminder(
            title=title,
            due_at=due_at,
            timezone=self.manager.default_timezone,
            recurrence=recurrence,
            recurrence_weekday=weekday,
            notes=notes,
        )
        inactive = check_active()
        if inactive is not None:
            return inactive
        reminder = self.manager.create(parsed, original_request=self.original_request)
        return ToolResult(self._serialize([reminder]))

    def _reschedule(self, reminder_id: str, when: str, check_active: Callable[[], ToolResult | None]) -> ToolResult:
        if not when:
            return ToolResult("Specify the new reminder time.", success=False)
        if not self._appears_in_request(when):
            return ToolResult("New reminder time must come from the user's request.", success=False)
        due_at, recurrence, weekday = self.manager.parser.parse_schedule(when)
        inactive = check_active()
        if inactive is not None:
            return inactive
        updated = self.manager.update(
            reminder_id,
            due_at=due_at,
            timezone=self.manager.default_timezone,
            recurrence=recurrence,
            recurrence_weekday=weekday,
        )
        return ToolResult(self._serialize([updated]) if updated else "Reminder update failed.", success=updated is not None)

    def _resolve(self, query: str):
        if not query:
            return ToolResult("Specify which reminder.", success=False)
        if not self._appears_in_request(query):
            return ToolResult("Reminder search must use words from the user's request.", success=False)
        matches = self.manager.find(query, status=ReminderStatus.PENDING)
        if not matches:
            return ToolResult("No matching pending reminder.", success=False)
        if len(matches) != 1:
            return ToolResult(self._serialize(matches[:8]) + " Clarify which reminder.", success=False)
        return matches[0]

    def _authorized(self, action: str) -> bool:
        has_reminder = "reminder" in self.request or "remind me" in self.request
        if action in {"list", "find"}:
            return has_reminder
        if action == "create":
            return has_reminder and bool(re.search(r"\b(?:remind me|set|create|schedule)\b", self.request))
        if action == "cancel":
            return has_reminder and bool(re.search(r"\b(?:cancel|delete|remove)\b", self.request))
        if action == "reschedule":
            return has_reminder and bool(re.search(r"\b(?:reschedule|move|shift|change|modify|update)\b", self.request))
        if action == "add_notes":
            return has_reminder and bool(re.search(r"\b(?:add|update|change|edit)\b.{0,60}\bnotes?\b", self.request))
        return False

    def _validate_arguments(self, arguments) -> dict[str, str]:
        if not isinstance(arguments, dict):
            raise ValueError("Reminder arguments must be an object.")
        allowed = set(self._LIMITS)
        if set(arguments) - allowed:
            raise ValueError("Reminder action contains unsupported arguments.")
        values: dict[str, str] = {}
        for key, limit in self._LIMITS.items():
            value = arguments.get(key, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"Reminder argument '{key}' is invalid.")
            values[key] = " ".join(value.split())
        if values["action"] not in self._actions:
            raise ValueError("Unsupported reminder action.")
        return values

    def _appears_in_request(self, value: str) -> bool:
        return " ".join(value.casefold().split()) in self.request

    @staticmethod
    def _serialize(reminders) -> str:
        payload = []
        for item in reminders:
            if item is None:
                continue
            local_due = item.due_at.astimezone(ZoneInfo(item.timezone))
            payload.append(
                {
                    "id": item.id,
                    "title": item.title,
                    "due": local_due.isoformat(),
                    "timezone": item.timezone,
                    "notes": item.notes,
                    "status": item.status.value,
                }
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
