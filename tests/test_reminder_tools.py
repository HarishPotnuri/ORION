import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from orion.config import Settings
from orion.reminders.manager import ReminderManager
from orion.reminders.models import ReminderStatus
from orion.reminders.tool import ReminderTool
from orion.reminders.store import SQLiteReminderStore


class ReminderToolTests(unittest.TestCase):
    def test_expired_request_does_not_access_store(self):
        manager = Mock()
        result = ReminderTool(manager, "List reminders").execute(
            {"action": "list"}, timeout=0, cancel_event=threading.Event()
        )
        self.assertFalse(result.success)
        self.assertIn("timed out", result.content)
        manager.list.assert_not_called()

    def test_cancellation_during_confirmation_does_not_cancel_reminder(self):
        manager = Mock()
        manager.find.return_value = [Mock(
            id="meeting", title="project meeting", timezone="Asia/Kolkata",
            due_at=datetime(2026, 10, 10, 13, tzinfo=ZoneInfo("Asia/Kolkata")),
        )]
        cancelled = threading.Event()

        def confirm(_prompt):
            cancelled.set()
            return True

        result = ReminderTool(
            manager, "Cancel project meeting reminder", confirm_action=confirm
        ).execute({"action": "cancel", "query": "project meeting"}, timeout=1, cancel_event=cancelled)
        self.assertFalse(result.success)
        manager.cancel.assert_not_called()

    def test_agent_cannot_cancel_when_user_only_requested_list(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ReminderManager(
                SQLiteReminderStore(Path(directory) / "orion.sqlite3")
            )
            try:
                manager.create_from_request(
                    "Remind me tomorrow at 1 PM about project meeting",
                    now=datetime(2026, 9, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                )
                tool = ReminderTool(manager, "What reminders do I have tomorrow?")

                result = tool.execute(
                    {"action": "cancel", "query": "project meeting"},
                    timeout=1,
                    cancel_event=threading.Event(),
                )

                self.assertFalse(result.success)
                self.assertIn("does not authorize", result.content)
                self.assertEqual(len(manager.list(status=ReminderStatus.PENDING)), 1)
            finally:
                manager.close()

    def test_agent_refuses_reminder_details_not_present_in_request(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ReminderManager.from_settings(
                Settings(database_path=Path(directory) / "orion.sqlite3")
            )
            try:
                tool = ReminderTool(
                    manager,
                    "Remind me tomorrow at 8 AM to carry an umbrella",
                )
                result = tool.execute(
                    {
                        "action": "create",
                        "title": "buy a laptop",
                        "when": "tomorrow at 8 AM",
                    },
                    timeout=1,
                    cancel_event=threading.Event(),
                )

                self.assertFalse(result.success)
                self.assertIn("must come from the user's request", result.content)
                self.assertEqual(manager.list(status=ReminderStatus.PENDING), [])
            finally:
                manager.close()

    def test_agent_cancellation_requires_confirmation_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ReminderManager(
                SQLiteReminderStore(Path(directory) / "orion.sqlite3")
            )
            prompts = []
            try:
                reminder = manager.create_from_request(
                    "Remind me tomorrow at 1 PM about project meeting",
                    now=datetime(2026, 9, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                )
                tool = ReminderTool(
                    manager,
                    "Cancel my project meeting reminder",
                    confirm_action=lambda prompt: prompts.append(prompt) or False,
                )

                result = tool.execute(
                    {"action": "cancel", "query": "project meeting"},
                    timeout=1,
                    cancel_event=threading.Event(),
                )

                self.assertFalse(result.success)
                self.assertIn("not confirmed", result.content)
                self.assertIn("project meeting", prompts[0])
                self.assertEqual(manager.get(reminder.id).status, ReminderStatus.PENDING)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
