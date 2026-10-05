import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from orion.app import main
from orion.reminders.manager import ReminderManager
from orion.reminders.models import ReminderStatus
from orion.reminders.ownership import SchedulerAlreadyRunningError
from orion.reminders.parser import ReminderParseError, ReminderParser
from orion.reminders.scheduler import ReminderNotifier, ReminderScheduler
from orion.reminders.store import SQLiteReminderStore


class ReminderParserTests(unittest.TestCase):
    def setUp(self):
        self.zone = ZoneInfo("Asia/Kolkata")
        self.parser = ReminderParser()
        self.now = datetime(2026, 9, 28, 10, 0, tzinfo=self.zone)

    def test_parses_tomorrow_title_and_notes_in_default_timezone(self):
        reminder = self.parser.parse(
            "Remind me tomorrow at 1 PM about project meeting. Discuss API migration.",
            now=self.now,
        )

        self.assertEqual(reminder.title, "project meeting")
        self.assertEqual(reminder.due_at, datetime(2026, 9, 29, 13, 0, tzinfo=self.zone))
        self.assertEqual(reminder.timezone, "Asia/Kolkata")
        self.assertEqual(reminder.notes, "Discuss API migration")
        self.assertEqual(reminder.recurrence, "none")

    def test_parses_title_before_time_with_explicit_ist(self):
        reminder = self.parser.parse(
            "Remind me about the project meeting at 1 PM IST tomorrow. "
            "We need to discuss the API migration and I need to bring "
            "performance test results.",
            now=self.now,
        )

        self.assertEqual(reminder.title, "project meeting")
        self.assertEqual(reminder.due_at, datetime(2026, 9, 29, 13, 0, tzinfo=self.zone))
        self.assertEqual(
            reminder.notes,
            "We need to discuss the API migration and I need to bring "
            "performance test results",
        )

    def test_parses_compact_time_and_tomorrow_abbreviation(self):
        reminder = self.parser.parse(
            "Remind me about project meeting at 2PM tmrw",
            now=self.now,
        )

        self.assertEqual(reminder.title, "project meeting")
        self.assertEqual(
            reminder.due_at,
            datetime(2026, 9, 29, 14, 0, tzinfo=self.zone),
        )

    def test_rejects_context_dependent_reminder_title(self):
        with self.assertRaisesRegex(ReminderParseError, "Add reminder details"):
            self.parser.parse("Create a reminder for the same at 2PM tomorrow", now=self.now)

    def test_parses_relative_minutes_hours_and_days(self):
        expected = {
            "Remind me in 30 minutes to check deployment.": datetime(2026, 9, 28, 10, 30, tzinfo=self.zone),
            "Remind me in 2 hours to check deployment.": datetime(2026, 9, 28, 12, 0, tzinfo=self.zone),
            "Remind me in 2 days to check deployment.": datetime(2026, 9, 30, 10, 0, tzinfo=self.zone),
        }
        for request, due_at in expected.items():
            with self.subTest(request=request):
                self.assertEqual(self.parser.parse(request, now=self.now).due_at, due_at)

    def test_parses_daily_and_weekly_recurrence(self):
        daily = self.parser.parse("Remind me every day at 8 PM to take medicine", now=self.now)
        weekly = self.parser.parse("Remind me every Monday at 9 AM to review tasks", now=self.now)

        self.assertEqual(daily.recurrence, "daily")
        self.assertEqual(daily.due_at, datetime(2026, 9, 28, 20, 0, tzinfo=self.zone))
        self.assertEqual(weekly.recurrence, "weekly")
        self.assertEqual(weekly.recurrence_weekday, 0)
        self.assertEqual(weekly.due_at, datetime(2026, 10, 5, 9, 0, tzinfo=self.zone))

    def test_parses_schedule_without_requiring_a_new_title(self):
        due_at, recurrence, weekday = self.parser.parse_schedule(
            "Reschedule project meeting to tomorrow at 3 PM",
            now=self.now,
        )

        self.assertEqual(due_at, datetime(2026, 9, 29, 15, 0, tzinfo=self.zone))
        self.assertEqual(recurrence, "none")
        self.assertIsNone(weekday)

    def test_rejects_missing_or_past_time_instead_of_guessing(self):
        with self.assertRaises(ReminderParseError):
            self.parser.parse("Remind me to check deployment", now=self.now)
        with self.assertRaises(ReminderParseError):
            self.parser.parse("Remind me today at 9 AM to check deployment", now=self.now)


class ReminderStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "orion.sqlite3"
        self.store = SQLiteReminderStore(self.path)
        self.manager = ReminderManager(self.store)
        self.zone = ZoneInfo("Asia/Kolkata")

    def tearDown(self):
        self.manager.close()
        self.temp_dir.cleanup()

    def create(self, request="Remind me tomorrow at 1 PM about project meeting. Discuss API migration."):
        return self.manager.create_from_request(
            request,
            now=datetime(2026, 9, 28, 10, 0, tzinfo=self.zone),
        )

    def test_persists_original_request_timezone_and_notes_after_reopen(self):
        reminder = self.create()
        self.manager.close()
        reopened = ReminderManager(SQLiteReminderStore(self.path))
        try:
            loaded = reopened.get(reminder.id)
            self.assertEqual(loaded.original_request, "Remind me tomorrow at 1 PM about project meeting. Discuss API migration.")
            self.assertEqual(loaded.timezone, "Asia/Kolkata")
            self.assertEqual(loaded.notes, "Discuss API migration")
            self.assertEqual(loaded.due_at.astimezone(self.zone).hour, 13)
            self.assertEqual(loaded.status, ReminderStatus.PENDING)
        finally:
            reopened.close()

    def test_list_find_update_notes_and_cancel(self):
        reminder = self.create()
        self.assertEqual(len(self.manager.list(status="pending")), 1)
        self.assertEqual(self.manager.find("project meeting")[0].id, reminder.id)

        updated = self.manager.update(
            reminder.id,
            title="API planning meeting",
            notes="Bring performance tests",
            due_at=datetime(2026, 9, 30, 14, 0, tzinfo=self.zone),
        )
        self.assertEqual(updated.title, "API planning meeting")
        self.assertEqual(updated.notes, "Bring performance tests")
        self.assertEqual(updated.due_at.astimezone(self.zone).day, 30)

        self.assertTrue(self.manager.cancel(reminder.id))
        self.assertFalse(self.manager.cancel(reminder.id))
        self.assertEqual(self.manager.get(reminder.id).status, ReminderStatus.CANCELLED)

    def test_due_query_requires_aware_clock(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.store.due(datetime(2026, 9, 29, 13, 0))

    def test_reminders_coexist_with_memory_schema_in_same_database(self):
        from orion.memory.manager import MemoryManager
        from orion.memory.policy import MemoryCandidate
        from orion.memory.store import SQLiteMemoryStore

        memory = MemoryManager(SQLiteMemoryStore(self.path))
        saved_memory = memory.save_explicit(
            MemoryCandidate("fact", "Orion project uses local SQLite", "architecture")
        )
        reminder = self.create()
        memory.close()

        reopened_memory = SQLiteMemoryStore(self.path)
        reopened_reminders = SQLiteReminderStore(self.path)
        try:
            self.assertEqual(reopened_memory.get(saved_memory.memory_id).content, "Orion project uses local SQLite")
            self.assertEqual(reopened_reminders.get(reminder.id).title, "project meeting")
        finally:
            reopened_memory.close()
            reopened_reminders.close()


class ReminderSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "orion.sqlite3"
        self.store = SQLiteReminderStore(self.path)
        self.zone = ZoneInfo("Asia/Kolkata")
        self.clock_value = datetime(2026, 9, 29, 14, 0, tzinfo=self.zone)

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def add(self, request):
        return ReminderManager(self.store).create_from_request(
            request,
            now=datetime(2026, 9, 28, 10, 0, tzinfo=self.zone),
        )

    def test_start_delivers_every_missed_reminder_once(self):
        first = self.add("Remind me tomorrow at 1 PM to check deployment")
        second = self.add("Remind me tomorrow at 2 PM to review tests")
        delivered = []
        scheduler = ReminderScheduler(
            self.store,
            delivered.append,
            clock=lambda: self.clock_value,
            wait=lambda _duration: None,
        )

        scheduler.start()
        scheduler.stop()

        self.assertEqual({item.id for item in delivered}, {first.id, second.id})
        self.assertEqual(self.store.get(first.id).status, ReminderStatus.COMPLETED)
        self.assertEqual(self.store.get(second.id).status, ReminderStatus.COMPLETED)
        self.assertEqual(scheduler.process_due(), 0)

    def test_delivery_error_returns_reminder_to_pending_for_retry(self):
        reminder = self.add("Remind me tomorrow at 1 PM to check deployment")
        calls = []

        def fail(_reminder):
            calls.append("failed")
            raise RuntimeError("speaker unavailable")

        scheduler = ReminderScheduler(self.store, fail, clock=lambda: self.clock_value)
        self.assertEqual(scheduler.process_due(), 0)
        self.assertEqual(self.store.get(reminder.id).status, ReminderStatus.PENDING)

        scheduler.notify = calls.append
        self.assertEqual(scheduler.process_due(), 1)
        self.assertEqual(self.store.get(reminder.id).status, ReminderStatus.COMPLETED)
        self.assertEqual(calls[-1].id, reminder.id)

    def test_daily_and_weekly_reminders_advance_past_missed_occurrences(self):
        daily = self.add("Remind me every day at 9 AM to review tasks")
        weekly = self.add("Remind me every Monday at 9 AM to review project")
        now = datetime(2026, 10, 6, 14, 0, tzinfo=self.zone)
        delivered = []
        scheduler = ReminderScheduler(
            self.store, delivered.append, clock=lambda: now
        )

        self.assertEqual(scheduler.process_due(), 2)

        next_daily = self.store.get(daily.id)
        next_weekly = self.store.get(weekly.id)
        self.assertEqual(next_daily.status, ReminderStatus.PENDING)
        self.assertGreater(next_daily.due_at, self.clock_value.astimezone(UTC))
        self.assertEqual(next_daily.due_at.astimezone(self.zone).day, 7)
        self.assertEqual(next_weekly.due_at.astimezone(self.zone).date().weekday(), 0)
        self.assertGreater(next_weekly.due_at, self.clock_value.astimezone(UTC))

    def test_interrupted_delivery_recovers_as_pending_on_restart(self):
        reminder = self.add("Remind me tomorrow at 1 PM to check deployment")
        claimed = self.store.claim_due(reminder.id, self.clock_value)
        self.assertEqual(self.store.get(reminder.id).status, ReminderStatus.DELIVERING)
        self.store.close()

        self.store = SQLiteReminderStore(self.path)
        self.assertEqual(self.store.get(reminder.id).status, ReminderStatus.DELIVERING)
        delivered = []
        scheduler = ReminderScheduler(
            self.store,
            delivered.append,
            clock=lambda: self.clock_value,
        )
        scheduler.start()
        scheduler.stop()

        self.assertEqual([item.id for item in delivered], [reminder.id])
        self.assertEqual(self.store.get(reminder.id).status, ReminderStatus.COMPLETED)

    def test_only_one_scheduler_can_own_a_database(self):
        other_store = SQLiteReminderStore(self.path)
        owner = ReminderScheduler(self.store, lambda _reminder: None)
        second = ReminderScheduler(other_store, lambda _reminder: None)
        try:
            owner.start()
            with self.assertRaises(SchedulerAlreadyRunningError):
                second.start()
            owner.stop()
            second.start()
            second.stop()
        finally:
            owner.stop()
            second.stop()
            other_store.close()

    def test_direct_due_processing_cannot_bypass_scheduler_ownership(self):
        owner = ReminderScheduler(self.store, lambda _reminder: None)
        other_store = SQLiteReminderStore(self.path)
        second = ReminderScheduler(other_store, lambda _reminder: None)
        try:
            owner.start()
            with self.assertRaises(SchedulerAlreadyRunningError):
                second.process_due()
        finally:
            owner.stop()
            other_store.close()

    def test_two_store_connections_cannot_claim_same_reminder(self):
        reminder = self.add("Remind me tomorrow at 1 PM to check deployment")
        other_store = SQLiteReminderStore(self.path)
        barrier = threading.Barrier(2)
        claims = []
        failures = []

        def claim(store):
            try:
                barrier.wait()
                claims.append(store.claim_due(reminder.id, self.clock_value))
            except Exception as error:
                failures.append(error)

        first = threading.Thread(target=claim, args=(self.store,))
        second = threading.Thread(target=claim, args=(other_store,))
        first.start()
        second.start()
        first.join()
        second.join()
        other_store.close()

        self.assertEqual(failures, [])
        self.assertEqual(sum(claim is not None for claim in claims), 1)

    def test_notifier_sets_led_and_speaks_title_and_notes(self):
        reminder = ReminderManager(self.store).create_from_request(
            "Remind me tomorrow at 1 PM about project meeting. Discuss API migration.",
            now=datetime(2026, 9, 28, 10, 0, tzinfo=self.zone),
        )
        statuses = []
        speech = []
        notifier = ReminderNotifier(speech.append, statuses.append)

        notifier(reminder)

        self.assertEqual(statuses, ["SPEAKING", "IDLE"])
        self.assertEqual(
            speech,
            ["Reminder: project meeting. Discuss API migration"],
        )

    def test_notifier_restores_idle_state_when_speech_fails(self):
        reminder = ReminderManager(self.store).create_from_request(
            "Remind me tomorrow at 1 PM to test speech",
            now=datetime(2026, 9, 28, 10, 0, tzinfo=self.zone),
        )
        statuses = []

        def fail_speech(_text):
            raise RuntimeError("speaker unavailable")

        notifier = ReminderNotifier(fail_speech, statuses.append)

        with self.assertRaisesRegex(RuntimeError, "speaker unavailable"):
            notifier(reminder)
        self.assertEqual(statuses, ["SPEAKING", "IDLE"])

    def test_cli_starts_reminder_service_mode(self):
        with patch("orion.app._run_reminder_service") as run_service:
            self.assertEqual(main(["--run-reminders"]), 0)
        run_service.assert_called_once()


if __name__ == "__main__":
    unittest.main()