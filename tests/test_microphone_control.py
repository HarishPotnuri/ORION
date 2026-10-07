import sqlite3
import threading
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from orion.audio.control import MicrophoneControl


class FaultConnection(sqlite3.Connection):
    fail_read = False
    fail_write = False
    fail_commit = False
    fail_rollback = False
    before_write = None
    before_commit = None

    def execute(self, sql, parameters=()):
        if sql.startswith("SELECT id") and self.fail_read:
            raise sqlite3.OperationalError("private read details")
        if sql.startswith("INSERT INTO microphone_control"):
            if self.before_write is not None:
                self.before_write()
            if self.fail_write:
                raise sqlite3.OperationalError("private write details")
        return super().execute(sql, parameters)

    def commit(self):
        if self.before_commit is not None:
            self.before_commit()
        if self.fail_commit:
            raise sqlite3.OperationalError("private commit details")
        return super().commit()

    def rollback(self):
        if self.fail_rollback:
            raise sqlite3.OperationalError("private rollback details")
        return super().rollback()


class MicrophoneControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "control.sqlite3"

    def control(self, **kwargs):
        control = MicrophoneControl(self.database, **kwargs)
        self.addCleanup(control.close)
        return control

    def fault_control(self, *, fail_read=False):
        connect = sqlite3.connect

        def open_store(*args, **kwargs):
            store = connect(*args, **kwargs, factory=FaultConnection)
            store.fail_read = fail_read
            return store

        with patch("orion.audio.control.sqlite3.connect", side_effect=open_store):
            return self.control()

    def persisted(self):
        with closing(sqlite3.connect(self.database)) as store, store:
            return store.execute(
                "SELECT muted FROM microphone_control WHERE id = 1"
            ).fetchone()

    def test_default_and_reopen_persistence(self):
        control = self.control()
        self.assertIs(control.muted, False)
        self.assertEqual(control.generation, 0)
        self.assertFalse(control.cancel_event.is_set())
        self.assertIsNone(self.persisted())
        self.assertIs(control.toggle(), True)
        control.close()
        restored = self.control()
        self.assertIs(restored.muted, True)
        self.assertTrue(restored.cancel_event.is_set())
        self.assertIs(restored.toggle(), False)
        restored.close()
        self.assertIs(self.control().muted, False)

    def test_generation_events_and_claim(self):
        control = self.control()
        token = control.generation
        active = control.cancel_event
        self.assertTrue(control.allows(token))
        self.assertTrue(control.claim(token))
        self.assertFalse(control.claim(token + 1))
        control.set_muted(True)
        self.assertTrue(active.is_set())
        self.assertEqual(control.generation, token + 1)
        self.assertTrue(control.cancel_event.is_set())
        self.assertFalse(control.claim(control.generation))
        muted_event = control.cancel_event
        control.set_muted(False)
        self.assertTrue(muted_event.is_set())
        self.assertIsNot(control.cancel_event, active)
        self.assertFalse(control.cancel_event.is_set())
        self.assertEqual(control.generation, token + 2)
        self.assertFalse(control.claim(token))
        self.assertTrue(control.claim(control.generation))

    def test_noop_does_not_cancel_or_notify(self):
        calls = []
        control = self.control(on_change=lambda: calls.append(True))
        active = control.cancel_event
        self.assertIs(control.set_muted(False), False)
        self.assertEqual(control.generation, 0)
        self.assertIs(control.cancel_event, active)
        self.assertEqual(calls, [])
        control.set_muted(True)
        muted_event = control.cancel_event
        control.set_muted(True)
        self.assertEqual(control.generation, 1)
        self.assertIs(control.cancel_event, muted_event)
        self.assertEqual(calls, [True])

    def test_additive_schema_and_constraints(self):
        with closing(sqlite3.connect(self.database)) as store, store:
            store.execute("CREATE TABLE memories (value TEXT)")
            store.execute("INSERT INTO memories VALUES ('keep')")
        control = self.control()
        self.assertEqual(
            control.store.execute("PRAGMA journal_mode").fetchone(), ("wal",)
        )
        timeout = control.store.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 1000)
        control.toggle()
        with closing(sqlite3.connect(self.database)) as store, store:
            self.assertEqual(store.execute("SELECT * FROM memories").fetchall(), [("keep",)])
            self.assertEqual(
                {row[0] for row in store.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )},
                {"memories", "microphone_control"},
            )
            for values in ((2, 0), (1, 2), (1, None)):
                with self.assertRaises(sqlite3.IntegrityError):
                    store.execute(
                        "INSERT OR REPLACE INTO microphone_control VALUES (?, ?)", values
                    )

    def test_invalid_records_fail_closed(self):
        for rows in ([(1, 2)], [(1, "secret")], [(1, None)], [(2, 0)], [(1, 0), (2, 0)]):
            with self.subTest(rows=rows):
                with closing(sqlite3.connect(self.database)) as store, store:
                    store.execute("DROP TABLE IF EXISTS microphone_control")
                    store.execute("CREATE TABLE microphone_control (id, muted)")
                    store.executemany("INSERT INTO microphone_control VALUES (?, ?)", rows)
                with self.assertLogs("orion.audio.control", level="ERROR") as logs:
                    control = self.control()
                self.assertTrue(control.muted)
                self.assertTrue(control.cancel_event.is_set())
                self.assertFalse(control.claim(control.generation))
                self.assertIn("ValueError", " ".join(logs.output))
                self.assertNotIn("secret", " ".join(logs.output))
                control.close()

    def test_read_failure_starts_muted_and_logs_only_type(self):
        with self.assertLogs("orion.audio.control", level="ERROR") as logs:
            control = self.fault_control(fail_read=True)
        self.assertTrue(control.muted)
        self.assertTrue(control.cancel_event.is_set())
        self.assertIn("OperationalError", " ".join(logs.output))
        self.assertNotIn("private", " ".join(logs.output))

    def test_open_failure_refuses_unmute(self):
        with patch(
            "orion.audio.control.sqlite3.connect",
            side_effect=sqlite3.OperationalError("private database path"),
        ):
            with self.assertLogs("orion.audio.control", level="ERROR") as logs:
                control = self.control()
        self.assertIsNone(control.store)
        self.assertTrue(control.muted)
        old_event = control.cancel_event
        with self.assertLogs("orion.audio.control", level="ERROR") as write_logs:
            self.assertIs(control.toggle(), True)
        self.assertEqual(control.generation, 1)
        self.assertTrue(old_event.is_set())
        self.assertTrue(control.cancel_event.is_set())
        self.assertIn("RuntimeError", " ".join(write_logs.output))
        self.assertNotIn("private", " ".join(logs.output + write_logs.output))

    def test_mute_cancels_before_write_and_failure_stays_muted(self):
        calls = []
        control = self.fault_control()
        control.on_change = lambda: calls.append(control.muted)
        old_event = control.cancel_event

        def before_write():
            self.assertTrue(old_event.is_set())
            self.assertTrue(control.muted)
            self.assertEqual(control.generation, 1)

        control.store.before_write = before_write
        control.store.fail_write = True
        with self.assertLogs("orion.audio.control", level="ERROR") as logs:
            self.assertIs(control.toggle(), True)
        self.assertTrue(control.muted)
        self.assertEqual(control.generation, 1)
        self.assertFalse(control.claim(0))
        self.assertEqual(calls, [True])
        self.assertIsNone(self.persisted())
        self.assertNotIn("private", " ".join(logs.output))

    def test_unmute_requires_commit_and_failed_commit_rolls_back(self):
        control = self.fault_control()
        control.set_muted(True)
        old_event = control.cancel_event
        calls = []
        control.on_change = lambda: calls.append(control.muted)

        def before_commit():
            self.assertTrue(control.muted)
            self.assertTrue(old_event.is_set())
            self.assertFalse(control.claim(control.generation))
            self.assertEqual(self.persisted(), (1,))

        control.store.before_commit = before_commit
        control.store.fail_commit = True
        with self.assertLogs("orion.audio.control", level="ERROR") as logs:
            self.assertIs(control.set_muted(False), True)
        self.assertEqual(self.persisted(), (1,))
        self.assertEqual(control.generation, 2)
        self.assertTrue(control.cancel_event.is_set())
        self.assertEqual(calls, [True])
        self.assertNotIn("private", " ".join(logs.output))
        control.store.fail_commit = False
        self.assertIs(control.toggle(), False)
        self.assertEqual(self.persisted(), (0,))
        self.assertFalse(control.cancel_event.is_set())
        self.assertEqual(calls, [True, False])

    def test_failed_noop_write_cancels_active_generation(self):
        control = self.fault_control()
        active = control.cancel_event
        control.store.fail_write = True
        control.store.fail_rollback = True
        with self.assertLogs("orion.audio.control", level="ERROR") as logs:
            self.assertIs(control.set_muted(False), True)
        self.assertTrue(active.is_set())
        self.assertEqual(control.generation, 1)
        self.assertFalse(control.claim(0))
        self.assertNotIn("private", " ".join(logs.output))

    def test_callback_outside_lock_and_settable(self):
        control = self.control()
        calls = []

        def listener():
            finished = threading.Event()

            def inspect_state():
                calls.append((control.muted, control.generation))
                finished.set()

            worker = threading.Thread(target=inspect_state, daemon=True)
            worker.start()
            self.assertTrue(finished.wait(1), "Callback held the control lock")
            worker.join(1)

        control.on_change = listener
        self.assertIs(control.on_change, listener)
        control.toggle()
        control.toggle()
        self.assertEqual(calls, [(True, 1), (False, 2)])
        control.on_change = None
        control.toggle()
        self.assertEqual(len(calls), 2)

    def test_callback_error_does_not_change_state_or_leak_details(self):
        def listener():
            raise ValueError("private capture details")

        control = self.control(on_change=listener)
        with self.assertLogs("orion.audio.control", level="ERROR") as logs:
            self.assertIs(control.toggle(), True)
        self.assertEqual(self.persisted(), (1,))
        self.assertIn("ValueError", " ".join(logs.output))
        self.assertNotIn("private", " ".join(logs.output))
        control.on_change = None

    def test_concurrent_toggles_are_serialized(self):
        control = self.control()
        barrier = threading.Barrier(9)
        failures = []

        def toggle():
            try:
                barrier.wait(2)
                for _ in range(5):
                    control.toggle()
            except Exception as error:
                failures.append(error)

        workers = [threading.Thread(target=toggle, daemon=True) for _ in range(8)]
        for worker in workers:
            worker.start()
        barrier.wait(2)
        for worker in workers:
            worker.join(5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertFalse(control.muted)
        self.assertEqual(control.generation, 40)
        self.assertEqual(self.persisted(), (0,))

    def test_close_cancels_once_without_persisting_mute(self):
        calls = []
        control = self.control(on_change=lambda: calls.append(True))
        control.set_muted(False)
        old_event = control.cancel_event
        control.close()
        self.assertTrue(control.muted)
        self.assertTrue(old_event.is_set())
        self.assertTrue(control.cancel_event.is_set())
        self.assertEqual(control.generation, 1)
        self.assertIsNone(control.store)
        self.assertFalse(control.claim(0))
        control.close()
        self.assertEqual(control.generation, 1)
        self.assertEqual(calls, [True])
        self.assertIs(control.toggle(), True)
        self.assertIs(control.set_muted(False), True)
        self.assertEqual(control.generation, 1)
        self.assertEqual(self.persisted(), (0,))
        self.assertFalse(self.control().muted)

    def test_set_muted_requires_bool(self):
        control = self.control()
        with self.assertRaises(TypeError):
            control.set_muted(1)
        self.assertFalse(control.muted)
        self.assertEqual(control.generation, 0)