import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from orion.core.contracts import ChatMessage, MemorySuggestion
from orion.config import Settings
from orion.memory.manager import MemoryManager
from orion.memory.policy import (
    MemoryCandidate,
    MemoryKind,
    MemorySaveStatus,
)
from orion.memory.store import SQLiteMemoryStore


class MemoryManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "orion.sqlite3"
        self.store = SQLiteMemoryStore(self.database_path)
        self.memory = MemoryManager(self.store, max_recent_messages=4)

    def tearDown(self):
        self.memory.close()
        self.temp_dir.cleanup()

    def test_recent_conversation_turns_stay_volatile(self):
        self.memory.add_turn("I prefer brief answers", "I will keep answers concise.")

        self.assertEqual(
            self.memory.recent_messages(2),
            (
                ChatMessage("user", "I prefer brief answers"),
                ChatMessage("assistant", "I will keep answers concise."),
            ),
        )
        self.assertEqual(self.store.list_memories(), [])

    def test_turns_disappear_after_restart_but_selected_memories_persist(self):
        saved = self.memory.save_explicit(
            MemoryCandidate(MemoryKind.PREFERENCE, "The user prefers concise answers", "style")
        )
        self.assertEqual(saved.status, MemorySaveStatus.SAVED)
        self.memory.add_turn("Transient request", "Transient answer")
        self.memory.close()

        reopened = MemoryManager(SQLiteMemoryStore(self.database_path))
        try:
            self.assertEqual(reopened.recent_messages(10), ())
            records = reopened.list_memories(kind="preference")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].content, "The user prefers concise answers")
        finally:
            reopened.close()

    def test_explicit_saves_and_important_recommendations(self):
        explicit = self.memory.save_explicit(
            MemoryCandidate("fact", "The user lives in Pune", "location")
        )
        recommendation = self.memory.save_recommendation(
            MemorySuggestion("decision", "Orion uses SQLite for initial memory", "architecture"),
            important=True,
        )
        unimportant = self.memory.save_recommendation(
            MemorySuggestion("fact", "The user mentioned a passing detail"),
            important=False,
        )

        self.assertEqual(explicit.status, MemorySaveStatus.SAVED)
        self.assertEqual(recommendation.status, MemorySaveStatus.SAVED)
        self.assertEqual(unimportant.status, MemorySaveStatus.REJECTED_NOT_IMPORTANT)
        self.assertEqual(len(self.store.list_memories()), 2)

    def test_explicit_remember_request_is_classified_deterministically(self):
        preference = self.memory.save_explicit_request(
            "Remember that I prefer quiet alerts"
        )
        decision = self.memory.save_explicit_request(
            "Please remember we decided to use SQLite"
        )

        self.assertEqual(preference.status, MemorySaveStatus.SAVED)
        self.assertEqual(decision.status, MemorySaveStatus.SAVED)
        self.assertEqual(len(self.store.list_memories(kind="preference")), 1)
        self.assertEqual(len(self.store.list_memories(kind="decision")), 1)

    def test_empty_explicit_remember_request_does_not_write(self):
        result = self.memory.save_explicit_request("Remember this")

        self.assertEqual(result.status, MemorySaveStatus.REJECTED_EMPTY)
        self.assertEqual(self.store.list_memories(), [])

    def test_sensitive_values_are_rejected_even_when_explicit(self):
        result = self.memory.save_explicit(
            MemoryCandidate("fact", "Remember my API key: secret-value", "credentials")
        )

        self.assertEqual(result.status, MemorySaveStatus.REJECTED_SENSITIVE)
        self.assertEqual(self.store.list_memories(), [])

    def test_incognito_blocks_every_write_but_allows_memory_reads(self):
        existing = self.memory.save_explicit(
            MemoryCandidate("fact", "Orion project uses local SQLite", "architecture")
        )
        self.assertEqual(existing.status, MemorySaveStatus.SAVED)
        before = self.store.list_memories()
        self.memory.set_incognito(True)

        explicit = self.memory.save_explicit(
            MemoryCandidate("preference", "The user prefers concise answers", "style")
        )
        recommended = self.memory.save_recommendation(
            MemorySuggestion("decision", "Use a single local model", "architecture"),
            important=True,
        )
        episode = self.memory.save_episode(
            "We discussed project plans", topic="project", important=True
        )
        self.memory.add_turn("Private transient request", "Private transient answer")

        self.assertTrue(self.memory.incognito)
        self.assertEqual(explicit.status, MemorySaveStatus.BLOCKED_INCOGNITO)
        self.assertEqual(recommended.status, MemorySaveStatus.BLOCKED_INCOGNITO)
        self.assertEqual(episode.status, MemorySaveStatus.BLOCKED_INCOGNITO)
        self.assertEqual(self.store.list_memories(), before)
        self.assertEqual(len(self.memory.search("local SQLite", limit=5)), 1)
        self.assertEqual(len(self.memory.recent_messages(2)), 2)

    def test_incognito_off_resumes_memory_writes(self):
        self.memory.set_incognito(True)
        self.memory.set_incognito(False)

        result = self.memory.save_explicit(
            MemoryCandidate("preference", "The user prefers quiet alerts", "alerts")
        )

        self.assertEqual(result.status, MemorySaveStatus.SAVED)

    def test_memory_search_result_count_is_bounded(self):
        for index in range(4):
            self.memory.save_explicit(
                MemoryCandidate("fact", f"Orion architecture detail {index}", "architecture")
            )
        bounded = MemoryManager(
            self.store, max_recent_messages=4, max_search_results=2
        )

        self.assertEqual(len(bounded.search("Orion architecture", limit=20)), 2)

    def test_settings_factory_applies_configured_path_and_limits(self):
        settings = replace(
            Settings(),
            database_path=self.database_path,
            memory_recent_message_limit=3,
            memory_search_limit=2,
        )
        configured = MemoryManager.from_settings(settings)
        try:
            self.assertEqual(configured.conversation._messages.maxlen, 3)
            self.assertEqual(configured.max_search_results, 2)
            self.assertEqual(Path(configured.store.database_path), self.database_path)
        finally:
            configured.close()

    def test_episode_delete_by_query_deletes_only_unambiguous_match(self):
        first = self.memory.save_episode(
            "We agreed to migrate the old REST API", topic="migration", important=True
        )
        second = self.memory.save_episode(
            "We chose SQLite for Orion memory", topic="architecture", important=True
        )
        third = self.memory.save_episode(
            "We reviewed Orion memory retrieval", topic="architecture", important=True
        )

        ambiguous = self.memory.delete_episode_by_query("Orion memory")
        deleted = self.memory.delete_episode_by_query("old REST API")

        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertEqual({record.id for record in ambiguous.matches}, {second.memory_id, third.memory_id})
        self.assertEqual(deleted.status, "deleted")
        self.assertIsNone(self.store.get(first.memory_id))
        self.assertEqual(len(self.store.search("Orion memory", kind="episode")), 2)

    def test_delete_episode_cannot_delete_durable_memory(self):
        fact = self.memory.save_explicit(
            MemoryCandidate("fact", "Orion uses Python", "technology")
        )

        self.assertFalse(self.memory.delete_episode(fact.memory_id))
        self.assertIsNotNone(self.store.get(fact.memory_id))

    def test_short_term_buffer_is_bounded(self):
        for index in range(4):
            self.memory.add_turn(f"user {index}", f"assistant {index}")

        messages = self.memory.recent_messages(10)

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].content, "user 2")
        self.assertEqual(messages[-1].content, "assistant 3")


if __name__ == "__main__":
    unittest.main()