import json
import threading
import time
import unittest
from dataclasses import replace

from orion.config import Settings
from orion.core.agent import AgentController
from orion.core.context import ContextBudgetError, ContextBuilder
from orion.core.contracts import ChatMessage, ModelStreamEvent, ToolResult
from orion.memory.manager import MemoryManager
from orion.memory.policy import MemoryCandidate, MemorySaveStatus
from orion.memory.store import SQLiteMemoryStore


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, *, max_tokens, timeout):
        self.calls.append((tuple(messages), max_tokens))
        self.timeout = timeout
        if not self.responses:
            raise AssertionError("Unexpected model call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StreamingModel:
    def __init__(self, *events):
        self.events = events
        self.stream_calls = []

    def complete(self, *_args, **_kwargs):
        raise AssertionError("Tool-free streaming must not call complete()")

    def stream(self, messages, *, max_tokens, timeout, cancel_event):
        self.stream_calls.append((tuple(messages), max_tokens, timeout))
        yield from self.events


class FakeTool:
    name = "lookup"

    def __init__(self, result=None, error=None, delay=0):
        self.result = result or ToolResult("Found compact data.")
        self.error = error
        self.delay = delay
        self.calls = []

    def execute(self, arguments, *, timeout, cancel_event):
        self.calls.append(dict(arguments))
        self.timeout = timeout
        self.cancel_event = cancel_event
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def tool_call(name="lookup", arguments=None, **extra):
    decision = {"type": "tool", "name": name, "arguments": arguments or {}}
    decision.update(extra)
    return json.dumps(decision)


def final(response="Done.", memory_suggestions=None):
    return json.dumps(
        {
            "type": "final",
            "response": response,
            "memory_suggestions": memory_suggestions or [],
        }
    )


class ContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.settings = replace(
            Settings(), model_context_size=1024, max_context_tokens=420
        )
        self.builder = ContextBuilder(self.settings, max_tool_result_tokens=20)

    def test_current_request_and_required_task_state_are_preserved(self):
        context = self.builder.build(
            "Explain the current error",
            task_state="Tool lookup completed successfully.",
        )

        self.assertEqual(context.messages[-1].content, "Explain the current error")
        self.assertIn("Tool lookup completed successfully", context.messages[0].content)
        self.assertLessEqual(context.estimated_tokens, self.settings.max_context_tokens)

    def test_drops_old_turns_and_memories_to_fit_budget(self):
        turns = [
            ChatMessage("user", "older conversation detail " * 30),
            ChatMessage("assistant", "old response " * 30),
            ChatMessage("user", "recent point"),
        ]
        memories = ["relevant preference " * 30, "second memory " * 30]

        context = self.builder.build(
            "Keep this request",
            recent_messages=turns,
            memories=memories,
        )
        contents = [message.content for message in context.messages]

        self.assertEqual(context.messages[-1].content, "Keep this request")
        self.assertGreater(context.dropped_turns, 0)
        self.assertGreater(context.dropped_memories, 0)
        self.assertIn("recent point", " ".join(contents))
        self.assertLessEqual(context.estimated_tokens, self.settings.max_context_tokens)

    def test_compresses_summary_and_tool_results(self):
        context = self.builder.build(
            "Summarize results",
            summary="summary detail " * 200,
            tool_results=["result detail " * 200],
        )
        content = " ".join(message.content for message in context.messages)

        self.assertIn("Conversation summary:", content)
        self.assertIn("Tool result:", content)
        result_message = next(
            message.content for message in context.messages if "Tool result:" in message.content
        )
        self.assertLessEqual(len(result_message), 100)
        self.assertLessEqual(context.estimated_tokens, self.settings.max_context_tokens)

    def test_respects_custom_token_counter_with_unicode_content(self):
        settings = replace(self.settings, max_context_tokens=1000)
        builder = ContextBuilder(
            settings,
            token_counter=lambda text: len(text),
            max_tool_result_tokens=80,
        )
        context = builder.build(
            "Summarize this",
            memories=["偏好" * 300],
            summary="背景" * 300,
            tool_results=["结果" * 300],
        )

        self.assertLessEqual(context.estimated_tokens, settings.max_context_tokens)
        self.assertEqual(context.messages[-1].content, "Summarize this")
        self.assertIn("Tool result:", " ".join(m.content for m in context.messages))

    def test_raises_when_required_context_exceeds_budget(self):
        tiny = replace(self.settings, max_context_tokens=2)
        builder = ContextBuilder(tiny)

        with self.assertRaises(ContextBudgetError):
            builder.build("This request must not be dropped")

    def test_prompt_does_not_put_schema_placeholders_in_spoken_response(self):
        context = self.builder.build("Explain Kafka")

        self.assertNotIn("user-facing answer", context.messages[0].content)

    def test_marks_partial_assistant_turn_as_incomplete_in_new_context(self):
        context = self.builder.build(
            "Continue",
            recent_messages=[
                ChatMessage("assistant", "A partial answer", incomplete=True)
            ],
        )

        prior_turn = next(message for message in context.messages if "partial answer" in message.content)
        self.assertIn("Incomplete response", prior_turn.content)


class AgentControllerTests(unittest.TestCase):
    def settings(self, **changes):
        values = {
            "model_context_size": 1024,
            "max_context_tokens": 900,
            "max_agent_steps": 4,
            "max_tool_execution_time": 0.5,
            "max_total_task_time": 2.0,
        }
        values.update(changes)
        return replace(Settings(), **values)

    def controller(self, model, tool=None, **kwargs):
        tools = {"lookup": tool} if tool else {}
        return AgentController(self.settings(), model, tools, **kwargs)

    def test_calls_tool_then_returns_final_response(self):
        model = ScriptedModel(
            tool_call(arguments={"query": "status"}),
            final("The service is healthy."),
        )
        tool = FakeTool()

        result = self.controller(model, tool).run("Check service status")

        self.assertTrue(result.completed)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.response, "The service is healthy.")
        self.assertEqual(tool.calls, [{"query": "status"}])
        self.assertFalse(tool.cancel_event.is_set())
        self.assertTrue(any("Found compact data." in message.content for message in model.calls[1][0]))

    def test_tool_free_stream_skips_planner_and_filters_reasoning_tags(self):
        model = StreamingModel(
            ModelStreamEvent(kind="text", text="Visible <thi"),
            ModelStreamEvent(kind="text", text="nk>private </thi"),
            ModelStreamEvent(kind="text", text="nk> answer"),
            ModelStreamEvent(kind="completed", finish_reason="stop"),
        )

        events = list(self.controller(model).stream("Explain Kafka"))

        self.assertEqual(
            " ".join(
                "".join(event.text for event in events if event.kind == "text").split()
            ),
            "Visible answer",
        )
        self.assertEqual(len(model.stream_calls), 1)
        self.assertEqual(model.stream_calls[0][1], 512)
        self.assertNotIn("Return exactly one JSON object", model.stream_calls[0][0][0].content)

    def test_reasoning_filter_handles_unicode_and_mixed_case_split_tags(self):
        model = StreamingModel(
            ModelStreamEvent(kind="text", text="ß<THI"),
            ModelStreamEvent(kind="text", text="NK>private</THI"),
            ModelStreamEvent(kind="text", text="NK>ß"),
            ModelStreamEvent(kind="completed", finish_reason="stop"),
        )

        events = list(self.controller(model).stream("Explain Kafka"))

        self.assertEqual(
            "".join(event.text for event in events if event.kind == "text"),
            "ßß",
        )

    def test_authorized_tool_plan_finishes_before_final_stream(self):
        class PlanAndStreamModel(ScriptedModel):
            def stream(self, messages, *, max_tokens, timeout, cancel_event):
                self.streamed_messages = tuple(messages)
                self.stream_budget = max_tokens
                yield ModelStreamEvent(kind="text", text="The service is healthy.")
                yield ModelStreamEvent(kind="completed", finish_reason="stop")

        model = PlanAndStreamModel(
            tool_call(arguments={"query": "status"}),
            final("The service is healthy."),
        )
        tool = FakeTool()

        events = list(
            self.controller(model, tool).stream(
                "Check status",
                initial_task_state="Use only authorized tools.",
                allowed_tool_names=("lookup",),
            )
        )

        self.assertEqual(tool.calls, [{"query": "status"}])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.stream_budget, 512)
        self.assertEqual(
            "".join(event.text for event in events if event.kind == "text"),
            "The service is healthy.",
        )
        self.assertNotIn('"type":"tool"', " ".join(message.content for message in model.streamed_messages))

    def test_preserves_planner_guidance_after_each_tool_step(self):
        model = ScriptedModel(
            tool_call(arguments={"query": "first"}),
            tool_call(arguments={"query": "second"}),
            final(),
        )
        tool = FakeTool()
        guidance = "Authorized tools: lookup. Ask for clarification; never guess missing details."

        result = self.controller(model, tool).run(
            "Look up first and then second", initial_task_state=guidance
        )

        self.assertTrue(result.completed)
        for index, (messages, _tokens) in enumerate(model.calls):
            with self.subTest(step=index):
                self.assertIn(guidance, messages[0].content)
                if index:
                    self.assertIn(f"Step {index}: called lookup; result succeeded.", messages[0].content)
        self.assertNotIn("Step 1:", model.calls[2][0][0].content)

    def test_does_not_pass_extra_reasoning_field_to_tool(self):
        model = ScriptedModel(
            tool_call(
                arguments={"query": "status"},
                reasoning="private chain of thought must not be exposed",
            ),
            final(),
        )
        tool = FakeTool()

        result = self.controller(model, tool).run("Check status")

        self.assertTrue(result.completed)
        self.assertEqual(tool.calls, [{"query": "status"}])
        self.assertNotIn(
            "private chain of thought",
            " ".join(message.content for call, _tokens in model.calls for message in call),
        )

    def test_tool_failure_is_observed_without_exposing_exception(self):
        model = ScriptedModel(tool_call(), final("The lookup failed."))
        tool = FakeTool(error=RuntimeError("secret internal detail"))

        result = self.controller(model, tool).run("Look this up")
        second_prompt = " ".join(message.content for message in model.calls[1][0])

        self.assertTrue(result.completed)
        self.assertIn("Tool lookup failed", second_prompt)
        self.assertNotIn("secret internal detail", second_prompt)

    def test_unknown_tool_is_reported_to_local_model(self):
        model = ScriptedModel(tool_call(name="not_registered"), final("Unavailable."))

        result = self.controller(model).run("Do the operation")

        self.assertTrue(result.completed)
        self.assertIn("not available", " ".join(m.content for m in model.calls[1][0]))

    def test_repeat_action_stops_loop(self):
        model = ScriptedModel(tool_call(), tool_call())
        tool = FakeTool()

        result = self.controller(model, tool).run("Look this up")

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "repeated_action")
        self.assertEqual(len(tool.calls), 1)

    def test_cancellation_after_planner_response_blocks_tool_execution(self):
        cancellation = threading.Event()

        class CancellingModel(ScriptedModel):
            def complete(self, messages, *, max_tokens, timeout):
                cancellation.set()
                return tool_call(arguments={"query": "status"})

        tool = FakeTool()
        result = self.controller(CancellingModel(), tool).run(
            "Check status", cancel_event=cancellation
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "cancelled")
        self.assertEqual(tool.calls, [])

    def test_cancellation_reaches_running_tool(self):
        cancellation = threading.Event()

        class CancellingTool(FakeTool):
            def execute(self, arguments, *, timeout, cancel_event):
                cancellation.set()
                self.cancel_event = cancel_event
                cancel_event.wait(0.2)
                return ToolResult("Cancelled", success=False)

        tool = CancellingTool()
        result = self.controller(ScriptedModel(tool_call()), tool).run(
            "Check status", cancel_event=cancellation
        )

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "cancelled")
        self.assertTrue(tool.cancel_event.is_set())

    def test_max_step_limit_stops_loop(self):
        model = ScriptedModel(
            tool_call(arguments={"query": "one"}),
            tool_call(arguments={"query": "two"}),
        )
        tool = FakeTool()
        controller = AgentController(
            self.settings(max_agent_steps=2), model, {"lookup": tool}
        )

        result = controller.run("Search twice")

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "max_steps")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(model.calls), 2)

    def test_tool_timeout_is_bounded_and_reported(self):
        model = ScriptedModel(tool_call())
        tool = FakeTool(delay=0.1)
        controller = AgentController(
            self.settings(max_tool_execution_time=0.01), model, {"lookup": tool}
        )
        started = time.monotonic()

        result = controller.run("Look this up")

        elapsed = time.monotonic() - started
        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "tool_timeout")
        self.assertEqual(len(model.calls), 1)
        self.assertGreater(tool.timeout, 0)
        self.assertTrue(tool.cancel_event.is_set())
        self.assertLess(elapsed, 0.08)

    def test_model_call_respects_total_task_deadline(self):
        class SlowModel:
            def complete(self, _messages, *, max_tokens, timeout):
                time.sleep(0.1)
                return final("Too late")

        controller = AgentController(
            self.settings(max_total_task_time=0.01), SlowModel(), {}
        )
        started = time.monotonic()

        result = controller.run("Answer within the task deadline")

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "model_timeout")
        self.assertLess(time.monotonic() - started, 0.08)

    def test_retries_transient_model_failure_once(self):
        model = ScriptedModel(RuntimeError("temporary model failure"), final("Recovered."))

        result = self.controller(model).run("Answer this")

        self.assertTrue(result.completed)
        self.assertEqual(result.response, "Recovered.")
        self.assertEqual(len(model.calls), 2)

    def test_retries_malformed_model_decision_once(self):
        model = ScriptedModel("not JSON", final("Recovered."))

        result = self.controller(model).run("Answer this")

        self.assertTrue(result.completed)
        self.assertEqual(result.response, "Recovered.")
        self.assertEqual(len(model.calls), 2)

    def test_retries_prompt_placeholder_final_response(self):
        model = ScriptedModel(
            final("user-facing answer"),
            final("Kafka streams events between systems."),
        )

        result = self.controller(model).run("Explain Kafka")

        self.assertTrue(result.completed)
        self.assertEqual(result.response, "Kafka streams events between systems.")
        self.assertEqual(len(model.calls), 2)
        self.assertLessEqual(model.calls[0][1], 128)

    def test_repeated_prompt_placeholder_fails_closed(self):
        model = ScriptedModel(
            final("user-facing answer"),
            final("user-facing answer"),
        )

        result = self.controller(model).run("Explain Kafka")

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "invalid_model_response")
        self.assertNotIn("user-facing answer", result.response)

    def test_invalid_model_decision_returns_safe_fallback(self):
        model = ScriptedModel("I think the answer is 42")

        result = self.controller(model).run("Answer this")

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "invalid_model_response")
        self.assertNotIn("42", result.response)

    def test_invalid_or_truncated_model_output_never_executes_tool(self):
        model = ScriptedModel("not JSON", '{"type":"tool","name":"lookup"')
        tool = FakeTool()

        result = self.controller(model, tool).run("Look this up")

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "invalid_model_response")
        self.assertEqual(tool.calls, [])

    def test_empty_request_does_not_call_model(self):
        model = ScriptedModel()

        result = self.controller(model).run("   ")

        self.assertEqual(result.reason, "empty_request")
        self.assertEqual(model.calls, [])

    def test_saves_local_model_memory_suggestion_through_policy(self):
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        model = ScriptedModel(
            final(
                "I will remember your preference.",
                [
                    {
                        "kind": "preference",
                        "content": "The user prefers concise responses",
                        "topic": "response style",
                    }
                ],
            )
        )

        result = self.controller(model, memory=memory).run(
            "I prefer concise responses."
        )

        self.assertTrue(result.completed)
        saved = memory.list_memories(kind="preference")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].source, "model_recommendation")
        memory.close()

    def test_incognito_blocks_model_suggested_memory(self):
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        existing = memory.save_explicit(
            MemoryCandidate("fact", "Orion uses SQLite for memory", "architecture")
        )
        memory.set_incognito(True)
        model = ScriptedModel(
            final(
                "Understood.",
                [{"kind": "fact", "content": "The user lives in Pune"}],
            )
        )

        result = self.controller(model, memory=memory).run(
            "How does Orion store memory?"
        )

        self.assertTrue(result.completed)
        saved = memory.list_memories()
        self.assertEqual([record.id for record in saved], [existing.memory_id])
        self.assertIn(
            "Orion uses SQLite",
            " ".join(message.content for message in model.calls[0][0]),
        )
        memory.close()

    def test_model_memory_suggestion_rejects_sensitive_data(self):
        memory = MemoryManager(SQLiteMemoryStore(":memory:"))
        model = ScriptedModel(
            final(
                "I will keep that in mind.",
                [
                    {
                        "kind": "fact",
                        "content": "The user's API key is secret-value",
                        "topic": "credentials",
                    }
                ],
            )
        )

        result = self.controller(model, memory=memory).run("Here is a private detail")

        self.assertTrue(result.completed)
        self.assertEqual(memory.list_memories(), [])
        memory.close()


if __name__ == "__main__":
    unittest.main()