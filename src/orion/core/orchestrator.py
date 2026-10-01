"""Route requests to deterministic tools, factual APIs, or local reasoning."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..config import Settings
from ..memory.manager import EpisodeDeleteResult, MemoryManager
from ..memory.policy import MemorySaveStatus
from ..reminders.manager import ReminderManager
from ..reminders.models import Reminder
from ..reminders.tool import ReminderTool
from ..tools.registry import ToolRegistry
from .agent import AgentController
from .capabilities import CapabilityPolicy
from .context import ContextBuilder
from .contracts import LocalModel, ToolResult
from .router import OrionRouter, RouteDecision, RouteKind


@dataclass(frozen=True, slots=True)
class RequestResult:
    response: str
    route: RouteDecision
    completed: bool


class _RequestBoundFactualTool:
    """Keep agent-selected online queries inside the user's explicit request."""

    def __init__(self, tool, request: str):
        self._tool = tool
        self.name = tool.name
        self.description = tool.description if hasattr(tool, "description") else ""
        self._request = " ".join(request.casefold().split())

    def execute(self, arguments, *, timeout: float, cancel_event: threading.Event) -> ToolResult:
        query = arguments.get("query") if isinstance(arguments, Mapping) else None
        if not isinstance(query, str):
            return ToolResult("Online lookup needs a query from the user's request.", success=False)
        normalized_query = " ".join(query.casefold().split())
        if not normalized_query or len(normalized_query) > 300 or normalized_query not in self._request:
            return ToolResult("Online lookup query must come from the user's request.", success=False)
        if self.name == "web_search" and arguments.get("category") == "news":
            if not re.search(r"\b(?:news|headlines)\b", self._request):
                return ToolResult("News search was not requested.", success=False)
        return self._tool.execute(arguments, timeout=timeout, cancel_event=cancel_event)


class RequestOrchestrator:
    def __init__(
        self,
        settings: Settings,
        model: LocalModel,
        tools: ToolRegistry,
        *,
        memory: MemoryManager | None = None,
        reminders: ReminderManager | None = None,
        router: OrionRouter | None = None,
        capabilities: CapabilityPolicy | None = None,
        command_handlers: Mapping[str, Callable[[], str]] | None = None,
        confirm_action: Callable[[str], bool] | None = None,
    ):
        self.settings = settings
        self.model = model
        self.tools = tools
        self.memory = memory
        self.reminders = reminders
        self.router = router or OrionRouter()
        self.capabilities = capabilities or CapabilityPolicy()
        self.command_handlers = dict(command_handlers or {})
        self.confirm_action = confirm_action
        self.context_builder = ContextBuilder(settings)
        self.agent = AgentController(
            settings,
            model,
            tools.tools,
            context_builder=self.context_builder,
            memory=memory,
        )

    def handle(self, request: str) -> RequestResult:
        decision = self.router.route(request)
        result = self._dispatch(request, decision)
        if self.memory is not None and request.strip() and result.response.strip():
            self.memory.add_turn(request, result.response)
        return result

    def _dispatch(self, request: str, decision: RouteDecision) -> RequestResult:
        if decision.kind == RouteKind.INCOGNITO:
            if self.memory is None:
                return self._result(decision, "Memory is not available in this session.", False)
            enabled = decision.action == "on"
            self.memory.set_incognito(enabled)
            state = "on. New memory writes are blocked." if enabled else "off. Memory saving rules are active."
            return self._result(decision, f"Incognito mode is {state}")

        if decision.kind == RouteKind.MEMORY_UPDATE:
            if self.memory is None:
                return self._result(decision, "Memory is not available in this session.", False)
            save_result = self.memory.save_explicit_request(request)
            messages = {
                MemorySaveStatus.SAVED: "I saved that to memory.",
                MemorySaveStatus.BLOCKED_INCOGNITO: "Incognito mode is on, so I did not save anything.",
                MemorySaveStatus.REJECTED_SENSITIVE: "I did not save that because it may contain sensitive information.",
                MemorySaveStatus.REJECTED_EMPTY: "Tell me what you want me to remember.",
            }
            response = messages.get(save_result.status, "I could not save that memory.")
            return self._result(decision, response, save_result.status in {MemorySaveStatus.SAVED, MemorySaveStatus.BLOCKED_INCOGNITO})

        if decision.kind == RouteKind.MEMORY_QUERY:
            if self.memory is None:
                return self._result(decision, "Memory is not available in this session.", False)
            records = self.memory.search(request, limit=self.settings.memory_search_limit)
            if not records:
                return self._result(decision, "I could not find a matching saved memory.")
            return self._result(decision, "Saved memories: " + "; ".join(records))

        if decision.kind == RouteKind.MEMORY_DELETE:
            return self._delete_episode(request, decision)

        if decision.kind == RouteKind.REMINDER:
            result = self._handle_reminder(request, decision)
            if result.completed:
                return result
            if decision.action in {"create", "update"}:
                return self._run_planner(request, decision)
            return result

        if decision.kind == RouteKind.SIMPLE_COMMAND:
            handler = self.command_handlers.get(decision.action or "")
            if handler is None:
                return self._result(decision, "That physical control is not connected yet.", False)
            try:
                return self._result(decision, handler())
            except Exception:
                return self._result(decision, "That control did not respond.", False)

        if decision.kind == RouteKind.LOCAL_TOOL:
            name = "time" if decision.action in {"date", "time"} else decision.action
            tool = self.tools.get(name or "")
            if tool is None:
                return self._result(decision, "That local tool is not available.", False)
            result = self._execute(tool.name, {"action": decision.action, "query": request})
            return self._result(decision, result.content, result.success)

        if decision.kind == RouteKind.ONLINE_FACTUAL_LOOKUP:
            capability = self.capabilities.assess(request)
            if not capability.supported:
                return self._result(decision, capability.response, False)
            tool_name = "weather" if decision.action == "weather" else "web_search"
            tool = self.tools.get(tool_name)
            if tool is None:
                return self._result(decision, "That factual lookup tool is not available.", False)
            tool_arguments = {"query": request}
            if decision.action == "news":
                tool_arguments["category"] = "news"
            observation = self._execute(tool_name, tool_arguments)
            if not observation.success:
                return self._result(decision, observation.content, False)
            agent_result = self.agent.run(
                request,
                initial_observations=[f"Allowlisted factual data from {tool_name}: {observation.content}"],
                initial_task_state=(
                    f"An explicit factual lookup using {tool_name} completed. "
                    "Interpret only the supplied factual result. Do not request more tools."
                ),
                allowed_tool_names=(),
            )
            return self._result(decision, agent_result.response, agent_result.completed)

        capability = self.capabilities.assess(request)
        if not capability.supported:
            return self._result(decision, capability.response, False)
        return self._run_planner(request, decision)

    def _run_planner(self, request: str, decision: RouteDecision) -> RequestResult:
        allowed_names = self._authorized_agent_tools(request)
        available_tools = dict(self.tools.tools)
        for name in {"weather", "web_search"}.intersection(allowed_names):
            tool = available_tools.get(name)
            if tool is not None:
                available_tools[name] = _RequestBoundFactualTool(tool, request)
        if "reminder" in allowed_names and self.reminders is not None:
            reminder_tool = ReminderTool(
                self.reminders, request, confirm_action=self.confirm_action
            )
            available_tools[reminder_tool.name] = reminder_tool
        descriptions = {
            "time": 'time: arguments {"action":"time|date"}',
            "system_status": "system_status: arguments {}",
            "weather": 'weather: arguments {"query":"copy the relevant phrase from the user request"}',
            "web_search": 'web_search: arguments {"query":"copy the search phrase from the user request","category":"general|news"}',
            "reminder": ReminderTool.description,
        }
        tool_guidance = "\n".join(
            descriptions[name] for name in sorted(allowed_names) if name in descriptions
        )
        task_state = (
            "You are the planner. If request is incomplete, ambiguous, or unsupported, ask a concise clarification. "
            "Never guess missing reminder title, time, or target. Use only authorized tools listed here. "
            "Complete each requested step before reporting success.\nAuthorized tools:\n"
            + (tool_guidance or "none")
        )
        agent = AgentController(
            self.settings,
            self.model,
            available_tools,
            context_builder=self.context_builder,
            memory=self.memory,
        )
        agent_result = agent.run(
            request,
            initial_task_state=task_state,
            allowed_tool_names=allowed_names,
        )
        return self._result(decision, agent_result.response, agent_result.completed)

    def _authorized_agent_tools(self, request: str) -> tuple[str, ...]:
        normalized = " ".join(request.casefold().split())
        allowed: set[str] = set()
        if re.search(r"\b(?:weather|forecast|temperature|rain|snow)\b", normalized):
            allowed.add("weather")
        if re.search(r"\b(?:search|look up|lookup|browse|current|latest|recent|news|headlines)\b", normalized):
            allowed.add("web_search")
        if re.search(r"\b(?:what time|current time|time now|date|what day|today)\b", normalized):
            allowed.add("time")
        if re.search(r"\b(?:system|device|memory|disk|storage)\b.{0,30}\b(?:status|health|free|available)\b", normalized):
            allowed.add("system_status")
        if self.reminders is not None and re.search(r"\b(?:reminder|remind me)\b", normalized):
            allowed.add("reminder")
        allowed.intersection_update(self.tools.names() + (("reminder",) if self.reminders is not None else ()))
        return tuple(sorted(allowed))

    def _execute(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"Tool '{name}' is not available.", success=False)
        try:
            return tool.execute(
                arguments,
                timeout=self.settings.max_tool_execution_time,
                cancel_event=threading.Event(),
            )
        except Exception as error:
            return ToolResult(f"Tool failed ({type(error).__name__}).", success=False)

    def _delete_episode(self, request: str, decision: RouteDecision) -> RequestResult:
        if self.memory is None:
            return self._result(decision, "Memory is not available in this session.", False)
        query = re.sub(
            r"^.*?\b(?:about|that|from)\b\s*", "", request, flags=re.IGNORECASE
        ).strip(" ?.!\t")
        if not query:
            return self._result(decision, "Tell me which saved discussion to forget.", False)
        deletion: EpisodeDeleteResult = self.memory.find_episode_by_query(query)
        if deletion.status == "found":
            record = deletion.matches[0]
            if self.confirm_action is not None and not self.confirm_action(
                f"Delete saved memory {record.content}? Say yes to confirm or no to cancel."
            ):
                return self._result(decision, "Memory deletion was not confirmed.", False)
            deleted = self.memory.delete_episode(record.id)
            if not deleted:
                return self._result(decision, "I could not find a matching saved memory.", False)
            return self._result(decision, "I deleted the matching episode.")
        if deletion.status == "ambiguous":
            choices = "; ".join(
                f"{record.created_at[:10]}: {record.content[:120]}"
                for record in deletion.matches
            )
            return self._result(
                decision,
                f"I found several episodes. Which one should I delete? {choices}",
                False,
            )
        return self._result(decision, "I could not find a matching episode.", False)

    def _handle_reminder(self, request: str, decision: RouteDecision) -> RequestResult:
        if self.reminders is None:
            return self._result(
                decision, "Reminder service is not available in this session.", False
            )
        action = decision.action
        if action == "create":
            try:
                reminder = self.reminders.create_from_request(request)
            except ValueError as error:
                return self._result(decision, str(error), False)
            local_due = reminder.due_at.astimezone(ZoneInfo(reminder.timezone))
            return self._result(
                decision,
                f"Reminder set: {reminder.title} at "
                f"{local_due.strftime('%A %d %B, %I:%M %p')} {reminder.timezone}.",
            )
        if action == "follow_up":
            return self._result(
                decision,
                "I don't have the earlier reminder details in this one-shot request. "
                "Please include the reminder title, date, and time together.",
                False,
            )
        if action == "missed":
            now = datetime.now(ZoneInfo(self.settings.reminder_timezone))
            overdue = self.reminders.due(now)
            if not overdue:
                return self._result(decision, "You have no overdue reminders.")
            items = [
                f"{item.title}, due "
                f"{item.due_at.astimezone(ZoneInfo(item.timezone)).strftime('%d %B %I:%M %p')} "
                f"{item.timezone}"
                for item in overdue[:10]
            ]
            return self._result(decision, "Overdue reminders: " + "; ".join(items))
        if action == "list":
            reminders = self.reminders.list(status="pending")
            requested_day = re.search(
                r"\b(today|tomorrow|tmrw|tmr)\b", request, re.IGNORECASE
            )
            if requested_day:
                day_word = requested_day.group(1).casefold()
                target_day = datetime.now(
                    ZoneInfo(self.settings.reminder_timezone)
                ).date()
                if day_word in {"tomorrow", "tmrw", "tmr"}:
                    target_day += timedelta(days=1)
                reminders = [
                    item
                    for item in reminders
                    if item.due_at.astimezone(ZoneInfo(item.timezone)).date()
                    == target_day
                ]
            if not reminders:
                day_label = f" for {requested_day.group(1)}" if requested_day else ""
                return self._result(
                    decision, f"You have no pending reminders{day_label}."
                )
            items = [
                f"{item.title}, {item.due_at.astimezone(ZoneInfo(item.timezone)).strftime('%d %B %I:%M %p')} {item.timezone}"
                for item in reminders[:10]
            ]
            return self._result(decision, "Pending reminders: " + "; ".join(items))
        query = self._reminder_query(request)
        matches = (
            self.reminders.find(query)
            if query
            else self.reminders.list(status="pending")
        )
        if not matches:
            return self._result(
                decision, "I could not find a matching pending reminder.", False
            )
        if len(matches) > 1:
            return self._result(
                decision,
                "I found several matching reminders: "
                + "; ".join(f"{item.title} ({item.id})" for item in matches[:8]),
                False,
            )
        reminder = matches[0]
        if action == "cancel":
            if self.confirm_action is not None:
                local_due = reminder.due_at.astimezone(ZoneInfo(reminder.timezone))
                prompt = (
                    f"Cancel reminder {reminder.title}, due "
                    f"{local_due.strftime('%A %d %B at %I:%M %p')}? "
                    "Say yes to confirm or no to cancel."
                )
                if not self.confirm_action(prompt):
                    return self._result(decision, "Reminder cancellation was not confirmed.", False)
            cancelled = self.reminders.cancel(reminder.id)
            response = "Reminder cancelled." if cancelled else "Reminder was no longer pending."
            return self._result(decision, response, cancelled)
        if action == "query":
            return self._result(
                decision,
                f"{reminder.title}. Due {reminder.due_at.astimezone(ZoneInfo(reminder.timezone)).isoformat()} "
                f"({reminder.timezone}). Notes: {reminder.notes or 'none'}. "
                f"Original request: {reminder.original_request}",
            )
        if action == "update":
            if re.search(r"\bnotes?\b", request, flags=re.IGNORECASE):
                if ":" not in request:
                    return self._result(
                        decision,
                        "Add reminder notes after a colon, for example: add notes to the meeting reminder: bring the test results.",
                        False,
                    )
                notes = request.split(":", 1)[1].strip()
                updated = self.reminders.update(reminder.id, notes=notes)
            else:
                try:
                    updated = self.reminders.reschedule_from_request(
                        reminder.id, request
                    )
                except ValueError as error:
                    return self._result(decision, str(error), False)
            if updated is None:
                return self._result(decision, "I could not update that reminder.", False)
            return self._result(decision, f"Updated reminder: {updated.title}.")
        return self._result(
            decision,
            "I found the reminder, but need a clear new time or notes to update it.",
            False,
        )

    @staticmethod
    def _reminder_query(request: str) -> str:
        query_source = request.split(":", 1)[0]
        cleaned = re.sub(
            r"\b(remind(?:er)?|cancel|delete|remove|show|list|what|do|did|i|have|today|my|please|"
            r"tell|you|when|created|original|request|details|reschedule|rescheduled|"
            r"change|modify|edit|update|add|notes?|to)\b",
            " ",
            query_source,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:at|on|for|in|to)\s+(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|tomorrow|today|\d{1,4}\s+(?:minutes?|hours?|days?))\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b(?:the|a|an)\b", " ", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split()).strip(" ?.!,'\"")

    @staticmethod
    def _result(
        decision: RouteDecision, response: str, completed: bool = True
    ) -> RequestResult:
        return RequestResult(response, decision, completed)
