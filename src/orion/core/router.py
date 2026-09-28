"""Lightweight deterministic request classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class RouteKind(StrEnum):
    SIMPLE_COMMAND = "SIMPLE_COMMAND"
    INCOGNITO = "INCOGNITO"
    REMINDER = "REMINDER"
    MEMORY_QUERY = "MEMORY_QUERY"
    MEMORY_DELETE = "MEMORY_DELETE"
    MEMORY_UPDATE = "MEMORY_UPDATE"
    LOCAL_TOOL = "LOCAL_TOOL"
    ONLINE_FACTUAL_LOOKUP = "ONLINE_FACTUAL_LOOKUP"
    LLM_REASONING = "LLM_REASONING"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    kind: RouteKind
    action: str | None = None


class OrionRouter:
    _simple_commands = (
        (re.compile(r"\b(mute yourself|be quiet|stop speaking|silence)\b"), "mute_speaker"),
        (re.compile(r"\b(unmute yourself|resume speaking)\b"), "unmute_speaker"),
        (re.compile(r"\b(mute microphone|mute mic|stop listening)\b"), "mute_microphone"),
        (re.compile(r"\b(unmute microphone|unmute mic|start listening)\b"), "unmute_microphone"),
    )
    _reminder_terms = re.compile(r"\bremind(?:er|ers)?\b")
    _time_only_followup = re.compile(
        r"^(?:at\s*)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)$",
        re.IGNORECASE,
    )
    _incognito_on = re.compile(
        r"\b(?:turn|switch|enable|start|activate)?\s*incognito(?: mode)?\s*(?:mode\s*)?(?:on|enable|enabled)\b"
    )
    _incognito_off = re.compile(
        r"\b(?:turn|switch|disable|stop|deactivate)?\s*incognito(?: mode)?\s*(?:mode\s*)?(?:off|disable|disabled)\b"
    )
    _memory_deletions = re.compile(
        r"\b(forget|delete|remove|erase)\b.{0,80}\b(memory|episode|conversation|discussion|decision|what we discussed)\b"
        r"|\b(forget|delete|remove|erase)\b.{0,80}\babout\b",
    )
    _memory_queries = re.compile(
        r"\b(what did we decide|what did i tell you|what do you remember|"
        r"what do you know about|do you remember|when did i say|recall)\b"
    )
    _memory_updates = re.compile(
        r"\b(remember this|remember that|remember me|remember i|save this|note that|store this|"
        r"my name is|i prefer|i like|i live in)\b"
    )
    _time_queries = re.compile(
        r"\b(what(?:'s| is) the time|what time is it|current time|time now|"
        r"tell me the time)\b"
    )
    _date_queries = re.compile(
        r"\b(what(?:'s| is) the date|what date is it|what day is it|"
        r"current date|today(?:'s| is the) date|tomorrow(?:'s| is the) date)\b"
    )
    _system_queries = re.compile(
        r"\b(system status|system health|device status|how much (?:free )?(?:memory|disk|storage)|"
        r"available (?:memory|storage))\b"
    )
    _online_weather = re.compile(r"\b(weather|forecast|temperature|rain|snow)\b")
    _online_news = re.compile(r"\b(news|headlines|breaking news)\b")
    _online_search = re.compile(
        r"\b(search|look up|lookup|browse|current|latest|recent|right now)\b"
    )

    def route(self, request: str) -> RouteDecision:
        normalized = " ".join(request.casefold().split())
        if not normalized:
            return RouteDecision(RouteKind.LLM_REASONING)

        if self._incognito_on.search(normalized):
            return RouteDecision(RouteKind.INCOGNITO, "on")
        if self._incognito_off.search(normalized):
            return RouteDecision(RouteKind.INCOGNITO, "off")

        for pattern, action in self._simple_commands:
            if pattern.search(normalized):
                return RouteDecision(RouteKind.SIMPLE_COMMAND, action)

        if self._time_only_followup.fullmatch(normalized.rstrip(" ?.!,")):
            return RouteDecision(RouteKind.REMINDER, "follow_up")

        if self._reminder_terms.search(normalized):
            return RouteDecision(RouteKind.REMINDER, self._reminder_action(normalized))

        if self._memory_deletions.search(normalized):
            return RouteDecision(RouteKind.MEMORY_DELETE)
        if self._memory_queries.search(normalized):
            return RouteDecision(RouteKind.MEMORY_QUERY)
        if self._memory_updates.search(normalized):
            return RouteDecision(RouteKind.MEMORY_UPDATE)

        if self._online_weather.search(normalized):
            return RouteDecision(RouteKind.ONLINE_FACTUAL_LOOKUP, "weather")
        if self._online_news.search(normalized):
            return RouteDecision(RouteKind.ONLINE_FACTUAL_LOOKUP, "news")
        if self._online_search.search(normalized):
            return RouteDecision(RouteKind.ONLINE_FACTUAL_LOOKUP, "web_search")

        if self._time_queries.search(normalized):
            return RouteDecision(RouteKind.LOCAL_TOOL, "time")
        if self._date_queries.search(normalized):
            return RouteDecision(RouteKind.LOCAL_TOOL, "date")
        if self._system_queries.search(normalized):
            return RouteDecision(RouteKind.LOCAL_TOOL, "system_status")

        return RouteDecision(RouteKind.LLM_REASONING)

    @staticmethod
    def _reminder_action(request: str) -> str:
        if re.search(r"\b(did i miss|missed|overdue|past due)\b", request):
            return "missed"
        if re.search(r"\b(cancel|delete|remove)\b", request):
            return "cancel"
        if re.search(r"\b(change|modify|edit|reschedule|update|add notes)\b", request):
            return "update"
        if re.search(
            r"\b(what did i (say|tell you)|original request|details|what is in)\b",
            request,
        ):
            return "query"
        if re.search(
            r"\b(list|show|what|which|any|do i have|have i got)\b.*\breminders?\b",
            request,
        ):
            return "list"
        return "create"
