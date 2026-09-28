"""Honest boundaries for requests beyond a small on-device model."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    supported: bool
    response: str = ""
    reason: str = ""


class CapabilityPolicy:
    _high_scope_patterns = (
        re.compile(
            r"\b(?:analy[sz]e|predict|forecast|project)\b.{0,100}\b(?:\d+|ten|twenty|decades?)\s+years?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:global|worldwide|across the world)\b.{0,100}\b(?:trend|industry|market|economy|companies)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhich company\b.{0,120}\b(?:dominate|lead|win)\b.{0,60}\b(?:next|future)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:predict|forecast)\b.{0,80}\b(?:next|coming)\s+(?:five|5|ten|10)\s+years?\b",
            re.IGNORECASE,
        ),
    )
    _fallback = (
        "I can't reliably perform that level of analysis with my on-device model. "
        "I can instead summarize available factual data or help with a narrower comparison."
    )

    def assess(self, request: str) -> CapabilityAssessment:
        normalized = " ".join(request.split())
        if any(pattern.search(normalized) for pattern in self._high_scope_patterns):
            return CapabilityAssessment(
                supported=False,
                response=self._fallback,
                reason="broad_long_horizon_analysis",
            )
        return CapabilityAssessment(supported=True)
