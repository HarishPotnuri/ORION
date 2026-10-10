"""Sanitized failure classification for the shared factual JSON transport."""

from urllib.error import HTTPError, URLError

from ..core.contracts import ToolResult
from ..core.policy import NetworkPolicyError


def factual_failure(label: str, provider: str, error: Exception) -> ToolResult:
    cause = error
    for _ in range(8):
        if isinstance(cause, TimeoutError) or (
            isinstance(cause, URLError) and isinstance(cause.reason, TimeoutError)
        ):
            return ToolResult(f"{label} timed out. Try again later.", success=False)
        if isinstance(cause, NetworkPolicyError):
            return ToolResult(f"{label} blocked by HTTPS host policy. Check the exact allowed provider host.", success=False)
        if isinstance(cause, HTTPError):
            hint = "Check that the SearXNG instance allows JSON search." if provider == "SearXNG" else "Try again later."
            return ToolResult(f"{provider} rejected the lookup (HTTP {cause.code}). {hint}", success=False)
        if isinstance(cause, (ValueError, TypeError, AttributeError, KeyError)):
            return ToolResult(f"{provider} returned invalid or incomplete data. Check provider configuration and try again later.", success=False)
        if cause.__cause__ is None:
            break
        cause = cause.__cause__
    hint = " and that the instance allows JSON search" if provider == "SearXNG" else ""
    return ToolResult(f"{provider} is unavailable. Check connectivity{hint} and try again later.", success=False)