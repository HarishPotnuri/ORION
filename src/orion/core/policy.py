"""Network policy for local model access and approved factual lookups."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


class NetworkPolicyError(ValueError):
    """Raised when a network request violates ORION's data boundary."""


class NetworkPolicy:
    def validate_model_endpoint(self, url: str) -> str:
        parsed = self._parse_url(url)
        if parsed.scheme not in {"http", "https"}:
            raise NetworkPolicyError("Local model endpoint must use HTTP or HTTPS")
        if parsed.username or parsed.password:
            raise NetworkPolicyError("Credentials are not allowed in model endpoint URLs")
        if not self._is_loopback(parsed.hostname or ""):
            raise NetworkPolicyError("Model endpoint must point to localhost")
        if parsed.query or parsed.fragment:
            raise NetworkPolicyError("Model endpoint must not include query or fragment")
        return url.rstrip("/")

    def validate_factual_url(self, url: str, allowed_hosts: set[str]) -> str:
        parsed = self._parse_url(url)
        hostname = (parsed.hostname or "").lower()
        hosts = {host.lower() for host in allowed_hosts}
        if parsed.scheme != "https":
            raise NetworkPolicyError("Factual lookup endpoints must use HTTPS")
        if parsed.username or parsed.password:
            raise NetworkPolicyError("Credentials are not allowed in factual lookup URLs")
        if parsed.fragment:
            raise NetworkPolicyError("Factual lookup URLs must not include fragments")
        if hostname not in hosts:
            raise NetworkPolicyError("Factual lookup host is not allowlisted")
        return url

    @staticmethod
    def _parse_url(url: str):
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError
            parsed.port
        except ValueError as error:
            raise NetworkPolicyError("Invalid network endpoint URL") from error
        return parsed

    @staticmethod
    def _is_loopback(hostname: str) -> bool:
        if hostname.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False
