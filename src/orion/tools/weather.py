"""Open-Meteo factual weather lookup with fixed HTTPS hosts."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from ..config import Settings
from ..core.contracts import ToolResult
from ..core.policy import NetworkPolicy
from ..llm.runtime import LocalJsonTransport

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_ALLOWED_HOSTS = {"geocoding-api.open-meteo.com", "api.open-meteo.com"}
_LOCATION_PATTERN = re.compile(
    r"\b(?:in|for|at)\s+([\w .'-]{2,80}?)(?=\s+(?:today|tomorrow|this week|next week)\b|[?.!,;]|$)",
    re.IGNORECASE,
)


class OpenMeteoWeatherTool:
    name = "weather"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Any | None = None,
        policy: NetworkPolicy | None = None,
    ):
        self.settings = settings
        self.transport = transport or LocalJsonTransport()
        self.policy = policy or NetworkPolicy()

    def execute(self, arguments, *, timeout: float, cancel_event: threading.Event) -> ToolResult:
        if not self.settings.online_tools_enabled:
            return ToolResult(
                "Weather lookup is disabled in offline mode. Enable online tools to use it.",
                success=False,
            )
        query = str(arguments.get("query", "")).strip()
        location = str(arguments.get("location", "")).strip() or self._location_from_query(query)
        if not location:
            return ToolResult("Please specify a city for the weather lookup.", success=False)
        if cancel_event.is_set():
            return ToolResult("Weather lookup cancelled.", success=False)
        if timeout <= 0:
            return ToolResult("Weather lookup timed out.", success=False)
        try:
            geocode = self.transport.get_factual_json(
                _GEOCODING_URL,
                params={"name": location, "count": 1, "language": "en", "format": "json"},
                allowed_hosts=_ALLOWED_HOSTS,
                timeout=max(0.1, timeout / 2),
                policy=self.policy,
            )
            results = geocode.get("results")
            if not isinstance(results, list) or not results:
                return ToolResult(f"No weather location matched '{location}'.", success=False)
            place = results[0]
            latitude = self._number(place, "latitude")
            longitude = self._number(place, "longitude")
            if cancel_event.is_set():
                return ToolResult("Weather lookup cancelled.", success=False)
            forecast = self.transport.get_factual_json(
                _FORECAST_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
                    "forecast_days": 2 if "tomorrow" in query.casefold() else 1,
                    "timezone": "auto",
                },
                allowed_hosts=_ALLOWED_HOSTS,
                timeout=max(0.1, timeout / 2),
                policy=self.policy,
            )
            compact = self._compact_forecast(place, forecast)
            return ToolResult(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
        except Exception as error:
            return ToolResult(f"Weather lookup failed ({type(error).__name__}).", success=False)

    @staticmethod
    def _location_from_query(query: str) -> str:
        match = _LOCATION_PATTERN.search(query)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _number(source: dict[str, Any], key: str) -> float:
        value = source.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError("Invalid geocoding coordinates")
        return float(value)

    @staticmethod
    def _compact_forecast(place: dict[str, Any], forecast: dict[str, Any]) -> dict[str, object]:
        current = forecast.get("current")
        daily = forecast.get("daily")
        if not isinstance(current, dict) or not isinstance(daily, dict):
            raise ValueError("Open-Meteo returned incomplete forecast data")
        location = ", ".join(
            part for part in (str(place.get("name", "")), str(place.get("admin1", "")), str(place.get("country", ""))) if part
        )
        result: dict[str, object] = {
            "location": location,
            "timezone": forecast.get("timezone", place.get("timezone", "")),
            "current": {
                key: current[key]
                for key in (
                    "time",
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                )
                if key in current
            },
        }
        dates = daily.get("time")
        if isinstance(dates, list):
            forecasts = []
            for index, day in enumerate(dates[:2]):
                item: dict[str, object] = {"date": day}
                for key in (
                    "temperature_2m_min",
                    "temperature_2m_max",
                    "precipitation_probability_max",
                    "sunrise",
                    "sunset",
                ):
                    values = daily.get(key)
                    if isinstance(values, list) and index < len(values):
                        item[key] = values[index]
                forecasts.append(item)
            result["daily"] = forecasts
        return result
