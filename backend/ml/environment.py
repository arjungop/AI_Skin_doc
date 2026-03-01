"""
Environmental data adapter — fetches UV index and weather for a city.

Used by the UV risk assessment endpoint. Supports WeatherAPI.com with
wttr.in as fallback.
"""

import requests
import logging
import os

logger = logging.getLogger(__name__)


class EnvironmentalAdapter:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY", "")
        self.base_url = "http://api.weatherapi.com/v1/current.json"

    def get_weather_context(self, city: str) -> dict | None:
        """Fetch current weather for a city. Returns None on failure."""
        if not city or not city.strip():
            return None

        # Try WeatherAPI first if key is configured
        if self.api_key:
            result = self._fetch_weatherapi(city)
            if result:
                return result

        # Fallback to free wttr.in
        return self._fetch_wttr(city)

    def _fetch_weatherapi(self, city: str) -> dict | None:
        try:
            params = {"key": self.api_key, "q": city, "aqi": "no"}
            response = requests.get(self.base_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                current = data["current"]
                return {
                    "city": data["location"]["name"],
                    "temp_c": current["temp_c"],
                    "humidity": current["humidity"],
                    "uv_index": current["uv"],
                    "description": current["condition"]["text"],
                    "context_tags": self._derive_tags(
                        current["temp_c"], current["humidity"], current["uv"]
                    ),
                }
            else:
                logger.warning("WeatherAPI returned %s: %s", response.status_code, response.text[:200])
                return None
        except Exception as e:
            logger.warning("WeatherAPI fetch failed: %s", e)
            return None

    def _fetch_wttr(self, city: str) -> dict | None:
        """Free fallback using wttr.in (no API key required)."""
        try:
            url = f"https://wttr.in/{city}?format=j1"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None

            data = response.json()
            current = data["current_condition"][0]
            temp = int(current.get("temp_C", 25))
            humidity = int(current.get("humidity", 50))
            uv = int(current.get("uvIndex", 0))

            return {
                "city": city,
                "temp_c": temp,
                "humidity": humidity,
                "uv_index": uv,
                "description": current["weatherDesc"][0]["value"],
                "context_tags": self._derive_tags(temp, humidity, uv),
            }
        except Exception as e:
            logger.warning("wttr.in fetch failed: %s", e)
            return None

    @staticmethod
    def _derive_tags(temp: float, humidity: float, uv: float) -> list[str]:
        tags = []
        if uv >= 8:
            tags.append("extreme_uv")
        elif uv >= 6:
            tags.append("high_uv")
        if humidity < 30:
            tags.append("dry_air")
        elif humidity > 70:
            tags.append("humid")
        if temp > 35:
            tags.append("very_hot")
        elif temp > 30:
            tags.append("hot")
        elif temp < 10:
            tags.append("cold")
        return tags


# Singleton
env_adapter = EnvironmentalAdapter()
