
import requests
import logging
import os

class EnvironmentalAdapter:
    def __init__(self):
        # User provided key
        self.api_key = os.getenv("WEATHER_API_KEY", "e345ea59c77641bd98941444262401")
        self.base_url = "http://api.weatherapi.com/v1/current.json"

    def get_weather_context(self, city: str):
        """
        Fetches weather from WeatherAPI.com (Key required).
        """
        try:
            params = {
                "key": self.api_key,
                "q": city,
                "aqi": "no"
            }
            response = requests.get(self.base_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                current = data['current']
                
                temp_c = current['temp_c']
                humidity = current['humidity']
                uv_index = current['uv']
                desc = current['condition']['text']
                
                return {
                    "city": data['location']['name'],
                    "temp_c": temp_c,
                    "humidity": humidity,
                    "uv_index": uv_index,
                    "description": desc,
                    "context_tags": self._derive_tags(temp_c, humidity, uv_index)
                }
            else:
                logging.error(f"WeatherAPI Error: {response.text}")
                return self._fallback_wttr(city) # Fallback if key fails
                
        except Exception as e:
            logging.error(f"Weather fetch failed: {e}")
            return self._fallback_wttr(city)

    def _fallback_wttr(self, city):
        """Original wttr.in implementation as backup"""
        try:
            url = f"https://wttr.in/{city}?format=j1"
            response = requests.get(url, timeout=3)
            data = response.json()
            current = data['current_condition'][0]
            return {
                "city": city,
                "temp_c": int(current.get('temp_C', 25)),
                "humidity": int(current.get('humidity', 50)),
                "uv_index": int(current.get('uvIndex', 0)),
                "description": current['weatherDesc'][0]['value'],
                "context_tags": []
            }
        except:
            return None

    def _derive_tags(self, temp, humidity, uv):
        tags = []
        if uv >= 6: tags.append("high_uv")
        if humidity < 30: tags.append("dry_air")
        elif humidity > 70: tags.append("humid")
        if temp > 30: tags.append("hot")
        elif temp < 10: tags.append("cold")
        return tags

# Singleton
env_adapter = EnvironmentalAdapter()
