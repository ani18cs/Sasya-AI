import os
import requests

# Approximate coordinates for key Karnataka districts
KARNATAKA_DISTRICTS = {
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Mysuru": {"lat": 12.2958, "lon": 76.6394},
    "Hubballi-Dharwad": {"lat": 15.3647, "lon": 75.1240},
    "Belagavi": {"lat": 15.8497, "lon": 74.4977},
    "Kodagu": {"lat": 12.3375, "lon": 75.8069},
    "Chikmagalur": {"lat": 13.3161, "lon": 75.7720},
    "Shivamogga": {"lat": 13.9299, "lon": 75.5681},
    "Davangere": {"lat": 14.4644, "lon": 75.9218},
    "Ballari": {"lat": 15.1394, "lon": 76.9214},
    "Kalaburagi": {"lat": 17.3297, "lon": 76.8343},
}

def is_spraying_advised(advisory_text: str) -> bool:
    """Check if the advisory text mentions spraying or fungicides."""
    text_lower = advisory_text.lower()
    keywords = ["spray", "fungicide", "pesticide", "insecticide"]
    return any(keyword in text_lower for keyword in keywords)

def get_spray_advisory_note(district: str, advisory_text: str) -> str:
    """
    Checks if spraying is advised and if rain is forecast in the next 24 hours.
    Returns a warning string if both conditions are met, otherwise returns None.
    """
    if not is_spraying_advised(advisory_text):
        return None

    if not district or not district.strip():
        return None

    # Check for mock/offline mode
    if os.environ.get("MOCK_WEATHER", "false").lower() == "true":
        # Simulate rain for testing purposes
        return "Note: Heavy rain is forecast in your district in the next 24 hours. Please delay spraying to prevent the treatment from washing away."

    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return None

    if district in KARNATAKA_DISTRICTS:
        coords = KARNATAKA_DISTRICTS[district]
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={coords['lat']}&lon={coords['lon']}&appid={api_key}"
    else:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={district}&appid={api_key}"


    try:
        # Use a short timeout so weather never blocks the main diagnosis flow
        response = requests.get(url, timeout=3.0)
        response.raise_for_status()
        data = response.json()

        # The API returns data in 3-hour chunks. 8 chunks = 24 hours.
        forecast_list = data.get("list", [])[:8]
        
        rain_forecast = False
        for forecast in forecast_list:
            # OpenWeatherMap weather conditions: id < 600 or "Rain"/"Thunderstorm"/"Drizzle" indicates rain
            weather_data = forecast.get("weather", [{}])[0]
            main_weather = weather_data.get("main", "").lower()
            if main_weather in ["rain", "drizzle", "thunderstorm"]:
                rain_forecast = True
                break

        if rain_forecast:
            return "Note: Heavy rain is forecast in your district in the next 24 hours. Please delay spraying to prevent the treatment from washing away."

    except Exception:
        # Silently fail on network issues, timeouts, or bad API keys
        return None

    return None
