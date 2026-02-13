"""
Open-Meteo Multi-Model Forecast Fetcher
Pulls temperature forecasts from multiple NWP models via the free Open-Meteo API.

Open-Meteo aggregates many models (GFS, ECMWF IFS, JMA, ICON, GEM, etc.) and
exposes them through a simple JSON REST API — no API key needed.

We fetch the daily max temperature from multiple models and treat each model's
forecast as one "ensemble member", giving us independent model diversity that
raw ensemble members from a single model can't provide.

Available models for temperature_2m_max:
  - gfs_seamless (NOAA GFS)
  - ecmwf_ifs025 (ECMWF IFS 0.25°)
  - jma_seamless (Japan Meteorological Agency)
  - icon_seamless (DWD ICON)
  - gem_seamless (Canadian GEM)
  - meteofrance_seamless (Météo-France ARPEGE)
  - best_match (Open-Meteo's own blended "best" forecast)
"""

import logging
from datetime import datetime, timezone

import httpx

import config

logger = logging.getLogger(__name__)

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# Models to query — each gives an independent max-temp forecast
OPENMETEO_MODELS = [
    "best_match",
    "gfs_seamless",
    "ecmwf_ifs025",
    "jma_seamless",
    "icon_seamless",
    "gem_seamless",
    "meteofrance_seamless",
]


async def fetch_openmeteo_forecast() -> dict:
    """
    Fetch daily high temperature forecasts from multiple Open-Meteo models.

    Returns:
        {
            "source": "Open-Meteo",
            "forecast_temps_f": [list of max temps in °F, one per model],
            "cycle": "multi-model",
            "member_details": {"model_name": temp_f, ...},
            "fetch_time": ISO timestamp,
        }
    """
    result = {
        "source": "Open-Meteo",
        "forecast_temps_f": [],
        "cycle": None,
        "member_details": {},
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    market = config.MARKET
    target_date_str = market.target_date.strftime("%Y-%m-%d")
    temp_unit = "celsius" if market.unit == "C" else "fahrenheit"

    async with httpx.AsyncClient(timeout=30) as client:
        for model in OPENMETEO_MODELS:
            params = {
                "latitude": market.station.lat,
                "longitude": market.station.lon,
                "daily": "temperature_2m_max",
                "temperature_unit": temp_unit,
                "timezone": market.timezone_str,
                "start_date": target_date_str,
                "end_date": target_date_str,
                "models": model,
            }

            try:
                resp = await client.get(OPENMETEO_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                daily = data.get("daily", {})
                dates = daily.get("time", [])
                maxes = daily.get("temperature_2m_max", [])

                if dates and maxes and maxes[0] is not None:
                    temp_f = round(float(maxes[0]), 1)
                    result["forecast_temps_f"].append(temp_f)
                    result["member_details"][model] = temp_f
                    logger.debug(f"  Open-Meteo {model}: {temp_f}°F")
                else:
                    logger.debug(f"  Open-Meteo {model}: no data for {target_date_str}")

            except httpx.HTTPError as e:
                logger.debug(f"  Open-Meteo {model} failed: {e}")
            except (KeyError, ValueError, IndexError) as e:
                logger.debug(f"  Open-Meteo {model} parse error: {e}")

    if result["forecast_temps_f"]:
        result["cycle"] = "multi-model"
        temps = result["forecast_temps_f"]
        logger.info(
            f"Open-Meteo: {len(temps)} models retrieved — "
            f"min={min(temps):.1f}°F, max={max(temps):.1f}°F, "
            f"mean={sum(temps)/len(temps):.1f}°F"
        )
    else:
        logger.warning("Open-Meteo: no model data retrieved")

    return result
