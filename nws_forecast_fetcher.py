"""
NWS Point Forecast Fetcher
Pulls the official NWS human-corrected forecast from api.weather.gov.

NWS forecasters manually adjust model output using local knowledge,
satellite imagery, and pattern recognition. This makes NWS forecasts
one of the most reliable single-point sources for US locations.

The forecast is deterministic (single value), so we treat it as a
high-confidence "member" with extra weight in blending.
"""

import logging
from datetime import datetime, timezone

import httpx

import config

logger = logging.getLogger(__name__)

NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
NWS_HEADERS = {"User-Agent": "(pmc-weather-pipeline, contact@example.com)"}


async def _resolve_gridpoint(
    client: httpx.AsyncClient,
) -> tuple[str, int, int] | None:
    """
    Resolve lat/lon to NWS grid office + coordinates.
    GET /points/{lat},{lon} → {properties: {gridId, gridX, gridY}}
    """
    url = NWS_POINTS_URL.format(lat=config.MARKET.station.lat, lon=config.MARKET.station.lon)
    try:
        resp = await client.get(url, headers=NWS_HEADERS)
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        office = props.get("gridId")       # e.g. "FWD" (Fort Worth)
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")
        if office and grid_x is not None and grid_y is not None:
            logger.info(f"NWS gridpoint: {office}/{grid_x},{grid_y}")
            return office, grid_x, grid_y
    except httpx.HTTPError as e:
        logger.warning(f"NWS gridpoint resolution failed: {e}")
    return None


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


async def fetch_nws_forecast() -> dict:
    """
    Fetch the NWS point forecast for the target station.

    Returns:
        {
            "source": "NWS",
            "forecast_temps_f": [max_temp_f],   # single deterministic value
            "cycle": "NWS-forecast",
            "member_details": {"nws_high": temp_f, "nws_periods": [...]},
            "fetch_time": ISO timestamp,
        }
    """
    result = {
        "source": "NWS",
        "forecast_temps_f": [],
        "cycle": None,
        "member_details": {},
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: Resolve gridpoint
        grid = await _resolve_gridpoint(client)
        if not grid:
            logger.error("Could not resolve NWS gridpoint")
            return result

        office, grid_x, grid_y = grid

        # Step 2: Fetch the 7-day forecast
        forecast_url = (
            f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}/forecast"
        )
        try:
            resp = await client.get(forecast_url, headers=NWS_HEADERS)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"NWS forecast fetch failed: {e}")
            return result

        periods = resp.json().get("properties", {}).get("periods", [])
        if not periods:
            logger.warning("NWS forecast returned no periods")
            return result

        # Step 3: Find the daytime period matching our target date
        target_date = config.MARKET.target_date.date()
        target_high_f = None

        for period in periods:
            if not period.get("isDaytime"):
                continue

            # Parse the period start time
            start_str = period.get("startTime", "")
            try:
                period_start = datetime.fromisoformat(start_str)
                period_date = period_start.date()
            except ValueError:
                continue

            if period_date == target_date:
                temp = period.get("temperature")
                unit = period.get("temperatureUnit", "F")
                if temp is not None:
                    # NWS returns F or C; convert to market unit
                    temp_c = float(temp) if unit == "C" else (float(temp) - 32) * 5 / 9
                    target_high_f = config.MARKET.celsius_to_unit(temp_c)
                    result["member_details"]["nws_period"] = {
                        "name": period.get("name"),
                        "temperature": temp,
                        "unit": unit,
                        "shortForecast": period.get("shortForecast"),
                        "detailedForecast": period.get("detailedForecast"),
                    }
                    break

        if target_high_f is not None:
            result["forecast_temps_f"] = [round(target_high_f, 1)]
            result["cycle"] = "NWS-forecast"
            logger.info(
                f"NWS forecast high for {target_date}: {target_high_f:.1f}°F "
                f"({result['member_details'].get('nws_period', {}).get('shortForecast', '')})"
            )
        else:
            logger.warning(f"NWS forecast: no daytime period found for {target_date}")

    # Add detailed summary logging similar to Open-Meteo
    if result["forecast_temps_f"]:
        temps = result["forecast_temps_f"]
        logger.info(
            f"NWS: {len(temps)} forecast retrieved — "
            f"high={temps[0]:.1f}°F"
        )
        logger.info("=== NWS Forecast Results ===")
        nws_period = result.get("member_details", {}).get("nws_period", {})
        logger.info(f"  nws_high: {temps[0]:.1f}°F")
        if nws_period:
            logger.info(f"  period: {nws_period.get('name', 'N/A')}")
            logger.info(f"  shortForecast: {nws_period.get('shortForecast', 'N/A')}")
        logger.info("===========================")

    return result
