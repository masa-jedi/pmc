"""
Current Conditions Fetcher
Retrieves live NWS observations to anchor the forecast distribution to reality.
"""

import logging
from datetime import datetime, timedelta

import httpx

from config import MARKET

logger = logging.getLogger("current_conditions")

NWS_OBS_URL = "https://api.weather.gov/stations/{icao}/observations"
STATION_UTC_OFFSET = timedelta(hours=-6)  # Dallas = CST (UTC-6)


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


async def fetch_current_conditions(
    icao: str,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """
    Fetch current temperature and today's observed high from NWS.

    Args:
        icao: Station ICAO code (e.g. "KDAL").
        as_of_utc: If set, only use observations at or before this UTC time.
                   Used by backtest.py to simulate a historical moment.

    Returns:
        {
            "current_temp_f": float,         # most recent reading
            "observed_high_f": float,        # max temp so far on the target local date
            "observed_at": datetime,         # UTC timestamp of the latest observation
        }
        or None if the fetch fails.
    """
    url = NWS_OBS_URL.format(icao=icao)

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"NWS observations fetch failed: {e}")
            return None

    features = resp.json().get("features", [])
    if not features:
        logger.warning("No NWS observations returned")
        return None

    # MARKET.target_date stores the date as the LOCAL calendar date (e.g. Feb 10).
    # Do NOT apply the UTC offset here — the date component is already the local date.
    target_local_date = MARKET.target_date.date()

    daily_temps: list[float] = []
    latest_obs: float | None = None
    latest_time: datetime | None = None

    for feature in features:
        props = feature.get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        ts_str = props.get("timestamp")

        if temp_c is None or ts_str is None:
            continue

        try:
            obs_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        obs_local_date = (obs_time + STATION_UTC_OFFSET).date()
        if obs_local_date != target_local_date:
            continue

        # For backtesting: skip observations after the as-of cutoff
        if as_of_utc is not None and obs_time > as_of_utc:
            continue

        temp_f = round(_celsius_to_fahrenheit(temp_c), 1)
        daily_temps.append(temp_f)

        if latest_time is None or obs_time > latest_time:
            latest_time = obs_time
            latest_obs = temp_f

    if latest_obs is None or latest_time is None:
        logger.warning(f"No observations found for local date {target_local_date}")
        return None

    # Find the time the observed high occurred (needed for peak detection)
    observed_high_f = max(daily_temps)
    observed_high_time: datetime | None = None
    for feature in features:
        props = feature.get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        ts_str = props.get("timestamp")
        if temp_c is None or ts_str is None:
            continue
        try:
            obs_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        obs_local_date = (obs_time + STATION_UTC_OFFSET).date()
        if obs_local_date != target_local_date:
            continue
        if as_of_utc is not None and obs_time > as_of_utc:
            continue
        temp_f = round(_celsius_to_fahrenheit(temp_c), 1)
        if temp_f == observed_high_f and (
            observed_high_time is None or obs_time < observed_high_time
        ):
            observed_high_time = obs_time  # Earliest occurrence of the daily high

    logger.info(f"Latest observation: {latest_obs}°F at {latest_time}")
    logger.info(f"Observed high for local day {target_local_date}: {observed_high_f}°F")

    return {
        "current_temp_f": latest_obs,
        "observed_high_f": observed_high_f,
        "observed_high_time": observed_high_time,  # When the daily high was recorded
        "observed_at": latest_time,
    }
