"""
Current Conditions Fetcher
Retrieves live NWS observations (US) or Open-Meteo (global) to anchor the
forecast distribution to reality.
"""

import logging
from datetime import datetime, timedelta

import httpx

import config

logger = logging.getLogger("current_conditions")

NWS_OBS_URL = "https://api.weather.gov/stations/{icao}/observations"
OPENMETEO_CURR_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


async def _fetch_openmeteo_current(
    lat: float,
    lon: float,
    target_local_date,
    utc_offset: timedelta,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """
    Fetch current conditions from Open-Meteo API (global coverage).

    Returns same format as NWS: {current_temp_f, observed_high_f, observed_at}.
    For simplicity, we use current temp as both current and observed high since
    Open-Meteo doesn't provide historical hourly data in the free tier.
    """
    from datetime import timezone as tz

    url = (
        f"{OPENMETEO_CURR_URL}"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m"
        f"&timezone=auto"
    )

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"Open-Meteo current weather fetch failed: {e}")
            return None

    data = resp.json()
    current = data.get("current", {})
    temp_c = current.get("temperature_2m")

    if temp_c is None:
        logger.warning("No temperature in Open-Meteo response")
        return None

    # Convert to market unit
    temp_f = round(config.MARKET.celsius_to_unit(temp_c), 1)

    # Get the timestamp - Open-Meteo returns local time
    time_str = current.get("time")
    if time_str:
        try:
            # Parse the ISO timestamp and convert to UTC
            obs_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            # Convert from whatever timezone it's in to UTC
            obs_time = obs_time.astimezone(tz.utc)
        except ValueError:
            obs_time = datetime.now(tz.utc)
    else:
        obs_time = datetime.now(tz.utc)

    # Open-Meteo doesn't give us hourly history, so we can't determine
    # the observed high. Return None for observed_high_f - the probability
    # engine will then skip the reality check (impossibility filtering).
    logger.info(f"Open-Meteo: latest observation: {temp_f}{config.MARKET.unit_symbol} at {obs_time} UTC (no hourly history)")

    return {
        "current_temp_f": temp_f,
        "observed_high_f": None,  # Unknown - no hourly history
        "observed_high_time": None,
        "observed_at": obs_time,
        "source": "openmeteo",
    }


async def _fetch_openmeteo_historical(
    lat: float,
    lon: float,
    target_local_date,
    utc_offset: timedelta,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """
    Fetch hourly historical temperature data from Open-Meteo Archive API
    to determine today's observed high.

    Returns same format as NWS: {current_temp_f, observed_high_f, observed_high_time, observed_at}.
    """
    from datetime import timezone as tz

    # Format dates for API (YYYY-MM-DD)
    start_date = target_local_date.strftime("%Y-%m-%d")
    end_date = target_local_date.strftime("%Y-%m-%d")

    url = (
        f"{OPENMETEO_ARCHIVE_URL}"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m"
        f"&timezone=auto"
    )

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"Open-Meteo archive fetch failed: {e}")
            return None

    data = resp.json()
    hourly_data = data.get("hourly", {})
    times = hourly_data.get("time", [])
    temps = hourly_data.get("temperature_2m", [])

    if not times or not temps:
        logger.warning("No hourly data in Open-Meteo archive response")
        return None

    # Find observations for the target local date
    daily_temps: list[float] = []
    latest_obs: float | None = None
    latest_time: datetime | None = None

    for i, (time_str, temp_c) in enumerate(zip(times, temps)):
        if temp_c is None:
            continue
        try:
            obs_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        # Open-Meteo returns local times (no timezone info) when timezone=auto
        # Convert to UTC and make timezone-aware
        obs_time = obs_time - utc_offset
        obs_time = obs_time.replace(tzinfo=tz.utc)

        # Convert to local date using the station's UTC offset
        obs_local_date = (obs_time + utc_offset).date()
        if obs_local_date != target_local_date:
            continue

        # For backtesting: skip observations after the as-of cutoff
        if as_of_utc is not None and obs_time > as_of_utc:
            continue

        temp_f = round(config.MARKET.celsius_to_unit(temp_c), 1)
        daily_temps.append(temp_f)

        if latest_time is None or obs_time > latest_time:
            latest_time = obs_time
            latest_obs = temp_f

    if not daily_temps:
        logger.warning(f"No observations found for local date {target_local_date}")
        return None

    # Find the observed high and when it occurred
    observed_high_f = max(daily_temps)
    observed_high_time: datetime | None = None
    for i, (time_str, temp_c) in enumerate(zip(times, temps)):
        if temp_c is None:
            continue
        try:
            obs_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        # Open-Meteo returns local times (no timezone info) when timezone=auto
        # Convert to UTC and make timezone-aware
        obs_time = obs_time - utc_offset
        obs_time = obs_time.replace(tzinfo=tz.utc)

        obs_local_date = (obs_time + utc_offset).date()
        if obs_local_date != target_local_date:
            continue
        if as_of_utc is not None and obs_time > as_of_utc:
            continue
        temp_f = round(config.MARKET.celsius_to_unit(temp_c), 1)
        if temp_f == observed_high_f and (
            observed_high_time is None or obs_time < observed_high_time
        ):
            observed_high_time = obs_time  # Earliest occurrence of the daily high

    u = config.MARKET.unit_symbol
    logger.info(f"Open-Meteo Archive: latest observation: {latest_obs}{u} at {latest_time}")
    logger.info(f"Open-Meteo Archive: observed high for local day {target_local_date}: {observed_high_f}{u}")

    return {
        "current_temp_f": latest_obs,
        "observed_high_f": observed_high_f,
        "observed_high_time": observed_high_time,
        "observed_at": latest_time,
        "source": "openmeteo-archive",
    }


async def fetch_current_conditions(
    icao: str,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """
    Fetch current temperature and today's observed high from NWS (US) or
    Open-Meteo (global fallback).

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
    # Try NWS first (US stations only)
    url = NWS_OBS_URL.format(icao=icao)

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if features:
                return await _process_nws_observations(
                    features, as_of_utc
                )
        except httpx.HTTPError as e:
            logger.info(f"NWS observations fetch failed ({e}), trying Open-Meteo...")

    # Fallback to Open-Meteo for non-US stations (Korea, etc.)
    # First try to get historical hourly data to determine observed high
    market = config.MARKET

    # Try archive API first to get observed_high_f
    result = await _fetch_openmeteo_historical(
        lat=market.station.lat,
        lon=market.station.lon,
        target_local_date=market.target_date.date(),
        utc_offset=market.utc_offset,
        as_of_utc=as_of_utc,
    )

    if result:
        logger.info(f"Using Open-Meteo Archive for current conditions (with observed high)")
    else:
        # Fallback to simple current weather if archive fails
        result = await _fetch_openmeteo_current(
            lat=market.station.lat,
            lon=market.station.lon,
            target_local_date=market.target_date.date(),
            utc_offset=market.utc_offset,
            as_of_utc=as_of_utc,
        )
        if result:
            logger.info(f"Using Open-Meteo current weather (archive unavailable)")

    return result


async def _process_nws_observations(
    features: list,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """Process NWS observation features into the standard format."""
    if not features:
        logger.warning("No NWS observations returned")
        return None

    # config.MARKET.target_date stores the date as the LOCAL calendar date (e.g. Feb 10).
    # Do NOT apply the UTC offset here — the date component is already the local date.
    target_local_date = config.MARKET.target_date.date()

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

        obs_local_date = (obs_time + config.MARKET.utc_offset).date()
        if obs_local_date != target_local_date:
            continue

        # For backtesting: skip observations after the as-of cutoff
        if as_of_utc is not None and obs_time > as_of_utc:
            continue

        temp_f = round(config.MARKET.celsius_to_unit(temp_c), 1)
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
        obs_local_date = (obs_time + config.MARKET.utc_offset).date()
        if obs_local_date != target_local_date:
            continue
        if as_of_utc is not None and obs_time > as_of_utc:
            continue
        temp_f = round(config.MARKET.celsius_to_unit(temp_c), 1)
        if temp_f == observed_high_f and (
            observed_high_time is None or obs_time < observed_high_time
        ):
            observed_high_time = obs_time  # Earliest occurrence of the daily high

    u = config.MARKET.unit_symbol
    logger.info(f"Latest observation: {latest_obs}{u} at {latest_time}")
    logger.info(f"Observed high for local day {target_local_date}: {observed_high_f}{u}")

    return {
        "current_temp_f": latest_obs,
        "observed_high_f": observed_high_f,
        "observed_high_time": observed_high_time,  # When the daily high was recorded
        "observed_at": latest_time,
        "source": "nws",
    }
