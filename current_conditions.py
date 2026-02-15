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
# Weather.com API for international stations (used for METAR data)
WEATHER_COM_INTL_URL = "https://api.weather.com/v1/location/{icao}:9:{country}/observations/current.json"


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9


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


async def _fetch_metar_conditions(
    icao: str,
    target_local_date,
    utc_offset: timedelta,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """
    Fetch current conditions from Weather.com API for international METAR stations.

    Returns same format as NWS: {current_temp_f, observed_high_f, observed_at}.
    Uses Weather.com which provides METAR data for international stations.
    """
    from datetime import timezone as tz

    # Determine country code from ICAO prefix
    # RKSI -> KR (Korea)
    country_map = {
        "R": "KR",  # Korea
        "Z": "CN",  # China
        "V": "TH",  # Thailand
        "WS": "SG", # Singapore
        "WB": "MY", # Malaysia
        "YM": "AU", # Australia
        "NZ": "NZ", # New Zealand
        "EG": "EG", # Egypt
        "L": "FR",  # France/Germany/etc (Europe)
        "E": "SE",  # Scandinavia
        "K": "US",  # US (handled elsewhere)
    }

    # Get country code from first 1-2 characters of ICAO
    country = "US"
    if len(icao) >= 1:
        prefix = icao[:1]
        if prefix in country_map:
            country = country_map[prefix]
        elif len(icao) >= 2:
            prefix2 = icao[:2]
            if prefix2 in country_map:
                country = country_map[prefix2]

    url = WEATHER_COM_INTL_URL.format(icao=icao, country=country)
    url_with_key = f"{url}?apiKey={config.WEATHER_COM_API_KEY}"

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url_with_key)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"METAR (Weather.com) fetch failed for {icao}: {e}")
            return None

    data = resp.json()
    observation = data.get("observation", {})
    # Temperature is in imperial object for international stations (Fahrenheit)
    imperial = observation.get("imperial", {})
    temp_f = imperial.get("temp")
    # Get the observed high for the day (max temp in last 24 hours)
    observed_high_f = imperial.get("temp_max_24hour")

    if temp_f is None:
        logger.warning("No temperature in Weather.com METAR response")
        return None

    # Convert to market's temperature unit
    temp_c = _fahrenheit_to_celsius(temp_f)
    temp_value = round(config.MARKET.celsius_to_unit(temp_c), 1)

    if observed_high_f is not None:
        observed_high_c = _fahrenheit_to_celsius(observed_high_f)
        observed_high_value = round(config.MARKET.celsius_to_unit(observed_high_c), 1)
    else:
        observed_high_value = None

    # Get the timestamp
    obs_time_gmt = observation.get("obs_time")
    if obs_time_gmt:
        try:
            obs_time = datetime.fromtimestamp(obs_time_gmt, tz=tz.utc)
        except ValueError:
            obs_time = datetime.now(tz.utc)
    else:
        obs_time = datetime.now(tz.utc)

    # Check if this observation is for the target local date
    obs_local_date = (obs_time + utc_offset).date()
    if obs_local_date != target_local_date:
        logger.warning(f"METAR observation {obs_time} is not for target date {target_local_date}")
        return None

    # For backtesting: skip observations after the as-of cutoff
    if as_of_utc is not None and obs_time > as_of_utc:
        logger.warning(f"METAR observation {obs_time} is after as_of_utc {as_of_utc}")
        return None

    u = config.MARKET.unit_symbol
    logger.info(f"METAR (Weather.com): latest observation: {temp_value}{u} at {obs_time} UTC for {icao}")

    return {
        "current_temp_f": temp_value,
        "observed_high_f": observed_high_value if observed_high_value is not None else temp_value,
        "observed_high_time": obs_time,
        "observed_at": obs_time,
        "source": "metar",
    }


async def fetch_current_conditions(
    icao: str,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """
    Fetch current temperature and today's observed high from NWS (US),
    METAR (international), or Open-Meteo (global fallback).

    Args:
        icao: Station ICAO code (e.g. "KDAL", "RKSI").
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
    market = config.MARKET

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
            logger.info(f"NWS observations fetch failed ({e}), trying METAR...")

    # Try METAR for international stations (e.g., RKSI for Korea)
    # METAR API supports stations worldwide via ICAO codes
    metar_result = await _fetch_metar_conditions(
        icao=icao,
        target_local_date=market.target_date.date(),
        utc_offset=market.utc_offset,
        as_of_utc=as_of_utc,
    )

    if metar_result:
        logger.info(f"Using METAR for current conditions: {icao}")
        return metar_result
    else:
        logger.info(f"METAR fetch failed or unavailable for {icao}, falling back to Open-Meteo...")

    # Fallback to Open-Meteo for non-US stations (Korea, etc.)
    # First try to get historical hourly data to determine observed high

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


# Weather.com historical observations URL template
WEATHER_COM_URL = "https://api.weather.com/v1/location/{icao}:9:US/observations/historical.json"


def _parse_precip_intensity(wx_phrase: str) -> float:
    """
    Parse precipitation intensity from wx_phrase.

    Returns intensity from 0.0-1.0:
    - "Light" -> 0.3
    - "Heavy" -> 1.0
    - "Moderate" or no prefix -> 0.6
    - "Thunder" / "T-Storm" -> 0.8 (storms are significant)
    """
    phrase_lower = wx_phrase.lower() if wx_phrase else ""

    if "thunder" in phrase_lower or "t-storm" in phrase_lower:
        return 0.8
    if "heavy" in phrase_lower:
        return 1.0
    if "light" in phrase_lower:
        return 0.3
    # Check for any precipitation keywords
    if "rain" in phrase_lower or "storm" in phrase_lower or "drizzle" in phrase_lower:
        return 0.6
    return 0.0


def _is_precipitation(wx_phrase: str) -> bool:
    """Check if wx_phrase indicates precipitation (rain, thunder, storm)."""
    if not wx_phrase:
        return False
    phrase_lower = wx_phrase.lower()
    precip_keywords = ["rain", "thunder", "t-storm", "storm", "drizzle", "showers", "precip"]
    return any(keyword in phrase_lower for keyword in precip_keywords)


async def fetch_weather_conditions(
    icao: str,
    target_date,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """
    Fetch weather conditions (precipitation, thunder, storm) for the target date.

    For US stations (ICAO format like KDAL): Uses weather.com historical observations API.
    For non-US stations: Falls back to Open-Meteo precipitation data.

    Returns:
        {
            "has_rain": bool,
            "has_thunder": bool,
            "has_storm": bool,
            "is_active_precipitation": bool,  # True if latest obs has rain/thunder
            "latest_phrase": str,
            "precip_intensity": float,  # 0.0-1.0 based on keywords
            "source": str,  # "weathercom" or "openmeteo"
        }
        or None if the fetch fails.
    """
    market = config.MARKET

    # Check if this is a US station (ICAO format: 3-4 letter code)
    # US ICAOs start with K for continental US
    is_us_station = icao.startswith("K") and len(icao) == 4

    if is_us_station:
        return await _fetch_weathercom_conditions(icao, target_date, as_of_utc)
    else:
        # Non-US station: use Open-Meteo precipitation
        return await _fetch_openmeteo_precip(icao, target_date, as_of_utc)


async def _fetch_weathercom_conditions(
    icao: str,
    target_date,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """Fetch weather conditions from weather.com API for US stations."""
    from datetime import timezone as tz

    # Format date for API (YYYYMMDD)
    date_str = target_date.strftime("%Y%m%d")

    url = WEATHER_COM_URL.format(icao=icao) + f"?apiKey={config.WEATHER_COM_API_KEY}&units=e&startDate={date_str}&endDate={date_str}"

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"Weather.com fetch failed: {e}")
            return None

    data = resp.json()
    observations = data.get("observations", [])

    if not observations:
        logger.warning("No observations in weather.com response")
        return None

    # Analyze all observations for the day
    has_rain = False
    has_thunder = False
    has_storm = False
    latest_phrase = None
    latest_obs_time = None

    for obs in observations:
        wx_phrase = obs.get("wx_phrase")
        valid_time_gmt = obs.get("valid_time_gmt")

        if not wx_phrase:
            continue

        # Track latest observation
        if valid_time_gmt:
            obs_time = datetime.fromtimestamp(valid_time_gmt, tz=tz.utc)
            # Skip observations after as_of_utc for backtesting
            if as_of_utc and obs_time > as_of_utc:
                continue
            if latest_obs_time is None or obs_time > latest_obs_time:
                latest_obs_time = obs_time
                latest_phrase = wx_phrase

        # Check for precipitation types
        phrase_lower = wx_phrase.lower()
        if "rain" in phrase_lower or "drizzle" in phrase_lower or "showers" in phrase_lower:
            has_rain = True
        if "thunder" in phrase_lower or "t-storm" in phrase_lower:
            has_thunder = True
        if "storm" in phrase_lower:
            has_storm = True

    # Determine if precipitation is active (in the latest observation)
    is_active = _is_precipitation(latest_phrase)
    precip_intensity = _parse_precip_intensity(latest_phrase) if latest_phrase else 0.0

    logger.info(
        f"Weather.com conditions for {icao}: phrase='{latest_phrase}', "
        f"rain={has_rain}, thunder={has_thunder}, storm={has_storm}, "
        f"active={is_active}, intensity={precip_intensity}"
    )

    return {
        "has_rain": has_rain,
        "has_thunder": has_thunder,
        "has_storm": has_storm,
        "is_active_precipitation": is_active,
        "latest_phrase": latest_phrase,
        "precip_intensity": precip_intensity,
        "source": "weathercom",
    }


async def _fetch_openmeteo_precip(
    icao: str,
    target_date,
    as_of_utc: datetime | None = None,
) -> dict | None:
    """Fetch precipitation data from Open-Meteo for non-US stations."""
    market = config.MARKET

    # Use Open-Meteo current weather to get precipitation
    url = (
        f"{OPENMETEO_CURR_URL}"
        f"?latitude={market.station.lat}&longitude={market.station.lon}"
        f"&current=precipitation,rain,showers,snowfall"
        f"&timezone=auto"
    )

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"Open-Meteo precipitation fetch failed: {e}")
            return None

    data = resp.json()
    current = data.get("current", {})
    precip = current.get("precipitation", 0.0)
    rain = current.get("rain", 0.0)
    showers = current.get("showers", 0.0)
    snowfall = current.get("snowfall", 0.0)

    # Any precipitation > 0 counts as rain
    has_precip = precip > 0 or rain > 0 or showers > 0 or snowfall > 0

    # Simple intensity based on amount (scale mm to 0-1)
    if has_precip:
        precip_intensity = min(1.0, (precip + rain + showers) / 5.0)
    else:
        precip_intensity = 0.0

    # For non-US stations, we don't have separate thunder/storm detection
    # Just indicate if there's any precipitation
    logger.info(
        f"Open-Meteo precip for {icao}: precip={precip}mm, "
        f"has_precip={has_precip}, intensity={precip_intensity}"
    )

    return {
        "has_rain": has_precip,
        "has_thunder": False,  # Not available from Open-Meteo current
        "has_storm": False,     # Not available from Open-Meteo current
        "is_active_precipitation": has_precip,
        "latest_phrase": f"Precipitation {precip}mm" if has_precip else "Clear",
        "precip_intensity": precip_intensity,
        "source": "openmeteo",
    }
