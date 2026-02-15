"""
NOAA HRRR (High-Resolution Rapid Refresh) Data Fetcher
Pulls 2m temperature forecasts from HRRR via NOMADS GRIB filter.

HRRR is a 3km CONUS model that runs every hour (00Z-23Z).
It's deterministic (no ensemble), so we fetch the last N hourly cycles
and treat each cycle's max-temp forecast as one "member" to capture
forecast uncertainty as the model evolves through the day.

Forecast hours: 0-18 for most cycles, 0-48 for 00Z and 12Z.
Data appears ~30-45 min after the cycle time on NOMADS.
"""

import logging
import struct
from collections import Counter
from datetime import datetime, timedelta, timezone

import httpx

import config
from config import (
    HRRR_BASE_URL,
    HRRR_N_CYCLES,
    HTTP_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)


def kelvin_to_fahrenheit(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32


def _find_latest_hrrr_cycles(
    client: httpx.Client,
    n_cycles: int = HRRR_N_CYCLES,
) -> list[tuple[str, str]]:
    """
    Find the N most recent available HRRR cycles on NOMADS.
    HRRR data typically appears 30-45 minutes after the cycle time.

    Returns:
        List of (date_str, cycle_hour) tuples, most recent first.
        E.g. [("20260210", "17"), ("20260210", "16"), ...]
    """
    now = datetime.now(timezone.utc)
    found = []

    for hours_back in range(1, 13):  # start at 1 to skip current (not yet available)
        check_time = now - timedelta(hours=hours_back)
        date_str = check_time.strftime("%Y%m%d")
        cycle = f"{check_time.hour:02d}"

        test_url = (
            f"{HRRR_BASE_URL}?"
            f"dir=%2Fhrrr.{date_str}%2Fconus"
            f"&file=hrrr.t{cycle}z.wrfsfcf01.grib2"
            f"&var_TMP=on"
            f"&lev_2_m_above_ground=on"
            f"&subregion="
            f"&toplat={config.MARKET.station.lat + 0.25}"
            f"&leftlon={config.MARKET.station.lon % 360}"
            f"&rightlon={(config.MARKET.station.lon % 360) + 0.25}"
            f"&bottomlat={config.MARKET.station.lat - 0.25}"
        )
        try:
            resp = client.head(test_url, timeout=15)
            if resp.status_code == 200:
                found.append((date_str, cycle))
                if len(found) >= n_cycles:
                    break
        except httpx.HTTPError:
            continue

    return found


def _compute_hrrr_hours(cycle_date: str, cycle_hour: str) -> list[int]:
    """
    Compute which HRRR forecast hours cover the target date's daytime.
    HRRR has hourly output: 0-18 for most cycles, 0-48 for 00Z and 12Z.
    """
    cycle_dt = datetime.strptime(f"{cycle_date}{cycle_hour}", "%Y%m%d%H").replace(
        tzinfo=timezone.utc
    )
    target = config.MARKET.target_date

    # Cover 6 AM CST (12Z) to midnight CST (06Z next day)
    target_start = target.replace(hour=12)
    target_end = target + timedelta(hours=30)

    max_fh = 48 if cycle_hour in ("00", "12") else 18

    hours = []
    for fh in range(0, max_fh + 1):
        valid_time = cycle_dt + timedelta(hours=fh)
        if target_start <= valid_time <= target_end:
            hours.append(fh)

    return hours


def _parse_grib2_temperature(
    data: bytes, expected_lat: float, expected_lon: float
) -> float | None:
    """
    Lightweight GRIB2 temperature extraction.
    Since we request a tiny subregion from NOMADS, the data contains
    very few grid points. We extract the 2m temperature value.

    Same heuristic as gfs_fetcher.py — scan for IEEE 754 floats in Kelvin range.
    """
    if len(data) < 100:
        return None

    try:
        grib_start = data.find(b"GRIB")
        if grib_start == -1:
            return None

        candidates = []
        for i in range(grib_start, len(data) - 4):
            try:
                val = struct.unpack(">f", data[i : i + 4])[0]
                if 220.0 <= val <= 330.0:
                    candidates.append(val)
            except struct.error:
                continue

        if candidates:
            most_common = Counter(candidates).most_common(1)[0][0]
            return most_common

    except Exception as e:
        logger.debug(f"GRIB2 parse heuristic failed: {e}")

    return None


async def fetch_hrrr_forecast(
    n_cycles: int = HRRR_N_CYCLES,
    pin_cycles: list[tuple[str, str]] | None = None,
) -> dict:
    """
    Fetch HRRR forecasts from the last N hourly cycles.

    Each cycle produces one max-temp estimate for the target day.
    Returns them as a pseudo-ensemble in the same shape as gfs_fetcher output.

    Args:
        n_cycles: Number of recent hourly cycles to fetch.
        pin_cycles: Optional list of (date_str, cycle_hour) to use instead of
                    auto-discovering. Used by backtest.py.

    Returns:
        {
            "source": "HRRR",
            "cycle": "20260210/17Z",  (most recent cycle used)
            "forecast_temps_f": [t1, t2, ...],  (one per cycle)
            "member_details": {"20260210/17Z": {fh: temp_f, ...}, ...},
            "fetch_time": ISO timestamp,
        }
    """
    result = {
        "source": "HRRR",
        "cycle": None,
        "forecast_temps_f": [],
        "member_details": {},
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    if pin_cycles is not None:
        cycles = pin_cycles
    else:
        with httpx.Client() as sync_client:
            cycles = _find_latest_hrrr_cycles(sync_client, n_cycles)

    if not cycles:
        logger.error("No available HRRR cycles found")
        return result

    result["cycle"] = f"{cycles[0][0]}/{cycles[0][1]}Z"
    logger.info(
        f"Fetching HRRR from {len(cycles)} cycles: "
        + ", ".join(f"{d}/{h}Z" for d, h in cycles)
    )

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for date_str, cycle in cycles:
            cycle_label = f"{date_str}/{cycle}Z"
            forecast_hours = _compute_hrrr_hours(date_str, cycle)
            if not forecast_hours:
                logger.warning(f"  HRRR {cycle_label}: no forecast hours cover target date")
                continue

            cycle_temps = {}

            for fh in forecast_hours:
                url = (
                    f"{HRRR_BASE_URL}?"
                    f"dir=%2Fhrrr.{date_str}%2Fconus"
                    f"&file=hrrr.t{cycle}z.wrfsfcf{fh:02d}.grib2"
                    f"&var_TMP=on"
                    f"&lev_2_m_above_ground=on"
                    f"&subregion="
                    f"&toplat={config.MARKET.station.lat + 0.25}"
                    f"&leftlon={config.MARKET.station.lon % 360}"
                    f"&rightlon={(config.MARKET.station.lon % 360) + 0.25}"
                    f"&bottomlat={config.MARKET.station.lat - 0.25}"
                )

                for attempt in range(MAX_RETRIES):
                    try:
                        resp = await client.get(url)
                        if resp.status_code == 200 and len(resp.content) > 50:
                            temp_k = _parse_grib2_temperature(
                                resp.content,
                                config.MARKET.station.lat,
                                config.MARKET.station.lon,
                            )
                            if temp_k:
                                temp_f = config.MARKET.kelvin_to_unit(temp_k)
                                cycle_temps[fh] = round(temp_f, 1)
                                logger.debug(
                                    f"  HRRR {cycle_label} f{fh:02d}: {temp_f:.1f}°F"
                                )
                            break
                        elif resp.status_code == 404:
                            logger.debug(f"  HRRR {cycle_label} f{fh:02d}: not available (404)")
                            break
                    except httpx.HTTPError as e:
                        if attempt < MAX_RETRIES - 1:
                            import asyncio
                            await asyncio.sleep(RETRY_DELAY_SECONDS)
                        else:
                            logger.warning(
                                f"  HRRR {cycle_label} f{fh:02d}: failed after {MAX_RETRIES} retries: {e}"
                            )

            if cycle_temps:
                max_temp = max(cycle_temps.values())
                result["member_details"][cycle_label] = cycle_temps
                result["forecast_temps_f"].append(max_temp)
                logger.info(
                    f"  HRRR {cycle_label}: max temp = {max_temp:.1f}°F "
                    f"({len(cycle_temps)} hours)"
                )
            else:
                logger.warning(f"  HRRR {cycle_label}: no data retrieved")

    logger.info(
        f"HRRR fetch complete: {len(result['forecast_temps_f'])}/{len(cycles)} cycles retrieved"
    )
    return result
