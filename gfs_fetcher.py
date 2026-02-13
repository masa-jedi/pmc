"""
NOAA GFS Ensemble (GEFS) Data Fetcher
Pulls 2m temperature forecasts from all 21 ensemble members via NOMADS GRIB filter.

The GEFS provides 21 forecast runs (1 control + 20 perturbed) every 6 hours.
Each member gives a slightly different temperature forecast, which we use
to build a probability distribution.
"""

import logging
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

from config import (
    DATA_DIR,
    GEFS_ALL_MEMBERS,
    GEFS_BASE_URL,
    GEFS_CYCLES,
    GEFS_LEVEL,
    GEFS_VARIABLE,
    HTTP_TIMEOUT_SECONDS,
    MARKET,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)


def kelvin_to_fahrenheit(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32


def _find_latest_available_cycle(client: httpx.Client) -> tuple[str, str] | None:
    """
    Find the most recent GEFS cycle that's actually available on NOMADS.
    GEFS data typically appears 4-6 hours after the cycle time.
    """
    now = datetime.now(timezone.utc)

    # Check last 24 hours of cycles
    for hours_back in range(0, 25, 6):
        check_time = now - timedelta(hours=hours_back)
        date_str = check_time.strftime("%Y%m%d")

        for cycle in sorted(GEFS_CYCLES, reverse=True):
            cycle_hour = int(cycle)
            cycle_time = check_time.replace(
                hour=cycle_hour, minute=0, second=0, microsecond=0
            )
            if cycle_time > now:
                continue

            # Quick availability check — try to access the control member directory
            test_url = (
                f"{GEFS_BASE_URL}?"
                f"dir=%2Fgefs.{date_str}%2F{cycle}%2Fatmos%2Fpgrb2sp25"
                f"&file=gec00.t{cycle}z.pgrb2s.0p25.f006"
                f"&var_{GEFS_VARIABLE}=on"
                f"&lev_{GEFS_LEVEL}=on"
                f"&subregion="
                f"&toplat={MARKET.station.lat + 0.5}"
                f"&leftlon={360 + MARKET.station.lon}"  # NOMADS uses 0-360
                f"&rightlon={360 + MARKET.station.lon + 0.5}"
                f"&bottomlat={MARKET.station.lat - 0.5}"
            )
            try:
                resp = client.head(test_url, timeout=15)
                if resp.status_code == 200:
                    logger.info(f"Found available GEFS cycle: {date_str}/{cycle}Z")
                    return date_str, cycle
            except httpx.HTTPError:
                continue

    return None


def _compute_forecast_hours(cycle_date: str, cycle_hour: str) -> list[int]:
    """
    Compute which forecast hours cover the target date's daytime (for max temp).
    Dallas local = UTC-6. We want to cover ~06:00 to 23:59 local = 12Z to 05Z+1.
    """
    cycle_dt = datetime.strptime(f"{cycle_date}{cycle_hour}", "%Y%m%d%H").replace(
        tzinfo=timezone.utc
    )
    target = MARKET.target_date

    # We want forecast valid times covering Feb 12 12:00 UTC through Feb 13 06:00 UTC
    # (that's Feb 12 06:00 CST through Feb 12 23:59 CST — when max temp occurs)
    target_start = target.replace(hour=12)  # 12Z = 6AM CST
    target_end = target.replace(hour=23, minute=59)  # ~6PM CST is usually max
    # Extend to capture late afternoon
    target_end_utc = target + timedelta(hours=30)  # Feb 13 06Z

    hours = []
    # GEFS forecast hours: 0-240 every 3h, then 240-384 every 6h
    for fh in list(range(0, 241, 3)) + list(range(246, 385, 6)):
        valid_time = cycle_dt + timedelta(hours=fh)
        if target_start <= valid_time <= target_end_utc:
            hours.append(fh)

    return hours


def _parse_grib2_temperature(data: bytes, expected_lat: float, expected_lon: float) -> float | None:
    """
    Lightweight GRIB2 temperature extraction.
    Since we request a tiny subregion from NOMADS, the data contains
    very few grid points. We extract the 2m temperature value.

    For production, use cfgrib/xarray. This is a fallback that works
    without heavy dependencies.
    """
    # Look for the temperature value pattern in the GRIB2 data
    # GRIB2 section 7 contains the data values
    # For a subregion request with ~1-4 grid points, NOMADS returns simple packing

    if len(data) < 100:
        return None

    try:
        # Try to find "GRIB" magic number
        grib_start = data.find(b"GRIB")
        if grib_start == -1:
            return None

        # For the simple case of NOMADS subregion filter,
        # we'll use a heuristic: scan for IEEE 754 floats in plausible Kelvin range
        # (220K to 330K for surface temp, i.e., -53°C to 57°C)
        candidates = []
        for i in range(grib_start, len(data) - 4):
            try:
                val = struct.unpack(">f", data[i : i + 4])[0]
                if 220.0 <= val <= 330.0:
                    candidates.append(val)
            except struct.error:
                continue

        if candidates:
            # Take the most common value (likely the actual data point)
            from collections import Counter
            most_common = Counter(candidates).most_common(1)[0][0]
            return most_common

    except Exception as e:
        logger.debug(f"GRIB2 parse heuristic failed: {e}")

    return None


async def fetch_gefs_ensemble(
    client: httpx.AsyncClient | None = None,
    pin_cycle: tuple[str, str] | None = None,
) -> dict:
    """
    Fetch all 21 GEFS ensemble member forecasts for the target date/station.

    Returns:
        {
            "source": "GEFS",
            "cycle": "20260211/12",
            "forecast_temps_f": [list of max temp forecasts in °F, one per member],
            "member_details": {member_id: {fhour: temp_f, ...}, ...},
            "fetch_time": ISO timestamp,
        }
    """
    result = {
        "source": "GEFS",
        "cycle": None,
        "forecast_temps_f": [],
        "member_details": {},
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    # Use pinned cycle (for backtest) or auto-discover
    if pin_cycle is not None:
        cycle_info = pin_cycle
        logger.info(f"Using pinned GEFS cycle: {pin_cycle[0]}/{pin_cycle[1]}Z")
    else:
        with httpx.Client() as sync_client:
            cycle_info = _find_latest_available_cycle(sync_client)

    if not cycle_info:
        logger.error("No available GEFS cycle found in last 24 hours")
        return result

    date_str, cycle = cycle_info
    result["cycle"] = f"{date_str}/{cycle}Z"

    forecast_hours = _compute_forecast_hours(date_str, cycle)
    if not forecast_hours:
        logger.warning(f"No forecast hours cover target date from cycle {date_str}/{cycle}Z")
        # Use closest available hours
        forecast_hours = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]

    logger.info(
        f"Fetching GEFS cycle {date_str}/{cycle}Z, "
        f"forecast hours {forecast_hours}, "
        f"21 members × {len(forecast_hours)} hours = {21 * len(forecast_hours)} requests"
    )

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)

    try:
        for member in GEFS_ALL_MEMBERS:
            member_temps = {}

            for fh in forecast_hours:
                url = (
                    f"{GEFS_BASE_URL}?"
                    f"dir=%2Fgefs.{date_str}%2F{cycle}%2Fatmos%2Fpgrb2sp25"
                    f"&file={member}.t{cycle}z.pgrb2s.0p25.f{fh:03d}"
                    f"&var_{GEFS_VARIABLE}=on"
                    f"&lev_{GEFS_LEVEL}=on"
                    f"&subregion="
                    f"&toplat={MARKET.station.lat + 0.25}"
                    f"&leftlon={360 + MARKET.station.lon}"
                    f"&rightlon={360 + MARKET.station.lon + 0.25}"
                    f"&bottomlat={MARKET.station.lat - 0.25}"
                )

                for attempt in range(MAX_RETRIES):
                    try:
                        resp = await client.get(url)
                        if resp.status_code == 200 and len(resp.content) > 50:
                            temp_k = _parse_grib2_temperature(
                                resp.content,
                                MARKET.station.lat,
                                MARKET.station.lon,
                            )
                            if temp_k:
                                temp_f = kelvin_to_fahrenheit(temp_k)
                                member_temps[fh] = round(temp_f, 1)
                                logger.debug(
                                    f"  {member} f{fh:03d}: {temp_f:.1f}°F"
                                )
                            break
                        elif resp.status_code == 404:
                            logger.debug(f"  {member} f{fh:03d}: not available (404)")
                            break
                    except httpx.HTTPError as e:
                        if attempt < MAX_RETRIES - 1:
                            import asyncio
                            await asyncio.sleep(RETRY_DELAY_SECONDS)
                        else:
                            logger.warning(f"  {member} f{fh:03d}: failed after {MAX_RETRIES} retries: {e}")

            if member_temps:
                max_temp = max(member_temps.values())
                result["member_details"][member] = member_temps
                result["forecast_temps_f"].append(max_temp)
                logger.info(f"  {member}: max temp = {max_temp:.1f}°F ({len(member_temps)} hours)")
            else:
                logger.warning(f"  {member}: no data retrieved")

    finally:
        if own_client:
            await client.aclose()

    logger.info(
        f"GEFS fetch complete: {len(result['forecast_temps_f'])}/{len(GEFS_ALL_MEMBERS)} members retrieved"
    )
    return result


def fetch_gefs_ensemble_sync() -> dict:
    """Synchronous wrapper for the GEFS fetcher."""
    import asyncio
    return asyncio.run(fetch_gefs_ensemble())


# ── Alternative: Direct GRIB2 download with cfgrib (if installed) ──────────

def fetch_gefs_with_cfgrib() -> dict:
    """
    Higher-fidelity approach using cfgrib + xarray to properly decode GRIB2.
    Requires: pip install cfgrib eccodes xarray

    This downloads full GRIB2 files and extracts the nearest grid point
    to Dallas Love Field.
    """
    try:
        import xarray as xr
    except ImportError:
        logger.error("cfgrib/xarray not installed. Use: pip install cfgrib eccodes xarray")
        return fetch_gefs_ensemble_sync()

    result = {
        "source": "GEFS-cfgrib",
        "cycle": None,
        "forecast_temps_f": [],
        "member_details": {},
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    with httpx.Client() as sync_client:
        cycle_info = _find_latest_available_cycle(sync_client)

    if not cycle_info:
        return result

    date_str, cycle = cycle_info
    result["cycle"] = f"{date_str}/{cycle}Z"
    forecast_hours = _compute_forecast_hours(date_str, cycle) or [24, 48, 72]

    data_dir = Path(DATA_DIR) / "grib"
    data_dir.mkdir(parents=True, exist_ok=True)

    for member in GEFS_ALL_MEMBERS:
        member_temps = {}
        for fh in forecast_hours:
            url = (
                f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/"
                f"gefs.{date_str}/{cycle}/atmos/pgrb2sp25/"
                f"{member}.t{cycle}z.pgrb2s.0p25.f{fh:03d}"
            )
            local_path = data_dir / f"{member}_{date_str}_{cycle}z_f{fh:03d}.grib2"

            try:
                if not local_path.exists():
                    resp = httpx.get(url, timeout=120)
                    if resp.status_code == 200:
                        local_path.write_bytes(resp.content)

                if local_path.exists():
                    ds = xr.open_dataset(
                        local_path,
                        engine="cfgrib",
                        filter_by_keys={"shortName": "2t", "typeOfLevel": "heightAboveGround"},
                    )
                    temp_k = float(
                        ds["t2m"].sel(
                            latitude=MARKET.station.lat,
                            longitude=360 + MARKET.station.lon,
                            method="nearest",
                        ).values
                    )
                    temp_f = kelvin_to_fahrenheit(temp_k)
                    member_temps[fh] = round(temp_f, 1)
            except Exception as e:
                logger.debug(f"cfgrib failed for {member} f{fh:03d}: {e}")

        if member_temps:
            max_temp = max(member_temps.values())
            result["member_details"][member] = member_temps
            result["forecast_temps_f"].append(max_temp)

    return result
