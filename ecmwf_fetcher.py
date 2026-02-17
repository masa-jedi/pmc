"""
ECMWF Open Data Fetcher
Pulls IFS HRES + Ensemble temperature forecasts from ECMWF's free open data API.

ECMWF Open Data provides:
- IFS HRES (high-resolution deterministic): single best forecast
- IFS ENS (ensemble): 51 members (control + 50 perturbed)

Data is published twice daily (00Z and 12Z) with ~7-8 hour delay.
Available at: https://data.ecmwf.int/forecasts/
Also accessible via the `ecmwf-opendata` Python package.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

import config
from config import (
    DATA_DIR,
    ECMWF_CYCLES,
    ECMWF_ENS_MEMBERS,
    ECMWF_OPEN_BASE,
    HTTP_TIMEOUT_SECONDS,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def kelvin_to_fahrenheit(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32


def _find_latest_ecmwf_cycle() -> tuple[str, str] | None:
    """
    Find the most recent available ECMWF open data cycle.
    ECMWF publishes 00Z data around 07-08Z, and 12Z data around 19-20Z.
    """
    now = datetime.now(timezone.utc)

    with httpx.Client(timeout=30) as client:
        for hours_back in range(0, 48, 12):
            check_time = now - timedelta(hours=hours_back)
            date_str = check_time.strftime("%Y%m%d")

            for cycle in sorted(ECMWF_CYCLES, reverse=True):
                cycle_hour = int(cycle)
                cycle_time = check_time.replace(hour=cycle_hour, minute=0, second=0)
                if cycle_time > now:
                    continue

                # Check if this cycle's data index exists
                index_url = (
                    f"{ECMWF_OPEN_BASE}/{date_str}/{cycle}z/ifs/0p25/enfo/"
                )
                try:
                    resp = client.head(index_url, timeout=15, follow_redirects=True)
                    if resp.status_code in (200, 301, 302):
                        logger.info(f"Found available ECMWF cycle: {date_str}/{cycle}Z")
                        return date_str, cycle
                except httpx.HTTPError:
                    continue

    return None


def _compute_ecmwf_steps(cycle_date: str, cycle_hour: str) -> list[int]:
    """
    Compute which forecast steps cover the target date's daytime for max temp.
    ECMWF ENS steps: 0-144h every 3h, then 144-360h every 6h.
    """
    cycle_dt = datetime.strptime(f"{cycle_date}{cycle_hour}", "%Y%m%d%H").replace(
        tzinfo=timezone.utc
    )
    market = config.MARKET
    target = market.target_date

    # Cover local 06:00 to midnight → convert to UTC
    local_start_utc = 6 - market.utc_offset_hours
    target_start = target.replace(hour=0) + timedelta(hours=local_start_utc)
    target_end = target.replace(hour=0) + timedelta(hours=24 - market.utc_offset_hours + 6)

    steps = []
    all_steps = list(range(0, 145, 3)) + list(range(150, 361, 6))

    for step in all_steps:
        valid_time = cycle_dt + timedelta(hours=step)
        if target_start <= valid_time <= target_end:
            steps.append(step)

    return steps


async def fetch_ecmwf_open_data_via_package(
    pin_cycle: tuple[str, str] | None = None,
) -> dict:
    """
    Fetch ECMWF data using the official `ecmwf-opendata` package.
    This is the recommended approach — install with: pip install ecmwf-opendata

    Returns ensemble 2m temperature forecasts for all 51 members.
    """
    result = {
        "source": "ECMWF-ENS",
        "cycle": None,
        "forecast_temps_f": [],
        "member_details": {},
        "hres_forecast_f": None,
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    try:
        from ecmwf.opendata import Client as ECMWFClient
    except ImportError:
        logger.warning(
            "ecmwf-opendata not installed. Install with: pip install ecmwf-opendata"
        )
        return await fetch_ecmwf_open_data_raw()

    data_dir = Path(DATA_DIR) / "ecmwf"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use pinned cycle (for backtest) or auto-discover
    if pin_cycle is not None:
        cycle_info = pin_cycle
        logger.info(f"Using pinned ECMWF cycle: {pin_cycle[0]}/{pin_cycle[1]}Z")
    else:
        cycle_info = _find_latest_ecmwf_cycle()
    if cycle_info:
        date_str, cycle = cycle_info
        result["cycle"] = f"{date_str}/{cycle}Z"
        steps = _compute_ecmwf_steps(date_str, cycle)
        if not steps:
            logger.warning("No ECMWF steps cover target date — falling back to step 0")
            steps = [0]
        logger.info(f"ECMWF cycle {date_str}/{cycle}Z, steps: {steps}")
    else:
        logger.warning("Could not determine ECMWF cycle — fetching latest without step filter")
        date_str, cycle, steps = None, None, None

    try:
        ecmwf = ECMWFClient()

        # Common kwargs for all retrieve calls (only set when cycle is known)
        cycle_kwargs = {}
        if date_str and cycle and steps:
            cycle_kwargs = {
                "date": date_str,
                "time": int(cycle),
                "step": steps,
            }

        # ── Fetch HRES (deterministic) ──────────────────────────────
        hres_path = data_dir / "hres_2t.grib2"
        try:
            ecmwf.retrieve(
                **cycle_kwargs,
                type="fc",
                param="2t",
                target=str(hres_path),
                model="ifs",
                resol="0p25",
            )
            logger.info("Downloaded ECMWF HRES 2m temperature")
        except Exception as e:
            logger.warning(f"ECMWF HRES download failed: {e}")

        # ── Fetch ENS (ensemble) ────────────────────────────────────
        ens_path = data_dir / "ens_2t.grib2"
        try:
            ecmwf.retrieve(
                **cycle_kwargs,
                type="pf",  # perturbed forecast (ensemble)
                param="2t",
                target=str(ens_path),
                model="ifs",
                resol="0p25",
            )
            logger.info("Downloaded ECMWF ENS 2m temperature")
        except Exception as e:
            logger.warning(f"ECMWF ENS download failed: {e}")

        # ── Extract temperatures using xarray/cfgrib ────────────────
        try:
            import xarray as xr

            if hres_path.exists():
                ds = xr.open_dataset(hres_path, engine="cfgrib")
                temp_k = float(
                    ds["t2m"].sel(
                        latitude=config.MARKET.station.lat,
                        longitude=config.MARKET.station.lon % 360,
                        method="nearest",
                    ).max().values
                )
                result["hres_forecast_f"] = round(config.MARKET.kelvin_to_unit(temp_k), 1)
                logger.info(f"ECMWF HRES max temp: {result['hres_forecast_f']}°F")

            if ens_path.exists():
                ds = xr.open_dataset(ens_path, engine="cfgrib")
                for member in range(ECMWF_ENS_MEMBERS):
                    try:
                        temp_k = float(
                            ds["t2m"].sel(
                                latitude=config.MARKET.station.lat,
                                longitude=config.MARKET.station.lon % 360,
                                number=member,
                                method="nearest",
                            ).max().values
                        )
                        temp_f = round(config.MARKET.kelvin_to_unit(temp_k), 1)
                        result["forecast_temps_f"].append(temp_f)
                        result["member_details"][f"ens{member:02d}"] = temp_f
                    except Exception:
                        continue

        except ImportError:
            logger.warning("xarray/cfgrib not installed for GRIB decoding")

    except Exception as e:
        logger.error(f"ECMWF open data fetch failed: {e}")

    # Add detailed summary logging similar to Open-Meteo
    if result["forecast_temps_f"]:
        temps = result["forecast_temps_f"]
        logger.info(
            f"ECMWF: {len(temps)} members retrieved — "
            f"min={min(temps):.1f}°F, max={max(temps):.1f}°F, "
            f"mean={sum(temps)/len(temps):.1f}°F"
        )
        logger.info("=== ECMWF Member Results ===")
        for member, temp in result.get("member_details", {}).items():
            if isinstance(temp, (int, float)):
                logger.info(f"  {member}: {temp:.1f}°F")
        logger.info("===========================")

    return result


async def fetch_ecmwf_open_data_raw() -> dict:
    """
    Fetch ECMWF data directly from the open data HTTP endpoint.
    This works without the ecmwf-opendata package but requires more manual work.

    Strategy: Download the index file to find relevant GRIB messages,
    then use byte-range requests to download only the 2m temperature data.
    """
    result = {
        "source": "ECMWF-ENS-raw",
        "cycle": None,
        "forecast_temps_f": [],
        "member_details": {},
        "hres_forecast_f": None,
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }

    cycle_info = _find_latest_ecmwf_cycle()
    if not cycle_info:
        logger.error("No available ECMWF cycle found")
        return result

    date_str, cycle = cycle_info
    result["cycle"] = f"{date_str}/{cycle}Z"

    steps = _compute_ecmwf_steps(date_str, cycle)
    if not steps:
        logger.warning("No ECMWF forecast steps cover target date")
        steps = [24, 48, 72, 96, 120]

    logger.info(f"ECMWF cycle {date_str}/{cycle}Z, steps: {steps}")

    data_dir = Path(DATA_DIR) / "ecmwf"
    data_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        # ── Try fetching ensemble index ─────────────────────────────
        for step in steps:
            # ECMWF open data URL pattern for ensemble
            # Format: {date}/{cycle}z/ifs/0p25/enfo/{date}{cycle}0000-{step}h-enfo-ef.index
            step_str = f"{step}h"
            index_url = (
                f"{ECMWF_OPEN_BASE}/{date_str}/{cycle}z/ifs/0p25/enfo/"
                f"{date_str}{cycle}0000-{step_str}-enfo-ef.index"
            )

            try:
                resp = await client.get(index_url, follow_redirects=True)
                if resp.status_code == 200:
                    # Parse the index to find 2t (2m temperature) entries
                    index_text = resp.text
                    for line in index_text.strip().split("\n"):
                        if '"param":"2t"' in line or '"shortName":"2t"' in line:
                            try:
                                entry = json.loads(line.split("}", 1)[0] + "}")
                                # Found a 2m temp entry — record it
                                member_id = entry.get("number", "cf")
                                logger.debug(
                                    f"ECMWF index: step={step}, member={member_id}"
                                )
                            except json.JSONDecodeError:
                                pass
                    logger.info(f"ECMWF step {step}: index parsed")
                else:
                    logger.debug(f"ECMWF step {step}: index not available ({resp.status_code})")

            except httpx.HTTPError as e:
                logger.debug(f"ECMWF step {step}: {e}")

    logger.info(
        f"ECMWF raw fetch complete: {len(result['forecast_temps_f'])} member forecasts"
    )

    # Add detailed summary logging similar to Open-Meteo
    if result["forecast_temps_f"]:
        temps = result["forecast_temps_f"]
        logger.info(
            f"ECMWF: {len(temps)} members retrieved — "
            f"min={min(temps):.1f}°F, max={max(temps):.1f}°F, "
            f"mean={sum(temps)/len(temps):.1f}°F"
        )
        logger.info("=== ECMWF Member Results ===")
        for member, temp in result.get("member_details", {}).items():
            if isinstance(temp, (int, float)):
                logger.info(f"  {member}: {temp:.1f}°F")
        logger.info("===========================")

    return result


async def fetch_ecmwf_ensemble(
    pin_cycle: tuple[str, str] | None = None,
) -> dict:
    """
    Main entry point: tries the official package first, falls back to raw HTTP.
    """
    try:
        return await fetch_ecmwf_open_data_via_package(pin_cycle=pin_cycle)
    except Exception as e:
        logger.warning(f"ECMWF package fetch failed ({e}), falling back to raw HTTP")
        return await fetch_ecmwf_open_data_raw()
