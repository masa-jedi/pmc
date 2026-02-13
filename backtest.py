#!/usr/bin/env python3
"""
Backtest: replay the pipeline at a historical time.

Fetches model data from the cycles that would have been available at the
specified time, filters observations to only what was known then, and
runs the probability engine — as if you were making a prediction at that moment.

Usage:
    python backtest.py --date 2026-02-10 --as-of-local 12:00
    python backtest.py --date 2026-02-10 --as-of-local 12:00 --actual-high 73.9
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from config import MARKET

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest")

STATION_UTC_OFFSET = timedelta(hours=-6)  # Dallas CST


def _compute_available_cycles(as_of_utc: datetime) -> dict:
    """
    Determine which model cycles would have been available at as_of_utc,
    accounting for publication delays.

    Returns dict with pin_cycle info for each fetcher.
    """
    # HRRR: publishes ~30-45 min after cycle. At noon CST (18Z), the 17Z
    # cycle is likely available. Grab last 4.
    hrrr_cycles = []
    for hours_back in range(1, 13):
        check_time = as_of_utc - timedelta(hours=hours_back)
        hrrr_cycles.append((check_time.strftime("%Y%m%d"), f"{check_time.hour:02d}"))
        if len(hrrr_cycles) >= 4:
            break

    # GEFS: publishes 4-6h after cycle. Runs at 00/06/12/18Z.
    gefs_cycle = None
    for hours_back in range(6, 30, 6):
        check_time = as_of_utc - timedelta(hours=hours_back)
        cycle_hour = (check_time.hour // 6) * 6
        candidate = check_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        if candidate + timedelta(hours=6) <= as_of_utc:  # allow 6h for publication
            gefs_cycle = (candidate.strftime("%Y%m%d"), f"{candidate.hour:02d}")
            break

    # ECMWF: publishes 7-8h after cycle. Runs at 00/12Z.
    ecmwf_cycle = None
    for hours_back in range(8, 48, 12):
        check_time = as_of_utc - timedelta(hours=hours_back)
        cycle_hour = 12 if check_time.hour >= 12 else 0
        candidate = check_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        if candidate + timedelta(hours=8) <= as_of_utc:
            ecmwf_cycle = (candidate.strftime("%Y%m%d"), f"{candidate.hour:02d}")
            break

    return {
        "hrrr_cycles": hrrr_cycles,
        "gefs_cycle": gefs_cycle,
        "ecmwf_cycle": ecmwf_cycle,
    }


async def run_backtest(
    target_date: str,
    as_of_local_hour: int,
    actual_high: float | None = None,
) -> None:
    """Run the pipeline as if it were a specific local time on the target date."""
    from gfs_fetcher import fetch_gefs_ensemble
    from hrrr_fetcher import fetch_hrrr_forecast
    from ecmwf_fetcher import fetch_ecmwf_ensemble
    from nws_forecast_fetcher import fetch_nws_forecast
    from openmeteo_fetcher import fetch_openmeteo_forecast
    from current_conditions import fetch_current_conditions
    from probability_engine import (
        apply_reality_check,
        compute_distribution,
        format_distribution,
    )

    # Parse target date
    target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Compute as-of time in UTC
    as_of_utc = target_dt.replace(hour=as_of_local_hour) - STATION_UTC_OFFSET
    logger.info("=" * 70)
    logger.info(f"BACKTEST: {MARKET.station.name} — {target_date}")
    logger.info(f"  Simulating pipeline at {as_of_local_hour:02d}:00 local "
                f"({as_of_utc.strftime('%H:%M')} UTC)")
    if actual_high is not None:
        logger.info(f"  Actual high (ground truth): {actual_high}°F")
    logger.info("=" * 70)

    # Determine which cycles would have been available
    cycles = _compute_available_cycles(as_of_utc)
    logger.info(f"\nAvailable cycles at {as_of_local_hour:02d}:00 local:")
    logger.info(f"  HRRR:  {[f'{d}/{h}Z' for d, h in cycles['hrrr_cycles']]}")
    logger.info(f"  GEFS:  {cycles['gefs_cycle'][0]}/{cycles['gefs_cycle'][1]}Z" if cycles['gefs_cycle'] else "  GEFS:  None")
    logger.info(f"  ECMWF: {cycles['ecmwf_cycle'][0]}/{cycles['ecmwf_cycle'][1]}Z" if cycles['ecmwf_cycle'] else "  ECMWF: None")

    run_start = time.time()

    # ── Fetch HRRR with pinned cycles ──────────────────────────────
    logger.info("\n[1/6] Fetching HRRR...")
    t0 = time.time()
    try:
        hrrr_data = await fetch_hrrr_forecast(pin_cycles=cycles["hrrr_cycles"])
        logger.info(
            f"  HRRR: {len(hrrr_data['forecast_temps_f'])} cycles retrieved "
            f"in {time.time() - t0:.1f}s"
        )
        if hrrr_data["forecast_temps_f"]:
            logger.info(
                f"  HRRR temps: min={min(hrrr_data['forecast_temps_f']):.1f}°F, "
                f"max={max(hrrr_data['forecast_temps_f']):.1f}°F, "
                f"mean={sum(hrrr_data['forecast_temps_f'])/len(hrrr_data['forecast_temps_f']):.1f}°F"
            )
    except Exception as e:
        logger.error(f"  HRRR fetch failed: {e}")
        hrrr_data = {"forecast_temps_f": [], "member_details": {}}

    # ── Fetch GEFS with pinned cycle ───────────────────────────────
    logger.info("\n[2/6] Fetching GEFS...")
    t0 = time.time()
    try:
        gefs_data = await fetch_gefs_ensemble(pin_cycle=cycles["gefs_cycle"])
        logger.info(
            f"  GEFS: {len(gefs_data['forecast_temps_f'])} members retrieved "
            f"in {time.time() - t0:.1f}s"
        )
        if gefs_data["forecast_temps_f"]:
            logger.info(
                f"  GEFS temps: min={min(gefs_data['forecast_temps_f']):.1f}°F, "
                f"max={max(gefs_data['forecast_temps_f']):.1f}°F"
            )
    except Exception as e:
        logger.error(f"  GEFS fetch failed: {e}")
        gefs_data = {"forecast_temps_f": [], "member_details": {}}

    # ── Fetch ECMWF ────────────────────────────────────────────────
    logger.info("\n[3/6] Fetching ECMWF...")
    t0 = time.time()
    try:
        ecmwf_data = await fetch_ecmwf_ensemble(pin_cycle=cycles["ecmwf_cycle"])
        logger.info(
            f"  ECMWF: {len(ecmwf_data['forecast_temps_f'])} members retrieved "
            f"in {time.time() - t0:.1f}s"
        )
        if ecmwf_data["forecast_temps_f"]:
            logger.info(
                f"  ECMWF temps: min={min(ecmwf_data['forecast_temps_f']):.1f}°F, "
                f"max={max(ecmwf_data['forecast_temps_f']):.1f}°F"
            )
    except Exception as e:
        logger.error(f"  ECMWF fetch failed: {e}")
        ecmwf_data = {"forecast_temps_f": [], "member_details": {}}

    # ── Fetch NWS Point Forecast ───────────────────────────────────
    logger.info("\n[4/6] Fetching NWS Point Forecast...")
    t0 = time.time()
    try:
        nws_data = await fetch_nws_forecast()
        logger.info(
            f"  NWS: {len(nws_data['forecast_temps_f'])} forecast(s) retrieved "
            f"in {time.time() - t0:.1f}s"
        )
        if nws_data["forecast_temps_f"]:
            logger.info(f"  NWS high: {nws_data['forecast_temps_f'][0]:.1f}°F")
    except Exception as e:
        logger.error(f"  NWS fetch failed: {e}")
        nws_data = {"forecast_temps_f": [], "member_details": {}}

    # ── Fetch Open-Meteo Multi-Model ───────────────────────────────
    logger.info("\n[5/6] Fetching Open-Meteo multi-model forecasts...")
    t0 = time.time()
    try:
        openmeteo_data = await fetch_openmeteo_forecast()
        logger.info(
            f"  Open-Meteo: {len(openmeteo_data['forecast_temps_f'])} models retrieved "
            f"in {time.time() - t0:.1f}s"
        )
        if openmeteo_data["forecast_temps_f"]:
            temps = openmeteo_data["forecast_temps_f"]
            logger.info(
                f"  Open-Meteo temps: min={min(temps):.1f}°F, "
                f"max={max(temps):.1f}°F, "
                f"mean={sum(temps)/len(temps):.1f}°F"
            )
    except Exception as e:
        logger.error(f"  Open-Meteo fetch failed: {e}")
        openmeteo_data = {"forecast_temps_f": [], "member_details": {}}

    # ── Fetch observations filtered to as-of time ──────────────────
    logger.info("\n[6/6] Fetching observations (filtered to as-of time)...")
    obs = await fetch_current_conditions(MARKET.station.icao, as_of_utc=as_of_utc)

    if obs:
        logger.info(
            f"  Obs as of {as_of_utc.strftime('%H:%M')} UTC: "
            f"current={obs['current_temp_f']:.1f}°F, "
            f"high so far={obs['observed_high_f']:.1f}°F"
        )

    # ── Compute distribution ───────────────────────────────────────
    if obs:
        distribution = apply_reality_check(
            gefs_data,
            ecmwf_data,
            current_temp_f=obs["current_temp_f"],
            observed_high_f=obs["observed_high_f"],
            observed_high_time=obs.get("observed_high_time"),
            observed_at=obs.get("observed_at"),
            hrrr_data=hrrr_data,
            nws_data=nws_data,
            openmeteo_data=openmeteo_data,
        )
    else:
        logger.warning("  No observations available")
        distribution = compute_distribution(
            gefs_data, ecmwf_data, hrrr_data=hrrr_data,
            nws_data=nws_data, openmeteo_data=openmeteo_data,
        )

    output = format_distribution(distribution)
    logger.info(f"\n{output}")

    # ── Compare against actual high ────────────────────────────────
    if actual_high is not None:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  GROUND TRUTH: {actual_high}°F")

        # Find which bucket the actual high falls into
        for bucket in distribution.buckets:
            if bucket.lower_f <= actual_high < bucket.upper_f:
                logger.info(
                    f"  Correct bucket: {bucket.label} "
                    f"(model gave {bucket.probability:.1%})"
                )
                break
            elif bucket.upper_f >= MARKET.bucket_max_f and actual_high >= bucket.lower_f:
                logger.info(
                    f"  Correct bucket: {bucket.label} "
                    f"(model gave {bucket.probability:.1%})"
                )
                break

        logger.info("=" * 70)

    elapsed = time.time() - run_start
    logger.info(f"\nBacktest complete in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest: replay pipeline at a historical time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --date 2026-02-10 --as-of-local 12:00
  python backtest.py --date 2026-02-10 --as-of-local 12:00 --actual-high 73.9
        """,
    )
    parser.add_argument(
        "--date", "-d",
        required=True,
        help="Target date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--as-of-local", "-t",
        required=True,
        help="Local time to simulate (HH:MM, CST)",
    )
    parser.add_argument(
        "--actual-high", "-a",
        type=float,
        help="Actual daily high for comparison (°F)",
    )

    args = parser.parse_args()

    # Parse local time
    as_of_local_hour = int(args.as_of_local.split(":")[0])

    asyncio.run(run_backtest(
        target_date=args.date,
        as_of_local_hour=as_of_local_hour,
        actual_high=args.actual_high,
    ))


if __name__ == "__main__":
    main()
