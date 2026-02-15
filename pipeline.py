#!/usr/bin/env python3
"""
Weather Prediction Data Ingestion Pipeline
==========================================

Pulls ensemble forecasts from multiple NWP models, converts them into
probability distributions across temperature buckets matching Polymarket
weather markets, and identifies arbitrage opportunities.

Supports multiple markets (Dallas, Seoul, etc.) via --market flag.

Usage:
    python pipeline.py                          # Dallas (default)
    python pipeline.py --market seoul           # Seoul
    python pipeline.py --continuous             # Run every hour
    python pipeline.py --polymarket-slug highest-temperature-in-dallas-on-february-12-2026
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from config import DATA_DIR, FETCH_INTERVAL_SECONDS, LOG_DIR, MARKETS

# ── Logging Setup ──────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


logger = logging.getLogger("pipeline")


# ── Pipeline Core ──────────────────────────────────────────────────────────

async def run_pipeline(
    market_prices: dict | None = None,
    save_output: bool = True,
) -> dict:
    """Execute one full pipeline run for the active market."""
    from gfs_fetcher import fetch_gefs_ensemble
    from ecmwf_fetcher import fetch_ecmwf_ensemble
    from hrrr_fetcher import fetch_hrrr_forecast
    from nws_forecast_fetcher import fetch_nws_forecast
    from openmeteo_fetcher import fetch_openmeteo_forecast
    from polymarket_scraper import fetch_polymarket_prices
    from probability_engine import (
        apply_reality_check,
        compute_distribution,
        find_arbitrage_opportunities,
        format_distribution,
    )
    from current_conditions import fetch_current_conditions, fetch_weather_conditions

    market = config.MARKET
    u = market.unit_symbol
    sources = market.sources

    run_start = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    logger.info("=" * 70)
    logger.info(f"Pipeline run {run_id}")
    logger.info(f"Target: {market.station.name} — {market.target_date.strftime('%B %d, %Y')} ({u})")
    logger.info(f"Sources: {', '.join(sources)}")
    logger.info("=" * 70)

    # Count total steps based on available sources
    step_count = len(sources) + 2  # +1 for observations, +1 for arbitrage
    step = 0

    # ── Fetch GEFS ──────────────────────────────────────────────────
    gefs_data = {"forecast_temps_f": [], "member_details": {}}
    if "gefs" in sources:
        step += 1
        logger.info(f"\n[{step}/{step_count}] Fetching NOAA GFS Ensemble (21 members)...")
        t0 = time.time()
        try:
            gefs_data = await fetch_gefs_ensemble()
            logger.info(
                f"  GEFS: {len(gefs_data['forecast_temps_f'])} members retrieved "
                f"in {time.time() - t0:.1f}s"
            )
            if gefs_data["forecast_temps_f"]:
                temps = gefs_data["forecast_temps_f"]
                logger.info(
                    f"  GEFS temps: min={min(temps):.1f}{u}, "
                    f"max={max(temps):.1f}{u}, "
                    f"mean={sum(temps)/len(temps):.1f}{u}"
                )
        except Exception as e:
            logger.error(f"  GEFS fetch failed: {e}")

    # ── Fetch HRRR ──────────────────────────────────────────────────
    hrrr_data = {"forecast_temps_f": [], "member_details": {}}
    if "hrrr" in sources:
        step += 1
        logger.info(f"\n[{step}/{step_count}] Fetching NOAA HRRR (last 4 cycles)...")
        t0 = time.time()
        try:
            hrrr_data = await fetch_hrrr_forecast()
            logger.info(
                f"  HRRR: {len(hrrr_data['forecast_temps_f'])} cycles retrieved "
                f"in {time.time() - t0:.1f}s"
            )
            if hrrr_data["forecast_temps_f"]:
                temps = hrrr_data["forecast_temps_f"]
                logger.info(
                    f"  HRRR temps: min={min(temps):.1f}{u}, "
                    f"max={max(temps):.1f}{u}, "
                    f"mean={sum(temps)/len(temps):.1f}{u}"
                )
        except Exception as e:
            logger.error(f"  HRRR fetch failed: {e}")

    # ── Fetch ECMWF ──────────────────────────────────────────────────
    ecmwf_data = {"forecast_temps_f": [], "member_details": {}}
    if "ecmwf" in sources:
        step += 1
        logger.info(f"\n[{step}/{step_count}] Fetching ECMWF Ensemble (51 members)...")
        t0 = time.time()
        try:
            ecmwf_data = await fetch_ecmwf_ensemble()
            logger.info(
                f"  ECMWF: {len(ecmwf_data['forecast_temps_f'])} members retrieved "
                f"in {time.time() - t0:.1f}s"
            )
            if ecmwf_data["forecast_temps_f"]:
                temps = ecmwf_data["forecast_temps_f"]
                logger.info(
                    f"  ECMWF temps: min={min(temps):.1f}{u}, "
                    f"max={max(temps):.1f}{u}"
                )
            if ecmwf_data.get("hres_forecast_f"):
                logger.info(f"  ECMWF HRES (deterministic): {ecmwf_data['hres_forecast_f']}{u}")
        except Exception as e:
            logger.error(f"  ECMWF fetch failed: {e}")

    # ── Fetch NWS Point Forecast ────────────────────────────────────
    nws_data = {"forecast_temps_f": [], "member_details": {}}
    if "nws" in sources:
        step += 1
        logger.info(f"\n[{step}/{step_count}] Fetching NWS Point Forecast...")
        t0 = time.time()
        try:
            nws_data = await fetch_nws_forecast()
            logger.info(
                f"  NWS: {len(nws_data['forecast_temps_f'])} forecast(s) retrieved "
                f"in {time.time() - t0:.1f}s"
            )
            if nws_data["forecast_temps_f"]:
                logger.info(f"  NWS high: {nws_data['forecast_temps_f'][0]:.1f}{u}")
        except Exception as e:
            logger.error(f"  NWS fetch failed: {e}")

    # ── Fetch Open-Meteo Multi-Model ────────────────────────────────
    openmeteo_data = {"forecast_temps_f": [], "member_details": {}}
    if "openmeteo" in sources:
        step += 1
        logger.info(f"\n[{step}/{step_count}] Fetching Open-Meteo multi-model forecasts...")
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
                    f"  Open-Meteo temps: min={min(temps):.1f}{u}, "
                    f"max={max(temps):.1f}{u}, "
                    f"mean={sum(temps)/len(temps):.1f}{u}"
                )
        except Exception as e:
            logger.error(f"  Open-Meteo fetch failed: {e}")

    # ── Fetch observations & compute probability distribution ──────
    step += 1
    logger.info(f"\n[{step}/{step_count}] Fetching observations & computing probabilities...")
    t0 = time.time()

    obs = await fetch_current_conditions(market.station.icao)

    # Fetch weather conditions (precipitation, thunder, storm)
    weather_conditions = await fetch_weather_conditions(
        market.station.icao,
        market.target_date,
    )

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
            weather_conditions=weather_conditions,
        )
    else:
        logger.warning("  No current observations — using pure forecast (no bias correction)")
        distribution = compute_distribution(
            gefs_data, ecmwf_data, hrrr_data=hrrr_data,
            nws_data=nws_data, openmeteo_data=openmeteo_data,
        )

    output = format_distribution(distribution)
    logger.info(f"\n{output}")
    logger.info(f"  Computed in {time.time() - t0:.3f}s")

    # ── Arbitrage scan ──────────────────────────────────────────────
    step += 1
    logger.info(f"\n[{step}/{step_count}] Scanning for arbitrage opportunities...")
    opportunities = find_arbitrage_opportunities(distribution, market_prices)

    if not opportunities:
        logger.info("  No arbitrage opportunities found (or no market prices provided)")
    else:
        logger.info(f"  Found {len(opportunities)} opportunities!")
        for opp in opportunities:
            logger.info(
                f"  → {opp['action']} \"{opp['bucket']}\" | "
                f"Market: {opp.get('market_yes', 0):.1%} | "
                f"Model: {opp['model_prob']:.1%} | "
                f"Edge: {opp['edge']:.1%} | "
                f"EV: ${opp['expected_value']:.3f} [{opp['confidence']}]"
            )

    # ── Save results ────────────────────────────────────────────────
    if save_output:
        output_dir = Path(DATA_DIR) / "runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"run_{run_id}.json"

        results = {
            "run_id": run_id,
            "target": {
                "station": market.station.name,
                "icao": market.station.icao,
                "date": market.target_date.isoformat(),
                "unit": market.unit,
                "lat": market.station.lat,
                "lon": market.station.lon,
            },
            "gefs": {
                "cycle": gefs_data.get("cycle"),
                "member_count": len(gefs_data["forecast_temps_f"]),
                "temps": gefs_data["forecast_temps_f"],
            },
            "ecmwf": {
                "cycle": ecmwf_data.get("cycle"),
                "member_count": len(ecmwf_data["forecast_temps_f"]),
                "temps": ecmwf_data["forecast_temps_f"],
                "hres": ecmwf_data.get("hres_forecast_f"),
            },
            "openmeteo": {
                "model_count": len(openmeteo_data["forecast_temps_f"]),
                "temps": openmeteo_data["forecast_temps_f"],
                "models": openmeteo_data.get("member_details", {}),
            },
            "distribution": {
                "method": distribution.method,
                "ensemble_mean": distribution.ensemble_mean_f,
                "ensemble_std": distribution.ensemble_std_f,
                "total_members": distribution.total_members,
                "most_likely_bucket": distribution.most_likely_bucket,
                "most_likely_prob": distribution.most_likely_prob,
                "buckets": [
                    {
                        "label": b.label,
                        "lower": b.lower_f,
                        "upper": b.upper_f,
                        "probability": round(b.probability, 6),
                        "gefs_prob": round(b.gefs_prob, 6),
                        "ecmwf_prob": round(b.ecmwf_prob, 6),
                        "member_count": b.member_count,
                    }
                    for b in distribution.buckets
                    if b.probability >= 0.001
                ],
            },
            "arbitrage": opportunities,
            "elapsed_seconds": round(time.time() - run_start, 1),
        }

        # Include optional sources in output
        if "hrrr" in sources:
            results["hrrr"] = {
                "cycle": hrrr_data.get("cycle"),
                "cycle_count": len(hrrr_data["forecast_temps_f"]),
                "temps": hrrr_data["forecast_temps_f"],
            }
        if "nws" in sources:
            results["nws"] = {
                "forecast_count": len(nws_data["forecast_temps_f"]),
                "temps": nws_data["forecast_temps_f"],
                "details": nws_data.get("member_details", {}),
            }

        output_file.write_text(json.dumps(results, indent=2))
        logger.info(f"\nResults saved to {output_file}")

    elapsed = time.time() - run_start
    logger.info(f"\nPipeline run complete in {elapsed:.1f}s")

    return results


async def run_continuous(
    interval: int = FETCH_INTERVAL_SECONDS,
    market_prices: dict | None = None,
    polymarket_slug: str | None = None,
) -> None:
    """Run the pipeline continuously at the specified interval."""
    logger.info(f"Starting continuous mode — running every {interval}s")
    logger.info("Press Ctrl+C to stop\n")

    run_count = 0
    while True:
        run_count += 1
        logger.info(f"\n{'━' * 70}")
        logger.info(f"Continuous run #{run_count}")
        logger.info(f"{'━' * 70}")

        try:
            prices = market_prices
            if polymarket_slug:
                from polymarket_scraper import fetch_polymarket_prices
                prices = await fetch_polymarket_prices(polymarket_slug)
            await run_pipeline(market_prices=prices)
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}", exc_info=True)

        logger.info(f"\nSleeping {interval}s until next run...")
        await asyncio.sleep(interval)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weather Prediction Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Markets: {', '.join(MARKETS.keys())}

Examples:
  python pipeline.py                          # Dallas (default)
  python pipeline.py --market seoul           # Seoul
  python pipeline.py --continuous             # Run every hour
  python pipeline.py --prices prices.json     # Compare against Polymarket
  python pipeline.py -v                       # Verbose logging
        """,
    )

    parser.add_argument(
        "--market", "-m",
        choices=list(MARKETS.keys()),
        default="dallas",
        help="Target market (default: dallas)",
    )
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuously at specified interval",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=FETCH_INTERVAL_SECONDS,
        help=f"Seconds between runs in continuous mode (default: {FETCH_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--polymarket-slug", "-pm",
        type=str,
        help="Event slug to fetch live Polymarket prices",
    )
    parser.add_argument(
        "--prices", "-p",
        type=str,
        help="Path to JSON file with Polymarket prices {bucket_label: yes_price}",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Set active market BEFORE importing fetchers (they bind config.MARKET at import)
    config.set_market(args.market)

    setup_logging(verbose=args.verbose)

    # Load market prices
    market_prices = None
    if args.polymarket_slug:
        from polymarket_scraper import fetch_polymarket_prices
        market_prices = asyncio.run(fetch_polymarket_prices(args.polymarket_slug))
        logger.info(f"Fetched {len(market_prices)} live Polymarket prices")
    elif args.prices:
        prices_path = Path(args.prices)
        if prices_path.exists():
            market_prices = json.loads(prices_path.read_text())
            logger.info(f"Loaded {len(market_prices)} Polymarket prices from {args.prices}")
        else:
            logger.warning(f"Prices file not found: {args.prices}")

    if args.continuous:
        asyncio.run(run_continuous(
            interval=args.interval,
            market_prices=market_prices,
            polymarket_slug=args.polymarket_slug,
        ))
    else:
        asyncio.run(run_pipeline(market_prices=market_prices))


if __name__ == "__main__":
    main()
