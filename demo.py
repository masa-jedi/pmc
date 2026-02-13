#!/usr/bin/env python3
"""
Demo: Run the probability engine with simulated ensemble data
to verify the pipeline logic without network access.

Usage:
    python demo.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("demo")

from config import MARKET
from probability_engine import (
    compute_distribution,
    find_arbitrage_opportunities,
    format_distribution,
)


def generate_simulated_gefs(
    true_temp: float = 56.0,
    spread: float = 4.5,
    n_members: int = 21,
) -> dict:
    """
    Simulate GEFS ensemble: 21 members with realistic spread.

    Real GEFS ensemble spread for 24-48h forecasts is typically ±3-6°F.
    The control run (gec00) tends to be closest to truth.
    """
    np.random.seed(42)  # Reproducible for demo

    # Control run: small bias from truth
    control_temp = true_temp + np.random.normal(0, 1.0)

    # Perturbed members: wider spread
    perturbed_temps = np.random.normal(true_temp, spread, n_members - 1)

    all_temps = [round(float(control_temp), 1)] + [
        round(float(t), 1) for t in perturbed_temps
    ]

    members = {}
    members["gec00"] = {"24": round(float(control_temp), 1)}
    for i, temp in enumerate(perturbed_temps, 1):
        members[f"gep{i:02d}"] = {"24": round(float(temp), 1)}

    return {
        "source": "GEFS-simulated",
        "cycle": "20260211/12Z",
        "forecast_temps_f": all_temps,
        "member_details": members,
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }


def generate_simulated_ecmwf(
    true_temp: float = 56.0,
    spread: float = 3.8,
    n_members: int = 51,
) -> dict:
    """
    Simulate ECMWF ensemble: 51 members with tighter spread.

    ECMWF typically has tighter ensemble spread than GFS (better calibrated).
    The HRES (deterministic) run has ~0.5°F lower MAE than ensemble mean.
    """
    np.random.seed(123)

    # HRES: very close to truth
    hres_temp = true_temp + np.random.normal(0, 0.8)

    # Ensemble members
    temps = np.random.normal(true_temp + 0.3, spread, n_members)
    temps = [round(float(t), 1) for t in temps]

    members = {f"ens{i:02d}": t for i, t in enumerate(temps)}

    return {
        "source": "ECMWF-simulated",
        "cycle": "20260211/12Z",
        "forecast_temps_f": temps,
        "member_details": members,
        "hres_forecast_f": round(float(hres_temp), 1),
        "fetch_time": datetime.now(timezone.utc).isoformat(),
    }


def generate_sample_polymarket_prices(
    center: float = 54.0,  # Market consensus (slightly off from truth)
    std: float = 5.0,
) -> dict[str, float]:
    """
    Simulate Polymarket prices with deliberate inefficiencies.

    Real weather markets often:
    - Overprice tails (people buy extreme outcomes "just in case")
    - Lag model updates by 2-6 hours
    - Have wider bid-ask spreads on off-peak hours
    """
    prices = {}
    width = MARKET.bucket_width_f

    for lower in range(MARKET.bucket_min_f, MARKET.bucket_max_f, width):
        upper = lower + width
        if lower == MARKET.bucket_min_f:
            label = f"Below {upper}°F"
        elif upper >= MARKET.bucket_max_f:
            label = f"{lower}°F or above"
        else:
            label = f"{lower}°F to {upper - 1}°F"

        # True probability based on market consensus
        prob = stats.norm.cdf(upper, center, std) - stats.norm.cdf(lower, center, std)

        # Add market inefficiencies
        if prob < 0.02:
            # Tails are overpriced (gamblers buy extreme outcomes)
            prob = max(prob * 2.5, 0.015)
        elif prob > 0.15:
            # Peak is slightly underpriced
            prob *= 0.92

        prices[label] = round(max(0.01, prob), 4)

    # Normalize
    total = sum(prices.values())
    prices = {k: round(v / total, 4) for k, v in prices.items()}

    return prices


def main():
    print("\n" + "🌡️ " * 35)
    print("  WEATHER PREDICTION PIPELINE — DEMO")
    print(f"  Target: {MARKET.station.name} — {MARKET.target_date.strftime('%B %d, %Y')}")
    print("🌡️ " * 35 + "\n")

    # ── Simulate "truth" ────────────────────────────────────────────
    # Feb 12 Dallas: climatological average high is ~57°F
    # Let's say the actual high will be 56°F
    TRUE_TEMP = 56.0
    print(f"Simulated truth: {TRUE_TEMP}°F (unknown to model)\n")

    # ── Step 1: Generate simulated ensemble data ────────────────────
    print("=" * 70)
    print("[1/4] Generating simulated GEFS ensemble (21 members)...")
    gefs_data = generate_simulated_gefs(true_temp=TRUE_TEMP)
    print(f"  Members: {len(gefs_data['forecast_temps_f'])}")
    print(f"  Temps: {sorted(gefs_data['forecast_temps_f'])}")
    print(f"  Mean: {np.mean(gefs_data['forecast_temps_f']):.1f}°F")
    print(f"  Std:  {np.std(gefs_data['forecast_temps_f']):.1f}°F")

    print(f"\n[2/4] Generating simulated ECMWF ensemble (51 members)...")
    ecmwf_data = generate_simulated_ecmwf(true_temp=TRUE_TEMP)
    print(f"  Members: {len(ecmwf_data['forecast_temps_f'])}")
    print(f"  HRES: {ecmwf_data['hres_forecast_f']}°F")
    print(f"  Mean: {np.mean(ecmwf_data['forecast_temps_f']):.1f}°F")
    print(f"  Std:  {np.std(ecmwf_data['forecast_temps_f']):.1f}°F")

    # ── Step 2: Compute probability distribution ────────────────────
    print(f"\n[3/4] Computing probability distribution...")
    distribution = compute_distribution(gefs_data, ecmwf_data)

    output = format_distribution(distribution)
    print(f"\n{output}")

    # ── Step 3: Compare against Polymarket ──────────────────────────
    print(f"\n[4/4] Comparing against simulated Polymarket prices...")
    print("  (Market consensus: 54°F — lagging behind models by ~2°F)")

    market_prices = generate_sample_polymarket_prices(center=54.0)

    print(f"\n  {'Bucket':<22} {'Market':>7} {'Model':>7} {'Delta':>7}")
    print(f"  {'─' * 22} {'─' * 7} {'─' * 7} {'─' * 7}")
    for bucket in distribution.buckets:
        if bucket.label in market_prices:
            mp = market_prices[bucket.label]
            delta = bucket.probability - mp
            if abs(delta) > 0.01:
                arrow = "↑" if delta > 0 else "↓"
                print(
                    f"  {bucket.label:<22} {mp:>6.1%} {bucket.probability:>6.1%} "
                    f"{delta:>+6.1%} {arrow}"
                )

    # ── Step 4: Find arbitrage ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ARBITRAGE OPPORTUNITIES")
    print(f"{'=' * 70}")

    opportunities = find_arbitrage_opportunities(distribution, market_prices)

    if opportunities:
        for opp in opportunities:
            print(f"\n  {'🟢' if opp['action'] == 'BUY_YES' else '🔴'} {opp['action']}: \"{opp['bucket']}\"")
            print(f"     Market YES price: {opp['market_yes']:.1%}")
            print(f"     Model probability: {opp['model_prob']:.1%}")
            print(f"     Edge: {opp['edge']:.1%}")
            print(f"     Expected value: ${opp['expected_value']:.3f} per $1 [{opp['confidence']}]")
    else:
        print("\n  No significant arbitrage opportunities found.")

    # ── Save results ────────────────────────────────────────────────
    data_dir = Path("data/demo")
    data_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "true_temp_f": TRUE_TEMP,
        "gefs_mean": round(float(np.mean(gefs_data["forecast_temps_f"])), 1),
        "ecmwf_mean": round(float(np.mean(ecmwf_data["forecast_temps_f"])), 1),
        "combined_mean": round(distribution.ensemble_mean_f, 1),
        "most_likely_bucket": distribution.most_likely_bucket,
        "most_likely_prob": round(distribution.most_likely_prob, 4),
        "market_prices": market_prices,
        "arbitrage": opportunities,
        "buckets": [
            {"label": b.label, "prob": round(b.probability, 4)}
            for b in distribution.buckets
            if b.probability > 0.001
        ],
    }

    output_file = data_dir / "demo_results.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {output_file}")

    print(f"\n{'🌡️ ' * 35}")
    print("  DEMO COMPLETE")
    print(f"  The model ensemble mean ({distribution.ensemble_mean_f:.1f}°F) vs truth ({TRUE_TEMP}°F)")
    print(f"  Most likely bucket: {distribution.most_likely_bucket} ({distribution.most_likely_prob:.1%})")
    print(f"{'🌡️ ' * 35}\n")


if __name__ == "__main__":
    main()
