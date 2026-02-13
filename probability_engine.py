"""
Probability Engine
Converts raw ensemble temperature forecasts into probability distributions
across 2°F buckets — the format Polymarket weather markets use.

Methods:
1. Simple histogram: count members in each bucket
2. KDE (Kernel Density Estimation): smooth probability surface
3. Combined: weighted blend of GEFS + ECMWF ensembles

The key insight: each ensemble member is an equally-likely future.
21 GEFS members + 51 ECMWF members = 72 independent samples of tomorrow's temperature.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy import stats

from config import (
    BLEND_WEIGHT_ECMWF,
    BLEND_WEIGHT_GEFS,
    BLEND_WEIGHT_HRRR,
    BLEND_WEIGHT_NWS,
    BLEND_WEIGHT_OPENMETEO,
    KDE_BANDWIDTH_F,
    MARKET,
    MIN_PROBABILITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class TemperatureBucket:
    """A 2°F temperature range with associated probability."""
    lower_f: int  # inclusive
    upper_f: int  # exclusive
    label: str    # e.g., "52°F to 54°F"
    probability: float = 0.0
    gefs_prob: float = 0.0
    ecmwf_prob: float = 0.0
    hrrr_prob: float = 0.0
    nws_prob: float = 0.0
    openmeteo_prob: float = 0.0
    member_count: int = 0  # How many ensemble members fell in this bucket


@dataclass
class ProbabilityDistribution:
    """Full probability distribution across all temperature buckets."""
    buckets: list[TemperatureBucket] = field(default_factory=list)
    ensemble_mean_f: float = 0.0
    ensemble_std_f: float = 0.0
    ensemble_min_f: float = 0.0
    ensemble_max_f: float = 0.0
    gefs_count: int = 0
    ecmwf_count: int = 0
    hrrr_count: int = 0
    nws_count: int = 0
    openmeteo_count: int = 0
    total_members: int = 0
    method: str = "kde"
    most_likely_bucket: str = ""
    most_likely_prob: float = 0.0


def _create_buckets() -> list[TemperatureBucket]:
    """Create 2°F temperature buckets covering the expected range.

    Polymarket format: first bucket is "69°F or below" (catch-all ≤69),
    then 2°F ranges (70-71, 72-73, ..., 78-79), then "80°F or higher".
    """
    buckets = []
    width = MARKET.bucket_width_f

    # First bucket: "69°F or below" (catch-all for temps ≤ bucket_min_f - 1)
    # Use upper = bucket_min_f so that temps < bucket_min_f are in range [lower, upper)
    label = f"{MARKET.bucket_min_f - 1}°F or below"
    buckets.append(TemperatureBucket(lower_f=-100, upper_f=MARKET.bucket_min_f, label=label))

    # Regular 2°F buckets starting from bucket_min_f
    for lower in range(MARKET.bucket_min_f, MARKET.bucket_max_f, width):
        upper = lower + width
        if upper >= MARKET.bucket_max_f:
            # Last bucket: "80°F or higher"
            label = f"{lower}°F or higher"
            upper = 999  # Effectively infinity
        else:
            # Middle buckets: "70-71°F"
            label = f"{lower}-{upper - 1}°F"
        buckets.append(TemperatureBucket(lower_f=lower, upper_f=upper, label=label))

    return buckets


def histogram_probabilities(
    temps_f: list[float],
) -> list[TemperatureBucket]:
    """
    Simple method: count how many ensemble members land in each bucket.
    Probability = count / total_members.
    """
    buckets = _create_buckets()
    n = len(temps_f)

    if n == 0:
        return buckets

    for temp in temps_f:
        for bucket in buckets:
            if bucket.lower_f <= temp < bucket.upper_f:
                bucket.member_count += 1
                break
        else:
            # Temperature outside our range — assign to nearest edge bucket
            if temp < MARKET.bucket_min_f:
                buckets[0].member_count += 1
            else:
                buckets[-1].member_count += 1

    for bucket in buckets:
        bucket.probability = bucket.member_count / n

    return buckets


def kde_probabilities(
    temps_f: list[float],
    bandwidth: float = KDE_BANDWIDTH_F,
) -> list[TemperatureBucket]:
    """
    Kernel Density Estimation: fit a smooth probability surface over the
    ensemble members, then integrate over each 2°F bucket.

    This gives more realistic probabilities especially in bucket boundaries
    and tails where simple counting would show 0%.
    """
    buckets = _create_buckets()
    temps = np.array(temps_f)
    n = len(temps)

    if n < 3:
        # Too few points for KDE — fall back to histogram
        return histogram_probabilities(temps_f)

    try:
        kde = stats.gaussian_kde(temps, bw_method=bandwidth / np.std(temps))
    except (np.linalg.LinAlgError, ValueError):
        logger.warning("KDE failed, falling back to histogram")
        return histogram_probabilities(temps_f)

    # Integrate KDE over each bucket
    for bucket in buckets:
        # Use numerical integration with fine grid
        x = np.linspace(bucket.lower_f, bucket.upper_f, 100)
        pdf_values = kde(x)
        bucket.probability = float(np.trapezoid(pdf_values, x))

        # Also count members for reference
        for temp in temps:
            if bucket.lower_f <= temp < bucket.upper_f:
                bucket.member_count += 1

    # Normalize so probabilities sum to 1.0
    total_prob = sum(b.probability for b in buckets)
    if total_prob > 0:
        for bucket in buckets:
            bucket.probability /= total_prob

    return buckets


def compute_distribution(
    gefs_data: dict,
    ecmwf_data: dict,
    hrrr_data: dict | None = None,
    nws_data: dict | None = None,
    openmeteo_data: dict | None = None,
    gefs_weight: float = BLEND_WEIGHT_GEFS,
    ecmwf_weight: float = BLEND_WEIGHT_ECMWF,
    hrrr_weight: float = BLEND_WEIGHT_HRRR,
    nws_weight: float = BLEND_WEIGHT_NWS,
    openmeteo_weight: float = BLEND_WEIGHT_OPENMETEO,
) -> ProbabilityDistribution:
    """
    Compute combined probability distribution from up to 5 sources:
    GEFS + ECMWF + HRRR + NWS + Open-Meteo.

    Weights are normalized automatically based on which sources have data.
    """
    gefs_temps = gefs_data.get("forecast_temps_f", [])
    ecmwf_temps = ecmwf_data.get("forecast_temps_f", [])
    hrrr_temps = hrrr_data.get("forecast_temps_f", []) if hrrr_data else []
    nws_temps = nws_data.get("forecast_temps_f", []) if nws_data else []
    openmeteo_temps = openmeteo_data.get("forecast_temps_f", []) if openmeteo_data else []

    # Zero out weights for missing sources
    if not gefs_temps:
        gefs_weight = 0.0
    if not ecmwf_temps:
        ecmwf_weight = 0.0
    if not hrrr_temps:
        hrrr_weight = 0.0
    if not nws_temps:
        nws_weight = 0.0
    if not openmeteo_temps:
        openmeteo_weight = 0.0

    total_weight = gefs_weight + ecmwf_weight + hrrr_weight + nws_weight + openmeteo_weight
    if total_weight == 0:
        logger.error("No forecast data available from any source!")
        return ProbabilityDistribution(buckets=_create_buckets(), method="none")

    gefs_weight /= total_weight
    ecmwf_weight /= total_weight
    hrrr_weight /= total_weight
    nws_weight /= total_weight
    openmeteo_weight /= total_weight

    # ── Compute individual distributions ────────────────────────────
    gefs_buckets = kde_probabilities(gefs_temps) if gefs_temps else _create_buckets()
    ecmwf_buckets = kde_probabilities(ecmwf_temps) if ecmwf_temps else _create_buckets()
    hrrr_buckets = kde_probabilities(hrrr_temps) if hrrr_temps else _create_buckets()
    nws_buckets = kde_probabilities(nws_temps) if nws_temps else _create_buckets()
    openmeteo_buckets = kde_probabilities(openmeteo_temps) if openmeteo_temps else _create_buckets()

    # ── Weighted combination ────────────────────────────────────────
    combined_buckets = _create_buckets()
    for i, bucket in enumerate(combined_buckets):
        bucket.gefs_prob = gefs_buckets[i].probability if gefs_temps else 0.0
        bucket.ecmwf_prob = ecmwf_buckets[i].probability if ecmwf_temps else 0.0
        bucket.hrrr_prob = hrrr_buckets[i].probability if hrrr_temps else 0.0
        bucket.nws_prob = nws_buckets[i].probability if nws_temps else 0.0
        bucket.openmeteo_prob = openmeteo_buckets[i].probability if openmeteo_temps else 0.0
        bucket.probability = (
            gefs_weight * bucket.gefs_prob
            + ecmwf_weight * bucket.ecmwf_prob
            + hrrr_weight * bucket.hrrr_prob
            + nws_weight * bucket.nws_prob
            + openmeteo_weight * bucket.openmeteo_prob
        )
        bucket.member_count = (
            gefs_buckets[i].member_count
            + ecmwf_buckets[i].member_count
            + hrrr_buckets[i].member_count
            + nws_buckets[i].member_count
            + openmeteo_buckets[i].member_count
        )

    # Normalize combined
    total_prob = sum(b.probability for b in combined_buckets)
    if total_prob > 0:
        for b in combined_buckets:
            b.probability /= total_prob

    # ── Aggregate statistics ────────────────────────────────────────
    all_temps = np.array(gefs_temps + ecmwf_temps + hrrr_temps + nws_temps + openmeteo_temps)

    best_bucket = max(combined_buckets, key=lambda b: b.probability)

    dist = ProbabilityDistribution(
        buckets=combined_buckets,
        ensemble_mean_f=float(np.mean(all_temps)) if len(all_temps) > 0 else 0.0,
        ensemble_std_f=float(np.std(all_temps)) if len(all_temps) > 0 else 0.0,
        ensemble_min_f=float(np.min(all_temps)) if len(all_temps) > 0 else 0.0,
        ensemble_max_f=float(np.max(all_temps)) if len(all_temps) > 0 else 0.0,
        gefs_count=len(gefs_temps),
        ecmwf_count=len(ecmwf_temps),
        hrrr_count=len(hrrr_temps),
        nws_count=len(nws_temps),
        openmeteo_count=len(openmeteo_temps),
        total_members=len(all_temps),
        method="kde_weighted",
        most_likely_bucket=best_bucket.label,
        most_likely_prob=best_bucket.probability,
    )

    return dist


def format_distribution(dist: ProbabilityDistribution) -> str:
    """Pretty-print the probability distribution for logging/display."""
    lines = [
        "=" * 70,
        f"  TEMPERATURE PROBABILITY DISTRIBUTION — {MARKET.station.name}",
        f"  Target: {MARKET.target_date.strftime('%B %d, %Y')} (Daily High)",
        f"  Method: {dist.method} | Sources: {dist.total_members} "
        f"(GEFS:{dist.gefs_count} ECMWF:{dist.ecmwf_count} HRRR:{dist.hrrr_count} "
        f"NWS:{dist.nws_count} OM:{dist.openmeteo_count})",
        f"  Ensemble Mean: {dist.ensemble_mean_f:.1f}°F ± {dist.ensemble_std_f:.1f}°F",
        f"  Range: {dist.ensemble_min_f:.1f}°F — {dist.ensemble_max_f:.1f}°F",
        "=" * 70,
        "",
        f"  {'Bucket':<22} {'Prob':>7} {'Bar':<24} {'GEFS':>6} {'ECMWF':>6} {'HRRR':>6} {'NWS':>6} {'OM':>6}",
        f"  {'─' * 22} {'─' * 7} {'─' * 24} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6}",
    ]

    for bucket in dist.buckets:
        bar_len = int(bucket.probability * 50)
        bar = "█" * bar_len

        lines.append(
            f"  {bucket.label:<22} {bucket.probability:>6.1%} {bar:<24} "
            f"{bucket.gefs_prob:>5.1%} {bucket.ecmwf_prob:>5.1%} {bucket.hrrr_prob:>5.1%} "
            f"{bucket.nws_prob:>5.1%} {bucket.openmeteo_prob:>5.1%}"
        )

    lines.extend([
        "",
        f"  ★ Most Likely: {dist.most_likely_bucket} ({dist.most_likely_prob:.1%})",
        "=" * 70,
    ])

    return "\n".join(lines)


_STATION_UTC_OFFSET = timedelta(hours=-6)  # Dallas CST


def apply_reality_check(
    gefs_data: dict,
    ecmwf_data: dict,
    current_temp_f: float,
    observed_high_f: float,
    observed_high_time: datetime | None = None,
    observed_at: datetime | None = None,
    hrrr_data: dict | None = None,
    nws_data: dict | None = None,
    openmeteo_data: dict | None = None,
    peak_passed_threshold_f: float = 1.5,
) -> "ProbabilityDistribution":
    """
    Apply real-world observations to anchor the forecast distribution to reality.

    Steps:
    1. Bias correction — shift ensemble members by (current_obs - model_avg)
    2. Impossibility check — zero out buckets already exceeded by observed high
    3. Time-aware peak detection — if temps are declining from observed high,
       discount higher buckets. Two time dimensions are considered:
       a) Current local time gates the threshold (larger drop required before noon)
          and modulates the discount (weaker before peak heating at 2-5 PM).
       b) When the high was observed determines a secondary discount factor
          (overnight high → mild, afternoon high → strong).
       The more conservative of the two factors is used.
    4. Renormalize
    """
    gefs_temps = gefs_data.get("forecast_temps_f", [])
    ecmwf_temps = ecmwf_data.get("forecast_temps_f", [])
    hrrr_temps = hrrr_data.get("forecast_temps_f", []) if hrrr_data else []
    nws_temps = nws_data.get("forecast_temps_f", []) if nws_data else []
    openmeteo_temps = openmeteo_data.get("forecast_temps_f", []) if openmeteo_data else []
    all_raw = gefs_temps + ecmwf_temps + hrrr_temps + nws_temps + openmeteo_temps

    if not all_raw:
        return compute_distribution(
            gefs_data, ecmwf_data, hrrr_data=hrrr_data,
            nws_data=nws_data, openmeteo_data=openmeteo_data,
        )

    # ── 1. Bias correction ───────────────────────────────────────────────────
    model_avg = sum(all_raw) / len(all_raw)
    bias = current_temp_f - model_avg
    logger.info(
        f"Applying bias correction: Obs={current_temp_f}, ModelAvg={model_avg:.1f}, "
        f"Bias={bias:+.1f}°F"
    )

    gefs_shifted = {**gefs_data, "forecast_temps_f": [t + bias for t in gefs_temps]}
    ecmwf_shifted = {**ecmwf_data, "forecast_temps_f": [t + bias for t in ecmwf_temps]}
    hrrr_shifted = None
    if hrrr_data:
        hrrr_shifted = {**hrrr_data, "forecast_temps_f": [t + bias for t in hrrr_temps]}
    nws_shifted = None
    if nws_data:
        nws_shifted = {**nws_data, "forecast_temps_f": [t + bias for t in nws_temps]}
    openmeteo_shifted = None
    if openmeteo_data:
        openmeteo_shifted = {**openmeteo_data, "forecast_temps_f": [t + bias for t in openmeteo_temps]}
    dist = compute_distribution(
        gefs_shifted, ecmwf_shifted, hrrr_data=hrrr_shifted,
        nws_data=nws_shifted, openmeteo_data=openmeteo_shifted,
    )

    # ── 2. Impossibility check ───────────────────────────────────────────────
    logger.info(f"Applying reality check: Observed High = {observed_high_f}°F")

    invalidated = 0
    for bucket in dist.buckets:
        if bucket.upper_f <= observed_high_f:
            bucket.probability = 0.0
            bucket.gefs_prob = 0.0
            bucket.ecmwf_prob = 0.0
            bucket.hrrr_prob = 0.0
            bucket.nws_prob = 0.0
            bucket.openmeteo_prob = 0.0
            invalidated += 1

    remaining_mass = sum(b.probability for b in dist.buckets)
    if invalidated:
        logger.info(
            f"  Reality check invalidated {invalidated} buckets. "
            f"Renormalizing remaining {remaining_mass:.1%} mass."
        )

    # ── 3. Time-aware peak detection ────────────────────────────────────────
    # Determine current local hour from the observation timestamp
    now_local_hour: int | None = None
    if observed_at is not None:
        now_local_hour = (observed_at + _STATION_UTC_OFFSET).hour

    # Time-dependent threshold: require a larger temperature drop before
    # concluding the peak has passed when afternoon heating is still ahead.
    if now_local_hour is not None:
        if now_local_hour < 12:
            effective_threshold = 4.0
        elif now_local_hour < 14:
            effective_threshold = 2.5
        elif now_local_hour < 17:
            effective_threshold = peak_passed_threshold_f  # 1.5 (original)
        else:
            effective_threshold = 1.0
    else:
        effective_threshold = peak_passed_threshold_f  # fallback: original

    peak_passed = current_temp_f < observed_high_f - effective_threshold
    if peak_passed:
        # Secondary factor: when was the high observed?
        peak_discount = 0.15  # default: moderate-strong (unknown timing)
        high_local_hour: int | None = None

        if observed_high_time is not None:
            high_local_hour = (observed_high_time + _STATION_UTC_OFFSET).hour
            if high_local_hour < 10:
                peak_discount = 0.40
            elif high_local_hour < 14:
                peak_discount = 0.15
            else:
                peak_discount = 0.05

        # Primary factor: current time of day — weaken the discount when
        # afternoon heating is still plausible.
        if now_local_hour is not None:
            if now_local_hour < 12:
                time_factor = 0.70  # retain 70%
            elif now_local_hour < 14:
                time_factor = 0.55  # retain 55%
            elif now_local_hour < 17:
                time_factor = 0.30  # retain 30%
            else:
                time_factor = 0.10  # retain 10%
            # Use the more conservative (higher) of the two factors
            peak_discount = max(peak_discount, time_factor)

        high_hour_str = f" at local hour {high_local_hour:02d}:xx" if high_local_hour is not None else ""
        now_hour_str = f", now local hour {now_local_hour:02d}:xx" if now_local_hour is not None else ""
        logger.info(
            f"  Peak appears to have passed (current={current_temp_f}°F < "
            f"high={observed_high_f}°F − {effective_threshold}°F{high_hour_str}"
            f"{now_hour_str}). "
            f"Discounting buckets above {observed_high_f}°F by {peak_discount:.0%}."
        )

        for bucket in dist.buckets:
            if bucket.lower_f >= observed_high_f and bucket.probability > 0:
                bucket.probability *= peak_discount
                bucket.gefs_prob *= peak_discount
                bucket.ecmwf_prob *= peak_discount
                bucket.hrrr_prob *= peak_discount
                bucket.nws_prob *= peak_discount
                bucket.openmeteo_prob *= peak_discount

    # ── 4. Renormalize ───────────────────────────────────────────────────────
    total_combined = sum(b.probability for b in dist.buckets)
    total_gefs = sum(b.gefs_prob for b in dist.buckets)
    total_ecmwf = sum(b.ecmwf_prob for b in dist.buckets)
    total_hrrr = sum(b.hrrr_prob for b in dist.buckets)
    total_nws = sum(b.nws_prob for b in dist.buckets)
    total_openmeteo = sum(b.openmeteo_prob for b in dist.buckets)

    if total_combined > 0:
        for b in dist.buckets:
            b.probability /= total_combined
    if total_gefs > 0:
        for b in dist.buckets:
            b.gefs_prob /= total_gefs
    if total_ecmwf > 0:
        for b in dist.buckets:
            b.ecmwf_prob /= total_ecmwf
    if total_hrrr > 0:
        for b in dist.buckets:
            b.hrrr_prob /= total_hrrr
    if total_nws > 0:
        for b in dist.buckets:
            b.nws_prob /= total_nws
    if total_openmeteo > 0:
        for b in dist.buckets:
            b.openmeteo_prob /= total_openmeteo

    # Update summary stats (ensemble stats already reflect shifted temps from compute_distribution)
    best_bucket = max(dist.buckets, key=lambda b: b.probability)
    dist.most_likely_bucket = best_bucket.label
    dist.most_likely_prob = best_bucket.probability

    return dist


def find_arbitrage_opportunities(
    dist: ProbabilityDistribution,
    market_prices: dict[str, float] | None = None,
) -> list[dict]:
    """
    Compare model probabilities against Polymarket prices to find edges.

    The strategy from the prompt:
    - If Polymarket prices an outcome at >3% but model says <1% → buy NO at 98-99¢
    - If Polymarket prices an outcome at <X% but model says >Y% → buy YES cheap

    Args:
        market_prices: {bucket_label: polymarket_yes_price, ...}
                       e.g., {"52°F to 53°F": 0.05} means YES is $0.05 (5%)

    Returns:
        List of arbitrage opportunities with expected value
    """
    if market_prices is None:
        logger.info("No market prices provided — skipping arbitrage scan")
        return []

    opportunities = []

    for bucket in dist.buckets:
        label = bucket.label
        if label not in market_prices:
            continue

        market_yes_price = market_prices[label]
        market_no_price = 1.0 - market_yes_price
        model_prob = bucket.probability

        # ── Strategy 1: Buy NO when market overprices an outcome ────
        # Market says >3% but model says <1% → buy NO
        if market_yes_price > 0.03 and model_prob < 0.01:
            ev_no = (1.0 - model_prob) * (1.0 - market_no_price) - model_prob * market_no_price
            opportunities.append({
                "action": "BUY_NO",
                "bucket": label,
                "market_yes": market_yes_price,
                "market_no": market_no_price,
                "model_prob": model_prob,
                "edge": market_yes_price - model_prob,
                "expected_value": ev_no,
                "confidence": "HIGH" if model_prob < 0.005 else "MEDIUM",
            })

        # ── Strategy 2: Buy YES when market underprices an outcome ──
        # Model says significantly higher probability than market
        elif model_prob > market_yes_price * 1.5 and model_prob > 0.05:
            ev_yes = model_prob * (1.0 - market_yes_price) - (1.0 - model_prob) * market_yes_price
            opportunities.append({
                "action": "BUY_YES",
                "bucket": label,
                "market_yes": market_yes_price,
                "model_prob": model_prob,
                "edge": model_prob - market_yes_price,
                "expected_value": ev_yes,
                "confidence": "HIGH" if ev_yes > 0.10 else "MEDIUM",
            })

    # Sort by expected value
    opportunities.sort(key=lambda x: x["expected_value"], reverse=True)

    if opportunities:
        logger.info(f"Found {len(opportunities)} arbitrage opportunities:")
        for opp in opportunities:
            logger.info(
                f"  {opp['action']} {opp['bucket']}: "
                f"market={opp.get('market_yes', opp.get('market_no')):.1%}, "
                f"model={opp['model_prob']:.1%}, "
                f"edge={opp['edge']:.1%}, "
                f"EV={opp['expected_value']:.3f} [{opp['confidence']}]"
            )

    return opportunities
