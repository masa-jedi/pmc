"""
Configuration for Weather Prediction Data Ingestion Pipeline
Target: Dallas Love Field Station (KDAL) — Feb 12, 2026
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class StationConfig:
    name: str
    icao: str
    lat: float
    lon: float


@dataclass(frozen=True)
class TargetMarket:
    station: StationConfig
    target_date: datetime
    bucket_width_f: int = 2  # 2°F buckets to match Polymarket
    # Range adjusted to match Polymarket's Feb 12 Dallas market (69-80°F+)
    # Extended to cover forecast uncertainty
    bucket_min_f: int = 70  # Regular buckets start at "70-71°F" (first catch-all is "69°F or below")
    bucket_max_f: int = 82  # Last bucket is "80°F or higher" (range goes 70, 72, 74, 76, 78, 80)


# ── Stations ────────────────────────────────────────────────────────────────
DALLAS_LOVE_FIELD = StationConfig(
    name="Dallas Love Field",
    icao="KDAL",
    lat=32.847,
    lon=-96.852,  # Negative = West
)

# ── Target Market ───────────────────────────────────────────────────────────
MARKET = TargetMarket(
    station=DALLAS_LOVE_FIELD,
    target_date=datetime(2026, 2, 12, tzinfo=timezone.utc),
)

# ── GFS Ensemble (GEFS) Configuration ──────────────────────────────────────
# NOMADS GRIB filter endpoint for GEFS 0.25° resolution
GEFS_BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"

# GEFS runs at 00Z, 06Z, 12Z, 18Z — 21 members (gec00 = control, gep01-gep20 = perturbed)
GEFS_CYCLES = ["00", "06", "12", "18"]
GEFS_CONTROL_MEMBER = "gec00"
GEFS_PERTURBED_MEMBERS = [f"gep{i:02d}" for i in range(1, 21)]
GEFS_ALL_MEMBERS = [GEFS_CONTROL_MEMBER] + GEFS_PERTURBED_MEMBERS  # 21 total

# Variable: 2m temperature (TMP), level: 2 m above ground
GEFS_VARIABLE = "TMP"
GEFS_LEVEL = "2_m_above_ground"

# ── ECMWF Open Data Configuration ──────────────────────────────────────────
# ECMWF publishes free open data (IFS HRES + ENS) at:
ECMWF_OPEN_BASE = "https://data.ecmwf.int/forecasts"
# Ensemble has 51 members (0 = control, 1-50 = perturbed)
ECMWF_ENS_MEMBERS = 51
ECMWF_CYCLES = ["00", "12"]  # ECMWF runs 00Z and 12Z

# ── HRRR Configuration ───────────────────────────────────────────────────
# NOMADS GRIB filter endpoint for HRRR 3km CONUS
HRRR_BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
HRRR_N_CYCLES = 4  # Fetch last N hourly cycles as pseudo-ensemble

# ── Blend Weights ────────────────────────────────────────────────────────
# Normalized automatically if a source is missing.
# Open-Meteo gets top weight: 7 independent models (GFS, ECMWF, JMA, ICON,
# GEM, Météo-France, best_match) provide true multi-model diversity.
# NWS is human-corrected — highly valuable when available.
# HRRR/ECMWF/GEFS are correlated (share physics/init) and had cold bias.
BLEND_WEIGHT_NWS = 0.30
BLEND_WEIGHT_HRRR = 0.10
BLEND_WEIGHT_OPENMETEO = 0.55
BLEND_WEIGHT_ECMWF = 0.025
BLEND_WEIGHT_GEFS = 0.025

# ── Pipeline Settings ──────────────────────────────────────────────────────
DATA_DIR = "data"
LOG_DIR = "logs"
FETCH_INTERVAL_SECONDS = 3600  # 1 hour
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30
HTTP_TIMEOUT_SECONDS = 60

# ── Probability Engine ──────────────────────────────────────────────────────
# Kernel Density Estimation bandwidth (°F) for smoothing ensemble spread
KDE_BANDWIDTH_F = 1.5
# Minimum probability to report
MIN_PROBABILITY_THRESHOLD = 0.001  # 0.1%
