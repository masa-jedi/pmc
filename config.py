"""
Configuration for Weather Prediction Data Ingestion Pipeline
Supports multiple markets (Dallas, Seoul, etc.)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


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
    bucket_width: int = 2       # Bucket width in market units (2°F or 1°C)
    bucket_min: int = 70        # Regular buckets start here
    bucket_max: int = 82        # Triggers last catch-all bucket
    unit: str = "F"             # "F" or "C"
    utc_offset_hours: int = -6  # Local timezone offset from UTC
    timezone_str: str = "America/Chicago"  # For Open-Meteo API
    sources: tuple = ("gefs", "ecmwf", "hrrr", "nws", "openmeteo")

    def kelvin_to_unit(self, k: float) -> float:
        """Convert Kelvin to the market's temperature unit."""
        if self.unit == "C":
            return k - 273.15
        return (k - 273.15) * 9 / 5 + 32

    def celsius_to_unit(self, c: float) -> float:
        """Convert Celsius to the market's temperature unit."""
        if self.unit == "C":
            return c
        return c * 9 / 5 + 32

    @property
    def unit_symbol(self) -> str:
        return f"°{self.unit}"

    @property
    def utc_offset(self) -> timedelta:
        return timedelta(hours=self.utc_offset_hours)


# ── Stations ────────────────────────────────────────────────────────────────
DALLAS_LOVE_FIELD = StationConfig(
    name="Dallas Love Field",
    icao="KDAL",
    lat=32.847,
    lon=-96.852,  # Negative = West
)

INCHEON_INTL = StationConfig(
    name="Incheon Intl Airport",
    icao="RKSI",
    lat=37.4692,
    lon=126.4505,
)

# ── Markets ────────────────────────────────────────────────────────────────
MARKET_DALLAS = TargetMarket(
    station=DALLAS_LOVE_FIELD,
    target_date=datetime(2026, 2, 14, tzinfo=timezone.utc),
    bucket_width=2,
    bucket_min=54,    # Regular buckets: 70-71, 72-73, ..., 78-79
    bucket_max=71,    # Last bucket: "80°F or higher"
    unit="F",
    utc_offset_hours=-6,
    timezone_str="America/Chicago",
    sources=("gefs", "ecmwf", "hrrr", "nws", "openmeteo"),
)

MARKET_SEOUL = TargetMarket(
    station=INCHEON_INTL,
    target_date=datetime(2026, 2, 15, tzinfo=timezone.utc),
    bucket_width=1,
    bucket_min=0,     # Regular buckets: 4, 5, 6, 7, 8
    bucket_max=10,    # Last bucket: "9°C or higher"
    unit="C",
    utc_offset_hours=9,
    timezone_str="Asia/Seoul",
    sources=("ecmwf", "openmeteo", "metar"),  # GEFS doesn't serve Korea well via NOMADS
)

MARKETS = {
    "dallas": MARKET_DALLAS,
    "seoul": MARKET_SEOUL,
}

# ── Active Market (set via set_market() before importing fetchers) ─────────
MARKET = MARKET_DALLAS


def set_market(name: str) -> None:
    """Switch the active market. Must be called before fetcher imports."""
    global MARKET
    if name not in MARKETS:
        raise ValueError(f"Unknown market '{name}'. Available: {list(MARKETS.keys())}")
    MARKET = MARKETS[name]

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
BLEND_WEIGHT_METAR = 0.05  # METAR is observation, small weight for reality anchoring

# ── Pipeline Settings ──────────────────────────────────────────────────────
DATA_DIR = "data"
LOG_DIR = "logs"
FETCH_INTERVAL_SECONDS = 3600  # 1 hour
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30
HTTP_TIMEOUT_SECONDS = 60

# ── Weather.com Configuration ────────────────────────────────────────────────
# Historical observations API for US stations
WEATHER_COM_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
WEATHER_COM_BASE_URL = "https://api.weather.com/v1/location/{icao}:9:US/observations/historical.json"

# ── Probability Engine ──────────────────────────────────────────────────────
# Kernel Density Estimation bandwidth for smoothing ensemble spread
KDE_BANDWIDTH_F = 1.5   # °F bandwidth (Dallas)
KDE_BANDWIDTH_C = 0.8   # °C bandwidth (Seoul) — roughly equivalent
# Minimum probability to report
MIN_PROBABILITY_THRESHOLD = 0.001  # 0.1%
