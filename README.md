# 🌡️ Weather Prediction Pipeline for Polymarket

**Data ingestion pipeline that pulls NOAA GFS + ECMWF ensemble forecasts every hour and converts them into probability distributions across 2°F temperature buckets for Polymarket weather markets.**

Target: **Dallas Love Field Station (KDAL) — February 12, 2026**

## Architecture

```
┌───────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│  NOAA NOMADS      │    │  ECMWF Open Data   │    │  Polymarket API  │
│  GFS Ensemble     │    │  IFS Ensemble      │    │  (price feed)    │
│  21 members/6hr   │    │  51 members/12hr   │    │                  │
└────────┬──────────┘    └────────┬───────────┘    └────────┬─────────┘
         │                        │                         │
         ▼                        ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Probability Engine                              │
│  • KDE smoothing over ensemble members                             │
│  • Weighted combination: 40% GFS / 60% ECMWF                      │
│  • Output: probability per 2°F bucket                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Arbitrage Scanner                               │
│  • Model prob vs Polymarket price for each bucket                  │
│  • BUY NO: market >3%, model <1% → buy NO @ 98-99¢                │
│  • BUY YES: model >> market price → buy YES cheap                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo (no network required — uses simulated data)
python demo.py

# Single pipeline run (fetches live NOAA + ECMWF data)
python pipeline.py

# Continuous mode — runs every hour
python pipeline.py --continuous

# With Polymarket price comparison
python pipeline.py --prices data/polymarket/sample_prices.json

# Verbose logging
python pipeline.py -v
```

## How It Works

### Data Sources

| Source | Members | Update Frequency | Delay | Resolution |
|--------|---------|-----------------|-------|------------|
| NOAA GEFS | 21 (1 control + 20 perturbed) | Every 6h (00/06/12/18Z) | 4-6h | 0.25° |
| ECMWF IFS ENS | 51 (1 control + 50 perturbed) | Every 12h (00/12Z) | 7-8h | 0.25° |

### Probability Calculation

Each ensemble member represents an equally-likely weather scenario. With 72 total members:

1. **Extract**: Pull 2m temperature forecasts for Dallas Love Field's grid point
2. **Max temp**: For each member, find the maximum temperature during Feb 12 daytime (6AM–6PM CST)
3. **KDE smoothing**: Fit a Gaussian kernel density estimate over all member forecasts
4. **Bucket integration**: Integrate the KDE over each 2°F range (e.g., 54°F–56°F)
5. **Weighted blend**: 40% GFS + 60% ECMWF (ECMWF gets more weight due to better calibration and more members)

### Arbitrage Strategy

```
For each 2°F bucket:
  market_price = Polymarket YES token price (implied probability)
  model_prob   = Our ensemble probability

  IF market_price > 3% AND model_prob < 1%:
    → BUY NO at 97-99¢  (market overpriced this outcome)
    → Expected profit: ~2-3¢ per contract

  IF model_prob > 1.5 × market_price AND model_prob > 5%:
    → BUY YES cheap  (market hasn't caught up to model)
    → Expected profit: varies
```

## File Structure

```
weather-pipeline/
├── pipeline.py              # Main orchestrator (run this)
├── gfs_fetcher.py           # NOAA GFS Ensemble data fetcher
├── ecmwf_fetcher.py         # ECMWF Open Data fetcher
├── probability_engine.py    # Ensemble → probability buckets + arbitrage
├── polymarket_scraper.py    # Polymarket price feed
├── config.py                # All configuration & constants
├── demo.py                  # Demo with simulated data
├── requirements.txt
└── data/
    ├── runs/                # Pipeline run outputs (JSON)
    ├── ecmwf/               # Downloaded ECMWF GRIB files
    ├── grib/                # Downloaded GFS GRIB files
    └── polymarket/          # Cached market prices
```

## Production Deployment ($5/month server)

```bash
# On a $5 VPS (DigitalOcean/Vultr/Hetzner)
# Add to crontab to run every hour:
crontab -e

# Add this line:
0 * * * * cd /opt/weather-pipeline && python pipeline.py --prices data/polymarket/latest.json >> /var/log/weather-pipeline.log 2>&1
```

## Extending to 20 Cities

The pipeline is designed for a single station but easily scales. In `config.py`:

```python
STATIONS = [
    StationConfig("Dallas Love Field", "KDAL", 32.847, -96.852),
    StationConfig("JFK Airport", "KJFK", 40.640, -73.779),
    StationConfig("LAX Airport", "KLAX", 33.943, -118.408),
    # ... 17 more
]
```

Then in `pipeline.py`, wrap `run_pipeline()` with `asyncio.gather()` across all stations.

## Optional: Better GRIB Decoding

The lightweight fetcher uses heuristic GRIB parsing. For production, install proper GRIB tools:

```bash
pip install cfgrib eccodes xarray ecmwf-opendata
```

This enables the `fetch_gefs_with_cfgrib()` path which properly decodes GRIB2 messages using ECMWF's eccodes library.
