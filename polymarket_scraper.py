"""
Polymarket Weather Market Price Scraper
Fetches current YES prices for temperature bucket markets via the Gamma API.

Uses the event slug to do a single lookup that returns all sub-markets
(temperature buckets) with their current prices.

API: https://gamma-api.polymarket.com/events?slug=<event-slug>
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import httpx

from config import DATA_DIR

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


async def fetch_polymarket_prices(event_slug: str) -> dict[str, float]:
    """
    Fetch current Polymarket YES prices for all buckets in a weather event.

    Args:
        event_slug: The event slug from the Polymarket URL, e.g.
                    "highest-temperature-in-dallas-on-february-12-2026"

    Returns:
        {bucket_label: yes_price, ...}
        e.g., {"69°F or below": 0.01, "70-71°F": 0.04, "78-79°F": 0.39, ...}
    """
    prices = {}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{GAMMA_API}/events",
                params={"slug": event_slug},
            )
            resp.raise_for_status()
            events = resp.json()

            if not events:
                logger.warning(f"No event found for slug: {event_slug}")
                return prices

            event = events[0]
            logger.info(f"Found event: {event['title']} (id={event['id']})")

            for market in event.get("markets", []):
                label = market.get("groupItemTitle", market.get("question", ""))
                outcome_prices = json.loads(market.get("outcomePrices", "[]"))
                # outcomePrices[0] = YES price, outcomePrices[1] = NO price
                if outcome_prices:
                    yes_price = float(outcome_prices[0])
                    prices[label] = yes_price
                    logger.info(f"  {label}: YES @ ${yes_price:.3f}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Gamma API HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.warning(f"Polymarket fetch failed: {e}")

    if prices:
        logger.info(f"Retrieved {len(prices)} Polymarket prices (sum={sum(prices.values()):.3f})")
        output_dir = Path(DATA_DIR) / "polymarket"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"prices_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        output_file.write_text(json.dumps(prices, indent=2))
        logger.info(f"Saved to {output_file}")
    else:
        logger.warning("No Polymarket prices retrieved")

    return prices


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    prices = asyncio.run(
        fetch_polymarket_prices("highest-temperature-in-dallas-on-february-12-2026")
    )
    if prices:
        print("\n── Polymarket Prices ──")
        for label, price in sorted(prices.items()):
            print(f"  {label:20s}  YES @ ${price:.3f}  ({price*100:.1f}%)")
        print(f"  {'TOTAL':20s}        {sum(prices.values()):.3f}")
