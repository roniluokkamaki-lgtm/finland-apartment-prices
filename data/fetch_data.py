"""
Fetch Finnish apartment price data from Statistics Finland (stat.fi) PxWeb API.
Saves raw data to data/raw/.
"""

import json
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Statistics Finland PxWeb API — apartment prices by area and quarter
API_URL = (
    "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin/ashi/statfin_ashi_pxt_112p.px"
)


def fetch_apartment_prices():
    """Query the StatFin API for apartment price data and save to CSV."""
    # Query body — adjust years/regions as needed
    query = {
        "query": [
            {
                "code": "Alue",
                "selection": {"filter": "item", "values": ["pks", "091", "049", "837", "853"]},
            },
            {
                "code": "Talotyyppi",
                "selection": {"filter": "item", "values": ["1", "3"]},  # kerrostalo, rivitalo
            },
        ],
        "response": {"format": "json-stat2"},
    }

    print("Querying Statistics Finland API...")
    response = requests.post(API_URL, json=query, timeout=30)
    response.raise_for_status()

    data = response.json()
    out_path = RAW_DIR / "statfin_raw.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Raw JSON saved to {out_path}")
    return data


if __name__ == "__main__":
    fetch_apartment_prices()
