"""
Fetch apartment price data from Statistics Finland PxWeb API.

Sources:
  - ASHI table 13mu: prices per sqm of old dwellings by postal code, yearly
  - ASHI table 13mx: prices per sqm of old dwellings by municipality, yearly

Run with: python app/fetch_data.py
"""

import json
import os
import time
import requests
import pandas as pd
import itertools

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://pxdata.stat.fi/PXWeb/api/v1/en/StatFin"


def get_metadata(table_id: str) -> dict:
    """Fetch variable metadata for a PxWeb table."""
    url = f"{BASE_URL}/{table_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_chunk(table_id: str, selections: dict) -> dict:
    """
    POST a query selecting explicit values for each variable.
    selections: {code: [val1, val2, ...]}  — all codes must be present.
    """
    url = f"{BASE_URL}/{table_id}"
    query = {
        "query": [
            {"code": code, "selection": {"filter": "item", "values": vals}}
            for code, vals in selections.items()
        ],
        "response": {"format": "json-stat2"},
    }
    resp = requests.post(url, json=query, timeout=120)
    if not resp.ok:
        print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    return resp.json()


def jsonstat2_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert a JSON-stat2 response to a tidy pandas DataFrame."""
    dims = list(data["id"])
    labels = {
        dim: list(data["dimension"][dim]["category"]["label"].values())
        for dim in dims
    }
    codes = {
        dim: list(data["dimension"][dim]["category"]["index"].keys())
        for dim in dims
    }

    rows = []
    for idx, combo in enumerate(itertools.product(*[range(len(labels[d])) for d in dims])):
        row = {dim: labels[dim][combo[i]] for i, dim in enumerate(dims)}
        row["value"] = data["value"][idx]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Postal-code level prices (table 13mu)
#    Dimensions: Year × Postal code × Building type × Information
#    Strategy: fetch one building type at a time to stay under cell limit
# ---------------------------------------------------------------------------

def fetch_ashi_postal_code():
    table_id = "statfin_ashi_pxt_13mu.px"
    print(f"Fetching metadata for {table_id} ...")
    meta = get_metadata(table_id)

    var_map = {v["code"]: v for v in meta["variables"]}
    for code, v in var_map.items():
        print(f"  {code}: {v['text']} ({len(v['values'])} values)")

    years       = var_map["Vuosi"]["values"]
    postcodes   = var_map["Postinumero"]["values"]
    btypes      = var_map["Talotyyppi"]["values"]
    btype_texts = var_map["Talotyyppi"]["valueTexts"]
    info_codes  = var_map["Tiedot"]["values"]

    chunks = []
    for btype, btext in zip(btypes, btype_texts):
        print(f"\n  Fetching building type: {btext} ...")
        selections = {
            "Vuosi":       years,
            "Postinumero": postcodes,
            "Talotyyppi":  [btype],
            "Tiedot":      info_codes,
        }
        data = fetch_chunk(table_id, selections)
        df = jsonstat2_to_dataframe(data)
        chunks.append(df)
        time.sleep(1)   # be polite to the API

    df_all = pd.concat(chunks, ignore_index=True)

    # Save
    csv_path = os.path.join(RAW_DIR, "ashi_postal_code.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Saved CSV ({len(df_all):,} rows) -> {csv_path}")
    return df_all


# ---------------------------------------------------------------------------
# 2. Municipality level prices (table 13mx)
#    Dimensions: Year × Municipality × Building type × Information
# ---------------------------------------------------------------------------

def fetch_ashi_municipality():
    table_id = "statfin_ashi_pxt_13mx.px"
    print(f"\nFetching metadata for {table_id} ...")
    meta = get_metadata(table_id)

    var_map = {v["code"]: v for v in meta["variables"]}
    for code, v in var_map.items():
        print(f"  {code}: {v['text']} ({len(v['values'])} values)")

    years       = var_map["Vuosi"]["values"]
    munis       = var_map["Kunta"]["values"]
    btypes      = var_map["Talotyyppi"]["values"]
    btype_texts = var_map["Talotyyppi"]["valueTexts"]
    info_codes  = var_map["Tiedot"]["values"]

    chunks = []
    for btype, btext in zip(btypes, btype_texts):
        print(f"\n  Fetching building type: {btext} ...")
        selections = {
            "Vuosi":      years,
            "Kunta":      munis,
            "Talotyyppi": [btype],
            "Tiedot":     info_codes,
        }
        data = fetch_chunk(table_id, selections)
        df = jsonstat2_to_dataframe(data)
        chunks.append(df)
        time.sleep(1)

    df_all = pd.concat(chunks, ignore_index=True)

    csv_path = os.path.join(RAW_DIR, "ashi_municipality.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Saved CSV ({len(df_all):,} rows) -> {csv_path}")
    return df_all


if __name__ == "__main__":
    print("=" * 60)
    print("ASHI: Postal code level prices")
    print("=" * 60)
    df_postal = fetch_ashi_postal_code()

    print("\n" + "=" * 60)
    print("ASHI: Municipality level prices")
    print("=" * 60)
    df_muni = fetch_ashi_municipality()

    print("\n\nAll done.")
    print(f"\nPostal code data shape: {df_postal.shape}")
    print(df_postal.head(6).to_string())
    print(f"\nMunicipality data shape: {df_muni.shape}")
    print(df_muni.head(6).to_string())
