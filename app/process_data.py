"""
Clean and process raw ASHI postal code data.

Steps:
  1. Pivot long -> wide (price per sqm + n_sales as separate columns)
  2. Split postal code column into code and area name
  3. Basic data checks
  4. Save processed CSV to data/processed/

Run with: python app/process_data.py
"""

import os
import pandas as pd
import numpy as np

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("Loading raw data ...")
df = pd.read_csv(os.path.join(RAW_DIR, "ashi_postal_code.csv"))
print(f"  Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Pivot to wide format
# ---------------------------------------------------------------------------
print("\nPivoting to wide format ...")

# Shorten the Tiedot values to usable column names before pivoting
tiedot_rename = {
    "Price per square meter (EUR/m2)":
        "price_eur_per_sqm",
    "Number of sales, asset transfer tax data starting from 2020":
        "n_sales",
}
df["Tiedot"] = df["Tiedot"].replace(tiedot_rename)

df_wide = df.pivot_table(
    index=["Vuosi", "Postinumero", "Talotyyppi"],
    columns="Tiedot",
    values="value",
    aggfunc="first",
).reset_index()
df_wide.columns.name = None

print(f"  Wide shape: {df_wide.shape}")
print(f"  Columns: {df_wide.columns.tolist()}")

# ---------------------------------------------------------------------------
# 3. Split postal code into numeric code + area name
# ---------------------------------------------------------------------------
print("\nSplitting postal code column ...")

# Format: "00100  Helsinki keskusta - Etu-Töölö (Helsinki)"
df_wide["postal_code"] = df_wide["Postinumero"].str.extract(r"^(\d{5})")
df_wide["area_name"]   = df_wide["Postinumero"].str.extract(r"^\d{5}\s+(.+)$")

# Extract municipality from parentheses at end of area name, e.g. "(Helsinki)"
df_wide["municipality"] = df_wide["area_name"].str.extract(r"\(([^)]+)\)$")
df_wide["area_name"]    = df_wide["area_name"].str.replace(r"\s*\([^)]+\)$", "", regex=True).str.strip()

# ---------------------------------------------------------------------------
# 4. Rename and reorder columns
# ---------------------------------------------------------------------------
df_wide = df_wide.rename(columns={
    "Vuosi":      "year",
    "Talotyyppi": "building_type",
})
df_wide = df_wide.drop(columns=["Postinumero"])

df_wide = df_wide[[
    "year", "postal_code", "area_name", "municipality",
    "building_type", "price_eur_per_sqm", "n_sales"
]]

# Ensure correct dtypes
df_wide["year"]             = df_wide["year"].str.replace("*", "", regex=False).astype(int)
df_wide["price_eur_per_sqm"] = pd.to_numeric(df_wide["price_eur_per_sqm"], errors="coerce")
df_wide["n_sales"]           = pd.to_numeric(df_wide["n_sales"], errors="coerce")

print(f"  Final columns: {df_wide.columns.tolist()}")

# ---------------------------------------------------------------------------
# 5. Data checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DATA CHECKS")
print("=" * 60)

checks = {}

# 5a. Shape
checks["total_rows"]         = len(df_wide)
checks["unique_postal_codes"] = df_wide["postal_code"].nunique()
checks["unique_municipalities"] = df_wide["municipality"].nunique()
checks["unique_building_types"] = df_wide["building_type"].nunique()
checks["year_range"]          = f"{df_wide['year'].min()} - {df_wide['year'].max()}"

print(f"\n[Shape]")
print(f"  Total rows:           {checks['total_rows']:,}")
print(f"  Unique postal codes:  {checks['unique_postal_codes']:,}")
print(f"  Unique municipalities:{checks['unique_municipalities']:,}")
print(f"  Building types:       {df_wide['building_type'].unique().tolist()}")
print(f"  Year range:           {checks['year_range']}")

# 5b. Missing values
print(f"\n[Missing values]")
missing = df_wide.isnull().sum()
missing_pct = (df_wide.isnull().mean() * 100).round(1)
for col in df_wide.columns:
    if missing[col] > 0:
        print(f"  {col}: {missing[col]:,} missing ({missing_pct[col]}%)")
checks["missing_price"]  = int(missing["price_eur_per_sqm"])
checks["missing_nsales"] = int(missing["n_sales"])

# 5c. Duplicates
dupes = df_wide.duplicated(["year", "postal_code", "building_type"]).sum()
print(f"\n[Duplicates]")
print(f"  Duplicate (year, postal_code, building_type): {dupes}")
checks["duplicates"] = int(dupes)

# 5d. Price plausibility
print(f"\n[Price per sqm distribution (EUR/m2)]")
price_stats = df_wide["price_eur_per_sqm"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print(price_stats.to_string())

low_price  = (df_wide["price_eur_per_sqm"] < 200).sum()
high_price = (df_wide["price_eur_per_sqm"] > 20000).sum()
print(f"\n  Suspiciously low  (< 200 EUR/m2):   {low_price}")
print(f"  Suspiciously high (> 20000 EUR/m2): {high_price}")
checks["low_price_outliers"]  = int(low_price)
checks["high_price_outliers"] = int(high_price)

# 5e. Transaction count checks
print(f"\n[Number of sales distribution]")
sales_stats = df_wide["n_sales"].describe(percentiles=[.25, .5, .75, .95, .99])
print(sales_stats.to_string())

# Note: n_sales only reliable from 2020 onwards
pre2020_sales = df_wide[df_wide["year"] < 2020]["n_sales"].notna().sum()
print(f"\n  NOTE: n_sales is from asset-transfer-tax data (reliable from 2020).")
print(f"  Non-null n_sales rows before 2020: {pre2020_sales} (expect 0)")
checks["pre2020_nsales_nonzero"] = int(pre2020_sales)

# 5f. postal_code / area_name parse check
unparsed_codes = df_wide["postal_code"].isna().sum()
unparsed_names = df_wide["area_name"].isna().sum()
print(f"\n[Postal code parsing]")
print(f"  Unparsed postal codes: {unparsed_codes}")
print(f"  Unparsed area names:   {unparsed_names}")
checks["unparsed_postal_codes"] = int(unparsed_codes)

# 5g. Coverage: rows with BOTH price and n_sales (2020+)
from2020 = df_wide[df_wide["year"] >= 2020]
both_present = from2020[["price_eur_per_sqm", "n_sales"]].notna().all(axis=1).sum()
either_missing = len(from2020) - both_present
print(f"\n[2020+ data completeness]")
print(f"  Rows with both price and n_sales: {both_present:,} / {len(from2020):,}")
print(f"  Rows missing one or both:         {either_missing:,}")

# ---------------------------------------------------------------------------
# 6. Drop rows with missing prices
# ---------------------------------------------------------------------------
print("\n[Dropping missing prices]")
n_before = len(df_wide)
df_wide = df_wide.dropna(subset=["price_eur_per_sqm"])
n_after = len(df_wide)
print(f"  Dropped {n_before - n_after:,} rows with missing price_eur_per_sqm")
print(f"  Remaining rows: {n_after:,}")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
out_path = os.path.join(PROC_DIR, "ashi_postal_code_wide.csv")
df_wide.to_csv(out_path, index=False, encoding="utf-8")
print(f"\nSaved processed file -> {out_path}")
print(f"Final shape: {df_wide.shape}")
print("\nSample rows:")
print(df_wide[df_wide["postal_code"] == "00100"].head(8).to_string(index=False))
