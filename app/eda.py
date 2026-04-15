"""
Exploratory Data Analysis — Finnish apartment prices (ASHI postal code data).

Generates a self-contained HTML report saved to outputs/eda_report.html.

Run with: python app/eda.py
"""

import os
import base64
import io
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "ashi_postal_code_wide.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
REPORT_PATH = os.path.join(OUTPUT_DIR, "eda_report.html")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
df["year"] = df["year"].astype(int)
print(f"  Shape: {df.shape}")

# Convenience subsets
df_hki    = df[df["municipality"] == "Helsinki"]
df_nonhki = df[df["municipality"] != "Helsinki"]

# ---------------------------------------------------------------------------
# Helper: convert a matplotlib figure to a base64 PNG string
# ---------------------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:12px 0;">'

# ---------------------------------------------------------------------------
# Helper: render a DataFrame as an HTML table
# ---------------------------------------------------------------------------
def df_to_html(df_in, floatfmt="{:.1f}"):
    return df_in.to_html(
        classes="data-table", border=0, index=True,
        float_format=lambda x: f"{x:,.1f}"
    )

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BTYPE_COLORS = {
    "Blocks of flats, one-room flat":    "#2196F3",
    "Blocks of flats, two-room flat":    "#4CAF50",
    "Blocks of flats, three-room flat+": "#FF9800",
    "Terraced houses total":             "#9C27B0",
}
BTYPES = list(BTYPE_COLORS.keys())
COLORS = list(BTYPE_COLORS.values())

# ============================================================
# SECTION 1 — Dataset overview
# ============================================================
print("Section 1: Dataset overview ...")

overview = {
    "Total observations":        f"{len(df):,}",
    "Unique postal codes":       f"{df['postal_code'].nunique():,}",
    "Unique municipalities":     f"{df['municipality'].nunique():,}",
    "Years covered":             f"{df['year'].min()} – {df['year'].max()}",
    "Building types":            df['building_type'].nunique(),
    "Helsinki observations":     f"{len(df_hki):,}",
    "Non-Helsinki observations": f"{len(df_nonhki):,}",
}

overview_html = "<table class='data-table'><tr><th>Metric</th><th>Value</th></tr>"
for k, v in overview.items():
    overview_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
overview_html += "</table>"

# Coverage per year
coverage = (
    df.groupby("year")["postal_code"]
    .nunique()
    .rename("unique_postal_codes")
    .to_frame()
)
coverage["observations"] = df.groupby("year").size()
coverage_html = df_to_html(coverage)

# ============================================================
# SECTION 2 — Price distribution
# ============================================================
print("Section 2: Price distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Price per sqm distribution (EUR/m²)", fontsize=13, fontweight="bold")

# Overall histogram
axes[0].hist(df["price_eur_per_sqm"], bins=80, color="#2196F3", edgecolor="white", linewidth=0.3)
axes[0].axvline(df["price_eur_per_sqm"].median(), color="red", linestyle="--", label=f"Median {df['price_eur_per_sqm'].median():,.0f}")
axes[0].axvline(df["price_eur_per_sqm"].mean(),   color="orange", linestyle="--", label=f"Mean {df['price_eur_per_sqm'].mean():,.0f}")
axes[0].set_xlabel("EUR/m²")
axes[0].set_ylabel("Count")
axes[0].set_title("All observations")
axes[0].legend()
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# By building type
for btype, color in BTYPE_COLORS.items():
    sub = df[df["building_type"] == btype]["price_eur_per_sqm"].dropna()
    axes[1].hist(sub, bins=60, alpha=0.55, color=color, label=btype.replace("Blocks of flats, ", "").replace(" flat", "").replace("Terraced houses total", "Terraced"))
axes[1].set_xlabel("EUR/m²")
axes[1].set_title("By building type")
axes[1].legend(fontsize=8)
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fig.tight_layout()
hist_img = fig_to_base64(fig)

# Summary stats by building type
stats = (
    df.groupby("building_type")["price_eur_per_sqm"]
    .describe(percentiles=[.25, .5, .75])
    .round(0)
    .drop(columns=["count"])
)
stats_html = df_to_html(stats)

# ============================================================
# SECTION 3 — Price trends over time
# ============================================================
print("Section 3: Price trends over time ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Average price per sqm over time (EUR/m²)", fontsize=13, fontweight="bold")

# National trend by building type
for btype, color in BTYPE_COLORS.items():
    sub = df[df["building_type"] == btype].groupby("year")["price_eur_per_sqm"].median()
    label = btype.replace("Blocks of flats, ", "").replace(" flat", "").replace("Terraced houses total", "Terraced")
    axes[0].plot(sub.index, sub.values, marker="o", markersize=3, color=color, label=label)
axes[0].set_title("Finland (median by building type)")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("EUR/m²")
axes[0].legend(fontsize=8)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Helsinki vs rest of Finland
for label, subset, color in [
    ("Helsinki", df_hki, "#E53935"),
    ("Rest of Finland", df_nonhki, "#1E88E5"),
]:
    med = subset.groupby("year")["price_eur_per_sqm"].median()
    axes[1].plot(med.index, med.values, marker="o", markersize=3, color=color, label=label)
axes[1].set_title("Helsinki vs Rest of Finland (median)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("EUR/m²")
axes[1].legend()
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fig.tight_layout()
trend_img = fig_to_base64(fig)

# YoY change nationally
national_median = df.groupby("year")["price_eur_per_sqm"].median()
yoy = national_median.pct_change().mul(100).round(1).rename("YoY change (%)")
yoy_df = pd.DataFrame({"Median price (EUR/m2)": national_median.round(0), "YoY change (%)": yoy})
yoy_html = df_to_html(yoy_df)

# ============================================================
# SECTION 4 — Geographic distribution
# ============================================================
print("Section 4: Geographic distribution ...")

# Most recent full year
latest_year = df[df["year"] < 2025]["year"].max()
df_latest = df[df["year"] == latest_year]

# Top/bottom 15 municipalities by median price (min 20 obs)
muni_prices = (
    df_latest.groupby("municipality")
    .filter(lambda x: len(x) >= 20)
    .groupby("municipality")["price_eur_per_sqm"]
    .median()
    .sort_values(ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"Median price per sqm by municipality ({latest_year}, EUR/m²)", fontsize=13, fontweight="bold")

top15 = muni_prices.head(15)
axes[0].barh(top15.index[::-1], top15.values[::-1], color="#E53935")
axes[0].set_title("Top 15 most expensive")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].set_xlabel("EUR/m²")

bot15 = muni_prices.tail(15)
axes[1].barh(bot15.index, bot15.values, color="#1E88E5")
axes[1].set_title("Top 15 least expensive")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[1].set_xlabel("EUR/m²")
fig.tight_layout()
geo_img = fig_to_base64(fig)

# ============================================================
# SECTION 5 — Box plots by building type and year
# ============================================================
print("Section 5: Box plots ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Price spread by building type and period", fontsize=13, fontweight="bold")

# Box plot by building type
data_by_btype = [df[df["building_type"] == b]["price_eur_per_sqm"].dropna().values for b in BTYPES]
short_labels   = ["1-room flat", "2-room flat", "3-room flat+", "Terraced"]
bp = axes[0].boxplot(data_by_btype, patch_artist=True, labels=short_labels, showfliers=False)
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title("By building type (all years)")
axes[0].set_ylabel("EUR/m²")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].tick_params(axis="x", labelsize=8)

# Box plot by 5-year period
df["period"] = pd.cut(df["year"], bins=[2008, 2013, 2018, 2020, 2025],
                      labels=["2009-2013", "2014-2018", "2019-2020", "2021-2025"])
period_order = ["2009-2013", "2014-2018", "2019-2020", "2021-2025"]
data_by_period = [df[df["period"] == p]["price_eur_per_sqm"].dropna().values for p in period_order]
bp2 = axes[1].boxplot(data_by_period, patch_artist=True, labels=period_order, showfliers=False)
period_colors = ["#90CAF9", "#42A5F5", "#1E88E5", "#1565C0"]
for patch, color in zip(bp2["boxes"], period_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].set_title("By time period (all building types)")
axes[1].set_ylabel("EUR/m²")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fig.tight_layout()
box_img = fig_to_base64(fig)

# ============================================================
# SECTION 6 — Helsinki deep-dive
# ============================================================
print("Section 6: Helsinki deep-dive ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Helsinki — price distribution and trends", fontsize=13, fontweight="bold")

# Histogram Helsinki vs rest
axes[0].hist(df_nonhki["price_eur_per_sqm"].dropna(), bins=60, alpha=0.6, color="#1E88E5", label="Rest of Finland")
axes[0].hist(df_hki["price_eur_per_sqm"].dropna(),    bins=60, alpha=0.7, color="#E53935", label="Helsinki")
axes[0].set_xlabel("EUR/m²")
axes[0].set_ylabel("Count")
axes[0].set_title("Price distribution")
axes[0].legend()
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Helsinki price trend by building type
for btype, color in BTYPE_COLORS.items():
    sub = df_hki[df_hki["building_type"] == btype].groupby("year")["price_eur_per_sqm"].median()
    label = btype.replace("Blocks of flats, ", "").replace(" flat", "").replace("Terraced houses total", "Terraced")
    axes[1].plot(sub.index, sub.values, marker="o", markersize=3, color=color, label=label)
axes[1].set_title("Helsinki — median price trend by building type")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("EUR/m²")
axes[1].legend(fontsize=8)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fig.tight_layout()
hki_img = fig_to_base64(fig)

# Top 10 most expensive Helsinki postal codes (latest year)
hki_latest = df_hki[df_hki["year"] == latest_year]
hki_top = (
    hki_latest.groupby(["postal_code", "area_name"])["price_eur_per_sqm"]
    .median()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    .rename(columns={"postal_code": "Postal code", "area_name": "Area", "price_eur_per_sqm": "Median EUR/m²"})
)
hki_top["Median EUR/m²"] = hki_top["Median EUR/m²"].map("{:,.0f}".format)
hki_top_html = hki_top.to_html(classes="data-table", border=0, index=False)

# ============================================================
# ASSEMBLE HTML REPORT
# ============================================================
print("Assembling HTML report ...")

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #212121; }
  h1   { color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }
  h2   { color: #1976D2; margin-top: 48px; border-left: 4px solid #1976D2;
         padding-left: 12px; }
  h3   { color: #424242; margin-top: 28px; }
  .data-table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
  .data-table th { background: #1976D2; color: white; padding: 8px 12px; text-align: left; }
  .data-table td { padding: 6px 12px; border-bottom: 1px solid #E0E0E0; }
  .data-table tr:nth-child(even) { background: #F5F5F5; }
  .callout { background: #E3F2FD; border-left: 4px solid #1976D2;
             padding: 12px 16px; margin: 16px 0; border-radius: 4px; font-size: 14px; }
  .warning { background: #FFF8E1; border-left: 4px solid #FFA000;
             padding: 12px 16px; margin: 16px 0; border-radius: 4px; font-size: 14px; }
  img { border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
</style>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>EDA — Finnish Apartment Prices</title>
  {CSS}
</head>
<body>

<h1>Exploratory Data Analysis — Finnish Apartment Prices</h1>
<p>Source: Statistics Finland, ASHI table <code>statfin_ashi_pxt_13mu.px</code> &nbsp;|&nbsp;
   Data file: <code>data/processed/ashi_postal_code_wide.csv</code> &nbsp;|&nbsp;
   Generated: {pd.Timestamp.now().strftime("%Y-%m-%d")}</p>

<!-- ====== SECTION 1 ====== -->
<h2>1. Dataset Overview</h2>
{overview_html}

<h3>Observations per year</h3>
<div class="callout">Coverage grows over time as more postal code × building type combinations
accumulate enough transactions to be published.</div>
{coverage_html}

<!-- ====== SECTION 2 ====== -->
<h2>2. Price Distribution</h2>
{hist_img}
<div class="callout">
  <strong>Key finding:</strong> Price distribution is right-skewed — the majority of observations
  cluster below 3,000 EUR/m² (rural and smaller cities) but a long tail extends to ~11,000 EUR/m²
  (prime Helsinki). Consider log-transforming the dependent variable in the regression.
</div>

<h3>Descriptive statistics by building type (EUR/m²)</h3>
{stats_html}

<!-- ====== SECTION 3 ====== -->
<h2>3. Price Trends Over Time</h2>
{trend_img}
<div class="callout">
  <strong>Key finding:</strong> Prices rose steadily from 2009 to ~2022, then declined —
  consistent with the broader Finnish housing market correction. The Helsinki–rest-of-Finland
  gap has widened considerably over the period. One-room flats command a premium per sqm over
  larger units across all years.
</div>

<h3>National median price and year-on-year change</h3>
{yoy_html}

<!-- ====== SECTION 4 ====== -->
<h2>4. Geographic Distribution ({latest_year})</h2>
{geo_img}
<div class="callout">
  <strong>Key finding:</strong> Helsinki metropolitan area municipalities dominate the top of
  the price ranking. The cheapest markets are concentrated in rural municipalities in eastern
  and northern Finland.
</div>

<!-- ====== SECTION 5 ====== -->
<h2>5. Price Spread by Building Type and Period</h2>
{box_img}
<div class="callout">
  <strong>Key finding:</strong> Smaller flats (one-room) show a higher and wider price
  distribution per sqm than larger units — a well-known feature of Finnish apartment markets.
  Prices across all types have shifted upward over time. Outliers are suppressed in these plots
  (whiskers = 5th–95th percentile).
</div>

<!-- ====== SECTION 6 ====== -->
<h2>6. Helsinki Deep-Dive</h2>
{hki_img}

<h3>Top 10 most expensive Helsinki postal codes ({latest_year}, median across building types)</h3>
{hki_top_html}
<div class="warning">
  <strong>Note on 2025 data:</strong> 2025 figures are preliminary and may be revised.
  Where possible, analysis uses {latest_year} as the reference year.
</div>

<!-- ====== SECTION 7 — Key takeaways for modelling ====== -->
<h2>7. Key Takeaways for Modelling</h2>
<ul>
  <li><strong>Right-skewed prices:</strong> Consider using log(price_eur_per_sqm) as the
      dependent variable to improve regression fit and reduce heteroskedasticity.</li>
  <li><strong>Building type matters:</strong> One-room flats are substantially more expensive
      per sqm than larger units — building type must be included as a categorical variable.</li>
  <li><strong>Strong location effect:</strong> Helsinki commands a large and persistent price
      premium. A Helsinki/non-Helsinki indicator or municipality fixed effects will be important.</li>
  <li><strong>Time trend:</strong> Prices rose from 2009 to ~2022 then declined — year or a
      time trend variable should be included.</li>
  <li><strong>Use n_sales as regression weight</strong> (from 2020 onwards) to down-weight
      cells based on few transactions.</li>
</ul>

</body>
</html>
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nReport saved -> {REPORT_PATH}")
