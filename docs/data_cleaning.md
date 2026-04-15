# Data Cleaning Documentation

## Source data

**Dataset:** Prices per square meter of old dwellings in housing companies by postal code area, yearly  
**Provider:** Statistics Finland (Tilastokeskus)  
**Table ID:** `statfin_ashi_pxt_13mu.px`  
**Access method:** PxWeb API (CC BY 4.0 licence)  
**Raw file:** `data/raw/ashi_postal_code.csv`  
**Processed file:** `data/processed/ashi_postal_code_wide.csv`  
**Processing script:** `app/process_data.py`

---

## Raw data structure

The raw data is delivered in **long format**: each row represents the value of one metric for one combination of year, postal code, and building type. Each (year, postal code, building type) cell therefore appears as exactly two rows — one for price and one for transaction count.

**Dimensions in the raw data:**

| Column | Description | # Unique values |
|---|---|---|
| `Vuosi` | Year | 17 (2009–2025) |
| `Postinumero` | Postal code + area name combined | 1,724 |
| `Talotyyppi` | Building type / room count | 4 |
| `Tiedot` | Metric name (price or count) | 2 |

**Building types published:**
- Blocks of flats, one-room flat
- Blocks of flats, two-room flat
- Blocks of flats, three-room flat+
- Terraced houses total

---

## Cleaning steps

### Step 1 — Pivot to wide format

The long-format data was pivoted so that price and transaction count become separate columns. The `Tiedot` column values were renamed for clarity:

| Original label | New column name |
|---|---|
| Price per square meter (EUR/m2) | `price_eur_per_sqm` |
| Number of sales, asset transfer tax data starting from 2020 | `n_sales` |

**Result:** 60,020 rows × 5 columns (before postal code splitting).

### Step 2 — Split postal code column

The raw `Postinumero` column combined the 5-digit postal code and the area name in a single string (e.g. `"00100  Helsinki keskusta - Etu-Töölö (Helsinki)"`). This was split into three separate columns:

- `postal_code` — 5-digit code (e.g. `00100`)
- `area_name` — neighbourhood/area name (e.g. `Helsinki keskusta - Etu-Töölö`)
- `municipality` — extracted from parentheses at the end (e.g. `Helsinki`)

All 1,724 postal codes and area names parsed without errors.

### Step 3 — Drop rows with missing prices

**43.5% of price values (26,110 rows) were missing** in the raw data. This is expected behaviour: Statistics Finland suppresses the price when a cell has too few transactions to publish reliably or to protect confidentiality. This is most common for:
- Terraced houses in small postal code areas
- Rural postal codes with low transaction volumes
- Earlier years (2009–2012) where coverage was lower

**Decision:** Rows with a missing `price_eur_per_sqm` were dropped. The remaining dataset covers only postal code × building type × year combinations with at least one published price observation.

**Post-drop shape:** 33,910 rows.

---

## Data checks (on wide format, before dropping missing prices)

| Check | Result | Status |
|---|---|---|
| Duplicate rows (year, postal_code, building_type) | 0 | OK |
| Price below 200 EUR/m2 | 0 | OK |
| Price above 20,000 EUR/m2 | 0 | OK |
| Unparsed postal codes | 0 | OK |
| Unparsed area names | 0 | OK |
| Missing prices | 26,110 (43.5%) | Dropped |
| Missing n_sales (2020+) | 9,973 | See note below |

**Price distribution (after cleaning):**

| Statistic | EUR/m2 |
|---|---|
| Min | 296 |
| 5th percentile | 772 |
| Median | 1,780 |
| Mean | 2,103 |
| 95th percentile | 4,614 |
| Max | 10,691 |

---

## Notes

### Missing prices (43.5%)

Missing prices are a structural feature of the data, not a data quality issue. Statistics Finland withholds prices when the number of transactions in a cell is too low. Dropping these rows means the model is trained on areas and building types with sufficient market activity. Predictions for very low-activity areas (e.g. terraced houses in small rural postal codes) should be interpreted with caution as these are extrapolated beyond the training data.

### Transaction count methodology change (2020)

The `n_sales` column is labelled "asset transfer tax data starting from 2020", reflecting a change in the underlying data source:

- **Before 2020:** transaction counts were derived from the deed registry
- **From 2020 onwards:** counts come from asset transfer tax declarations

Both sources provide transaction counts and the values appear consistent across the break. However, minor discrepancies may exist around the 2020 boundary due to differences in registration timing and scope between the two systems. For the regression model, `n_sales` is used as a weight variable (to down-weight cells with few transactions) rather than as an explanatory variable, so a small methodology break is not a material concern.

### 2025 data

The 2025 data is marked as preliminary (`*`) in the source. It is retained in the dataset but should be treated with caution in any time-series analysis.
