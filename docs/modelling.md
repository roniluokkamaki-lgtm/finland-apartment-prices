# Modelling Documentation

## Overview

Two OLS regression models are estimated to explain apartment prices in Finland using
postal-code-level panel data from Statistics Finland (2009–2025).  
The modelling script is `app/model.py`.

---

## Dependent Variable

### Log transformation of price per square metre

The dependent variable is `log(price_eur_per_sqm)` rather than the raw price.

**Motivation (from EDA):**
- `price_eur_per_sqm` is right-skewed — most observations cluster below 3,000 €/m² but a
  long tail extends to ~11,000 €/m² in prime Helsinki.
- The log transformation compresses the tail, improves residual normality, and reduces
  heteroskedasticity.
- Coefficients become directly interpretable as **percentage effects** (see below).

**Back-transformation for prediction:**
```
predicted_price = exp(predicted_log_price)
```

---

## Feature Engineering

| Feature | Construction | Notes |
|---|---|---|
| `log_price` | `log(price_eur_per_sqm)` | Dependent variable |
| `helsinki` | `1 if municipality == "Helsinki" else 0` | Binary dummy (Model 1 only) |
| `building_type` | Categorical, 4 levels | Reference: "Blocks of flats, one-room flat" |
| `year` | Integer (2009–2025) | Linear time trend |
| `municipality` | Categorical, 272 levels | Used as fixed effects in Model 2; reference = Helsinki |

**Weights:** All models are estimated as Weighted Least Squares (WLS) using `n_sales`
as the observation weight. This down-weights cells that are based on few transactions
and therefore have noisier price estimates.

---

## Train / Test Split

| Set | Years | Rows |
|---|---|---|
| Train | 2009–2023 | ~27,000 |
| Test | 2024–2025 | ~6,000 |

A temporal holdout is used (future years as test) to evaluate out-of-sample predictive
performance in a realistic setting. The 2025 data is preliminary.

---

## Model Specifications

### Model 1 — Simple WLS (Helsinki dummy)

```
log(price_eur_per_sqm) = α
                        + β₁ · helsinki
                        + β₂ · building_type
                        + β₃ · year
                        + ε
```

**Purpose:** Provides a clean estimate of the **Helsinki price premium** relative to
the rest of Finland, along with building-type and time-trend effects.

**Coefficient interpretation:**
- `β₁` (helsinki): `(exp(β₁) − 1) × 100` = % price premium of Helsinki vs rest of Finland
- `β₂` (building_type): % price difference between building types
- `β₃` (year): % annual price change across all of Finland

**Limitation:** Does not control for variation within non-Helsinki municipalities
(e.g., Tampere vs Oulu), so location effects beyond the Helsinki/non-Helsinki split
are absorbed into the residuals.

---

### Model 2 — Fixed-Effects WLS (municipality FEs)

```
log(price_eur_per_sqm) = α
                        + γₘ · municipality  (fixed effects, 272 levels)
                        + β₁ · building_type
                        + β₂ · year
                        + ε
```

**Helsinki as the reference category:** Helsinki is set as the `Treatment` reference
level for `C(municipality)`. This means:
- The intercept represents the baseline Helsinki price (for the reference building type
  and a given year).
- Every other municipality coefficient `γₘ` is the log price difference *relative to
  Helsinki* — a negative value means the municipality is cheaper than Helsinki.
- The Helsinki premium is therefore implicitly encoded: all other municipalities show
  their discount from the Helsinki baseline.

**Why municipality FEs and not just a Helsinki dummy?**  
Municipality fixed effects control for all time-invariant location characteristics
(geography, local amenities, commuting zones) across all 272 municipalities, not just
the Helsinki/non-Helsinki split. This substantially improves model fit and enables
municipality-level price predictions for the Streamlit UI.

**Coefficient interpretation:**
- `γₘ` (municipality FE): `(exp(γₘ) − 1) × 100` = % price premium/discount vs Helsinki
- `β₁` (building_type): % price difference between building types, controlling for location
- `β₂` (year): % annual price change, controlling for location

---

## Outputs

| File | Description |
|---|---|
| `outputs/model_metrics.csv` | R², RMSE, MAE, MAPE for both models on the test set |
| `outputs/model2_fe_summary.txt` | Full statsmodels summary for Model 2 (272 municipality coefficients) |
| `outputs/model2_municipality_coefs.csv` | Municipality fixed-effect coefficients and % premium/discount vs Helsinki |
| `models/model1_simple.pkl` | Serialised Model 1 object |
| `models/model2_fe.pkl` | Serialised Model 2 object (used by Streamlit app) |

---

## Notes

### Prediction for new observations
The Streamlit app uses **Model 2** for price predictions. The user inputs a
municipality, building type, and year; the model returns `exp(predicted_log_price)`.

Municipalities not seen during training cannot receive a fixed-effect estimate from
Model 2. In practice this is unlikely as all 272 municipalities are present across
multiple years in the training data.

### Helsinki premium summary (Model 1)
The `β_helsinki` coefficient from Model 1 directly answers the question:
*"How much more expensive are Helsinki apartments compared to the rest of Finland,
holding building type and year constant?"*

The answer is `(exp(β_helsinki) − 1) × 100` percent.

### Year as a linear trend
Both models include `year` as a continuous variable, which assumes a constant annual
percentage change in prices. The EDA showed prices rose from 2009 to ~2022 then
declined, suggesting a linear trend is a simplification. If predictive performance
is poor near the recent peak, a polynomial year term or a post-2022 dummy could be
considered in a future iteration.

### n_sales before 2020
The `n_sales` methodology changed in 2020 (see `docs/data_cleaning.md`). All values
are used as WLS weights without adjustment since the counts appear consistent across
the methodology break and the weighting role is not sensitive to small discrepancies.
