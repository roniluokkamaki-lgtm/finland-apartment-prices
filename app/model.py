"""
Regression modelling - Finnish apartment prices (ASHI postal code data).

Steps
-----
1. Load cleaned data
2. Feature engineering: log-price, Helsinki dummy, postal code as string,
   piecewise linear time trend (breakpoint 2022)
3. Train / test split (hold out 2024-2025)
4. Model 1 - Simple WLS:        log_price ~ helsinki + building_type + year + post2022
5. Model 2 - Fixed-effects WLS: log_price ~ postal_code_FE + building_type + year + post2022
            ~1,700 postal code fixed effects replace the 272 municipality FEs.
            Reference postal code: 100 (Helsinki Etu-Tooloo, first alphabetically).
            Piecewise time trend: year captures 2009-2022 growth; post2022 = max(0, year-2022)
            captures the structural break and post-peak decline starting 2023.
6. Evaluate both models on the hold-out set (R2, RMSE, MAE, MAPE)
7. Save model JSON (for Streamlit app), metrics, and key coefficients

Run with: python app/model.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "ashi_postal_code_wide.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
df["year"] = df["year"].astype(int)
print(f"  Shape: {df.shape}")

# ─── Feature engineering ──────────────────────────────────────────────────────
print("\nFeature engineering ...")

# 1. Log-transform the dependent variable
df["log_price"] = np.log(df["price_eur_per_sqm"])
print(f"  log_price - mean: {df['log_price'].mean():.3f}, std: {df['log_price'].std():.3f}")

# 2. Helsinki dummy (used in Model 1 only)
df["helsinki"] = (df["municipality"] == "Helsinki").astype(int)
print(f"  Helsinki observations: {df['helsinki'].sum():,} / {len(df):,}")

# 3. Piecewise linear time trend — breakpoint at 2022
#    post2022 = max(0, year - 2022)
#    Interpretation:
#      year coefficient      = annual % change in prices 2009-2022 (growth phase)
#      year + post2022 coef  = annual % change in prices 2023+    (decline phase)
#    Data shows peak in 2022 followed by sharp decline in 2023 (-8.5% Helsinki,
#    -6.2% national weighted), continuing into 2024.
BREAKPOINT = 2022
df["post2022"] = (df["year"] - BREAKPOINT).clip(lower=0)
print(f"  post2022 range: {df['post2022'].min()} - {df['post2022'].max()}")

# 4. Postal code as string so patsy treats it as categorical
#    Reference: postal code "100" (Helsinki Etu-Tooloo) - first alphabetically
df["postal_code_str"] = df["postal_code"].astype(str)
print(f"  Unique postal codes: {df['postal_code_str'].nunique():,}")

# Build postal code -> area name lookup (used in JSON export for the UI)
postal_to_area = (
    df[["postal_code_str", "area_name", "municipality"]]
    .drop_duplicates("postal_code_str")
    .set_index("postal_code_str")[["area_name", "municipality"]]
    .to_dict("index")
)

# ─── Train / test split ───────────────────────────────────────────────────────
TEST_YEARS = [2024, 2025]
train = df[~df["year"].isin(TEST_YEARS)].copy()
test  = df[ df["year"].isin(TEST_YEARS)].copy()
print(f"\nTrain: {len(train):,} rows ({train['year'].min()}-{train['year'].max()})")
print(f"Test : {len(test):,}  rows ({', '.join(str(y) for y in TEST_YEARS)})")

# ─── Model 1 - Simple WLS ─────────────────────────────────────────────────────
# log(price) = alpha + beta1*helsinki + beta2*building_type + beta3*year + eps
# Weights: n_sales. Reference building type: first alphabetically.
print("\n" + "="*70)
print("Model 1 - Simple WLS (Helsinki dummy)")
print("="*70)

formula_1 = "log_price ~ helsinki + C(building_type) + year + post2022"
m1 = smf.wls(formula_1, data=train, weights=train["n_sales"]).fit()
print(m1.summary())

hki_beta = m1.params["helsinki"]
hki_pct  = (np.exp(hki_beta) - 1) * 100
print(f"\nHelsinki premium: beta={hki_beta:.4f} -> exp(beta)={np.exp(hki_beta):.3f} -> +{hki_pct:.1f}% vs rest of Finland")

# ─── Model 2 - Postal code fixed-effects WLS ──────────────────────────────────
# log(price) = alpha + C(postal_code) + beta1*building_type + beta2*year + eps
#
# ~1,700 postal code FEs replace the 272 municipality FEs.
# Reference postal code: "100" (Helsinki Etu-Tooloo, first alphabetically).
# All other postal code coefficients are discounts/premiums vs postal code 100.
# Weights: n_sales.
print("\n" + "="*70)
print("Model 2 - Fixed-effects WLS (postal code FEs)")
print("="*70)

formula_2 = "log_price ~ C(postal_code_str) + C(building_type) + year + post2022"
m2 = smf.wls(formula_2, data=train, weights=train["n_sales"]).fit()

summary_path = os.path.join(OUTPUTS_DIR, "model2_fe_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(str(m2.summary()))
print(f"Full summary saved -> {summary_path}")

print(f"\nModel 2 - Overall fit")
print(f"  R2          : {m2.rsquared:.4f}")
print(f"  Adj. R2     : {m2.rsquared_adj:.4f}")
print(f"  Observations: {int(m2.nobs):,}")

# Non-postal-code coefficients (building type + year)
key_params  = m2.params[[p for p in m2.params.index if "postal_code" not in p]]
key_pvalues = m2.pvalues[key_params.index]
key_df = pd.DataFrame({
    "coefficient": key_params.round(4),
    "pct_effect" : ((np.exp(key_params) - 1) * 100).round(1),
    "p_value"    : key_pvalues.round(4),
})
print("\nKey coefficients (excluding postal code FEs):")
print(key_df.to_string())

# Top 10 most expensive and cheapest postal codes vs reference (100)
pc_params = m2.params[[p for p in m2.params.index if "postal_code" in p]]
pc_df = pd.DataFrame({
    "postal_code"        : pc_params.index.str.extract(r"\[T\.(\w+)\]")[0].values,
    "log_diff_vs_ref"    : pc_params.values,
    "pct_diff_vs_ref"    : (np.exp(pc_params.values) - 1) * 100,
})
pc_df = pc_df.dropna(subset=["postal_code"])
pc_df["area_name"] = pc_df["postal_code"].map(
    lambda c: postal_to_area.get(c, {}).get("area_name", "")
)
pc_df = pc_df.sort_values("pct_diff_vs_ref", ascending=False)

print("\nTop 10 most expensive postal codes vs reference (100, Helsinki Etu-Tooloo):")
print(pc_df.head(10)[["postal_code", "area_name", "pct_diff_vs_ref"]].round(1).to_string(index=False))

print("\nTop 10 cheapest postal codes vs reference (100, Helsinki Etu-Tooloo):")
print(pc_df.tail(10)[["postal_code", "area_name", "pct_diff_vs_ref"]].round(1).to_string(index=False))

# ─── Evaluation ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Evaluation on hold-out set (2024-2025)")
print("="*70)

def evaluate(model, data, name):
    pred_log   = model.predict(data)
    pred_eur   = np.exp(pred_log)
    actual_eur = data["price_eur_per_sqm"]
    rmse = np.sqrt(((actual_eur - pred_eur) ** 2).mean())
    mae  = (actual_eur - pred_eur).abs().mean()
    mape = ((actual_eur - pred_eur).abs() / actual_eur).mean() * 100
    print(f"\n{name}")
    print(f"  Train R2           : {model.rsquared:.4f}")
    print(f"  Test  RMSE (EUR/m2): {rmse:,.0f}")
    print(f"  Test  MAE  (EUR/m2): {mae:,.0f}")
    print(f"  Test  MAPE         : {mape:.1f}%")
    return {
        "model"    : name,
        "train_r2" : round(model.rsquared, 4),
        "test_rmse": round(rmse, 0),
        "test_mae" : round(mae, 0),
        "test_mape": round(mape, 2),
    }

# Filter test set to postal codes seen in training
known_postal = set(train["postal_code_str"].unique())
test_known   = test[test["postal_code_str"].isin(known_postal)].copy()
dropped      = len(test) - len(test_known)
if dropped:
    print(f"  Note: {dropped} test rows dropped (postal codes unseen in training)")

metrics = []
metrics.append(evaluate(m1, test,       "Model 1 - Simple"))
metrics.append(evaluate(m2, test_known, "Model 2 - Postal Code FEs"))

metrics_df   = pd.DataFrame(metrics)
metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics saved -> {metrics_path}")

# ─── Save postal code coefficients CSV ───────────────────────────────────────
pc_coef_path = os.path.join(OUTPUTS_DIR, "model2_postal_code_coefs.csv")
pc_df.to_csv(pc_coef_path, index=False)
print(f"Postal code coefficients saved -> {pc_coef_path}")

# ─── Serialise model objects ──────────────────────────────────────────────────
for name, model in [("model1_simple", m1), ("model2_fe", m2)]:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model pickle saved -> {path}")

# ─── Export Model 2 coefficients as JSON for the Streamlit app ───────────────
# Prediction:
#   log_price = intercept
#             + postal_code_coef  (0 for reference postal code "100")
#             + building_type_coef (0 for reference building type)
#             + year_coef * year
#   price = exp(log_price)

# Postal code coefficients (reference "100" = 0.0)
reference_postal = min(df["postal_code_str"].unique(), key=lambda x: x)
pc_coefs = {}
for param, val in m2.params.items():
    if "postal_code" in param and "[T." in param:
        code = param.split("[T.")[1].rstrip("]")
        pc_coefs[code] = float(val)
pc_coefs[reference_postal] = 0.0   # reference

# Building type coefficients
reference_btype = sorted(df["building_type"].unique())[0]
btype_coefs = {}
for param, val in m2.params.items():
    if "building_type" in param and "[T." in param:
        btype = param.split("[T.")[1].rstrip("]")
        btype_coefs[btype] = float(val)
btype_coefs[reference_btype] = 0.0

model_json = {
    "intercept"          : float(m2.params["Intercept"]),
    "year_coef"          : float(m2.params["year"]),
    "post2022_coef"      : float(m2.params["post2022"]),
    "breakpoint_year"    : BREAKPOINT,
    "postal_code_coefs"  : pc_coefs,
    "building_type_coefs": btype_coefs,
    "postal_to_area"     : postal_to_area,
    "reference_postal"   : reference_postal,
    "train_years"        : [int(train["year"].min()), int(train["year"].max())],
    "r2"                 : round(float(m2.rsquared), 4),
    "test_rmse"          : int(metrics[1]["test_rmse"]),
}

json_path = os.path.join(MODELS_DIR, "model2_fe.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(model_json, f, indent=2, ensure_ascii=False)
print(f"Model JSON saved -> {json_path}")

# Spot-check: postal code 100 (reference), 2-room flat, 2026
ref_btype   = "Blocks of flats, two-room flat"
spot_year   = 2026
spot_post22 = max(0, spot_year - BREAKPOINT)
log_pred = (
    model_json["intercept"]
    + pc_coefs[reference_postal]
    + btype_coefs[ref_btype]
    + model_json["year_coef"] * spot_year
    + model_json["post2022_coef"] * spot_post22
)
print(f"Spot-check (postal 100 / Helsinki Etu-Tooloo, 2-room, 2026): {round(np.exp(log_pred))} EUR/m2")

# Print piecewise trend interpretation
pre_annual  = (np.exp(m2.params["year"]) - 1) * 100
post_annual = (np.exp(m2.params["year"] + m2.params["post2022"]) - 1) * 100
print(f"\nTime trend:")
print(f"  Annual change 2009-2022: {pre_annual:+.1f}%/year")
print(f"  Annual change 2023+    : {post_annual:+.1f}%/year")

print("\nDone.")
