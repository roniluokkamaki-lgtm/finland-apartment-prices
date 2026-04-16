"""
Regression modelling - Finnish apartment prices (ASHI postal code data).

Steps
-----
1. Load cleaned data
2. Feature engineering: log-price, Helsinki dummy
3. Train / test split (hold out 2024-2025)
4. Model 1 - Simple WLS: log_price ~ helsinki + building_type + year
5. Model 2 - Fixed-effects WLS: log_price ~ municipality_FE + building_type + year
            Helsinki is the reference category, so every municipality coefficient
            represents a discount (or premium) relative to Helsinki.
6. Evaluate both models on the hold-out set (R2, RMSE, MAE in original EUR/m2 scale)
7. Save model objects, metrics, and key coefficients

Run with: python app/model.py
"""

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
#    Motivation: price_eur_per_sqm is right-skewed (EDA Section 2).
#    Log-transform improves residual normality, reduces heteroskedasticity,
#    and makes coefficients interpretable as percentage effects.
df["log_price"] = np.log(df["price_eur_per_sqm"])
print(f"  log_price  -  mean: {df['log_price'].mean():.3f}, "
      f"std: {df['log_price'].std():.3f}")

# 2. Helsinki dummy
#    Captures the Helsinki price premium over the rest of Finland (Model 1).
#    In Model 2 Helsinki serves as the reference category in the municipality FEs.
df["helsinki"] = (df["municipality"] == "Helsinki").astype(int)
print(f"  Helsinki observations: {df['helsinki'].sum():,} / {len(df):,}")

# ─── Train / test split ───────────────────────────────────────────────────────
# Hold out 2024-2025 as the test set (2025 is preliminary, 2024 is the most
# recent full year). Train on 2009-2023.
TEST_YEARS  = [2024, 2025]
train = df[~df["year"].isin(TEST_YEARS)].copy()
test  = df[ df["year"].isin(TEST_YEARS)].copy()
print(f"\nTrain: {len(train):,} rows ({train['year'].min()}-{train['year'].max()})")
print(f"Test : {len(test):,}  rows ({', '.join(str(y) for y in TEST_YEARS)})")

# ─── Model 1 - Simple WLS ─────────────────────────────────────────────────────
# Specification:
#   log(price) = alpha + beta₁*helsinki + beta₂*building_type + beta₃*year + eps
#
# Weights: n_sales (down-weights cells based on few transactions).
# Reference categories: statsmodels drops the first alphabetical level for each
# C() factor - "Blocks of flats, one-room flat" for building type.
print("\n" + "="*70)
print("Model 1 - Simple WLS")
print("="*70)

formula_1 = "log_price ~ helsinki + C(building_type) + year"
m1 = smf.wls(formula_1, data=train, weights=train["n_sales"]).fit()
print(m1.summary())

# Helsinki premium interpretation
hki_beta = m1.params["helsinki"]
hki_pct  = (np.exp(hki_beta) - 1) * 100
print(f"\nHelsinki premium  beta = {hki_beta:.4f}  ->  exp(beta) = {np.exp(hki_beta):.3f}"
      f"  ->  +{hki_pct:.1f}% vs rest of Finland")

# ─── Model 2 - Fixed-effects WLS ──────────────────────────────────────────────
# Specification:
#   log(price) = alpha + C(municipality) + beta₁*building_type + beta₂*year + eps
#
# Helsinki is set as the Treatment reference for C(municipality), so:
#   • The intercept represents Helsinki prices (for the reference building type
#     and at year = 0, rescaled by the year coefficient).
#   • Every other municipality coefficient is the log price difference
#     relative to Helsinki (negative = cheaper than Helsinki).
# Weights: n_sales.
print("\n" + "="*70)
print("Model 2 - Fixed-effects WLS (municipality FEs, Helsinki = reference)")
print("="*70)

formula_2 = (
    "log_price ~ C(municipality, Treatment('Helsinki'))"
    " + C(building_type) + year"
)
m2 = smf.wls(formula_2, data=train, weights=train["n_sales"]).fit()

# Full summary has 272 municipality lines - save to file, print key stats only
summary_path = os.path.join(OUTPUTS_DIR, "model2_fe_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(str(m2.summary()))
print(f"Full summary saved -> {summary_path}")
print(f"\nModel 2 - Overall fit")
print(f"  R2          : {m2.rsquared:.4f}")
print(f"  Adj. R2     : {m2.rsquared_adj:.4f}")
print(f"  Observations: {int(m2.nobs):,}")

# Non-municipality coefficients (building type + year)
key_params = m2.params[
    [p for p in m2.params.index if "municipality" not in p]
]
key_pvalues = m2.pvalues[key_params.index]
key_df = pd.DataFrame({
    "coefficient": key_params.round(4),
    "pct_effect" : ((np.exp(key_params) - 1) * 100).round(1),
    "p_value"    : key_pvalues.round(4),
})
print("\nKey coefficients (excluding municipality FEs):")
print(key_df.to_string())

# Top 10 most expensive and cheapest municipalities vs Helsinki
muni_params = m2.params[[p for p in m2.params.index if "municipality" in p]]
muni_df = pd.DataFrame({
    "log_diff_vs_helsinki": muni_params,
    "pct_diff_vs_helsinki": (np.exp(muni_params) - 1) * 100,
})
muni_df.index = (
    muni_df.index
    .str.replace(r"C\(municipality.*\)\[T\.", "", regex=True)
    .str.replace(r"\]", "", regex=True)
)
muni_df = muni_df.sort_values("pct_diff_vs_helsinki", ascending=False)

print("\nTop 10 most expensive municipalities vs Helsinki:")
print(muni_df.head(10).round(2).to_string())

print("\nTop 10 cheapest municipalities vs Helsinki:")
print(muni_df.tail(10).round(2).to_string())

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
    print(f"  Train R2          : {model.rsquared:.4f}")
    print(f"  Test  RMSE (EUR/m2): {rmse:,.0f}")
    print(f"  Test  MAE  (EUR/m2): {mae:,.0f}")
    print(f"  Test  MAPE       : {mape:.1f}%")
    return {
        "model"     : name,
        "train_r2"  : round(model.rsquared, 4),
        "test_rmse" : round(rmse, 0),
        "test_mae"  : round(mae, 0),
        "test_mape" : round(mape, 2),
    }

# Filter test set to municipalities seen in training (FE model requires this)
known_munis = train["municipality"].unique()
test_known  = test[test["municipality"].isin(known_munis)].copy()
dropped     = len(test) - len(test_known)
if dropped:
    print(f"  Note: {dropped} test rows dropped (municipalities unseen in training)")

metrics = []
metrics.append(evaluate(m1, test,       "Model 1 - Simple"))
metrics.append(evaluate(m2, test_known, "Model 2 - Fixed Effects"))

metrics_df = pd.DataFrame(metrics)
metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics saved -> {metrics_path}")

# ─── Save key coefficients ────────────────────────────────────────────────────
coef_path = os.path.join(OUTPUTS_DIR, "model2_municipality_coefs.csv")
muni_df.to_csv(coef_path)
print(f"Municipality coefficients saved -> {coef_path}")

# ─── Serialise models ─────────────────────────────────────────────────────────
for name, model in [("model1_simple", m1), ("model2_fe", m2)]:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {path}")

print("\nDone.")
