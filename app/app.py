"""
Streamlit UI -- Suomalainen asuntojen neliöhintalaskuri.
Run with: streamlit run app/app.py
"""

import json
import math
import os

import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model2_fe.json")

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(page_title="Asuntojen neliöhintalaskuri", layout="centered")

# ─── Nordic CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* Title */
h1 {
    font-weight: 600;
    font-size: 1.9rem !important;
    letter-spacing: -0.3px;
    color: #1A2B2E;
    padding-bottom: 4px;
}

/* Caption / small text */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #5C7275;
    font-size: 0.82rem;
    font-weight: 400;
}

/* Selectbox labels */
label[data-testid="stWidgetLabel"] p {
    font-weight: 500;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #3B6E76;
}

/* Metric card */
[data-testid="stMetric"] {
    background-color: #1A2B2E;
    border-radius: 10px;
    padding: 28px 32px;
}

[data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #8FAFB3 !important;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    font-size: 2.6rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
}

/* Divider */
hr {
    border-color: #D4DEDE;
    margin: 18px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, encoding="utf-8") as f:
        return json.load(f)

model = load_model()

postal_codes   = sorted(model["postal_code_coefs"].keys(), key=lambda x: int(x))
building_types = sorted(model["building_type_coefs"].keys())
current_year   = 2026
year_options   = list(range(2009, 2031))

BTYPE_FI = {
    "Blocks of flats, one-room flat"    : "Kerrostalo, yksiö",
    "Blocks of flats, two-room flat"    : "Kerrostalo, kaksio",
    "Blocks of flats, three-room flat+" : "Kerrostalo, kolmio tai isompi",
    "Terraced houses total"             : "Omakotitalo",
}

def postal_label(code):
    info = model["postal_to_area"].get(code, {})
    area = info.get("area_name", "")
    return f"{int(code):05d} \u2013 {area}"

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("Asuntojen neliöhintalaskuri")
st.caption(
    f"OLS-malli \u00b7 Tilastokeskuksen aineisto 2009\u20132023 \u00b7 "
    f"R\u00b2\u00a0=\u00a0{model['r2']:.2f} \u00b7 "
    f"Keskivirhe\u00a0\u2248\u00a0{model['test_rmse']:,}\u00a0EUR/m\u00b2"
)

st.divider()

# ─── Inputs ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    selected_postal = st.selectbox(
        "Postinumero",
        options=postal_codes,
        index=postal_codes.index("100"),
        format_func=postal_label,
    )

with col2:
    selected_btype_en = st.selectbox(
        "Talotyyppi",
        options=building_types,
        index=building_types.index("Blocks of flats, two-room flat"),
        format_func=lambda b: BTYPE_FI.get(b, b),
    )

with col3:
    selected_year = st.selectbox(
        "Vuosi",
        options=year_options,
        index=year_options.index(current_year),
    )

# ─── Prediction ───────────────────────────────────────────────────────────────
post2022  = max(0, selected_year - model["breakpoint_year"])
log_price = (
    model["intercept"]
    + model["postal_code_coefs"][selected_postal]
    + model["building_type_coefs"][selected_btype_en]
    + model["year_coef"] * selected_year
    + model["post2022_coef"] * post2022
)
price = math.exp(log_price)

st.divider()

st.metric(
    label="Hinta-arvio",
    value=f"{price:,.0f} EUR/m\u00b2",
)
