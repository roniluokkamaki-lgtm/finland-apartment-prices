# Finland Apartment Price Estimator — Project Plan

## Overview

A three-phase project to build a Finnish apartment price estimator using public data from Statistics Finland.

---

## Phase 1 — Data

- Fetch apartment price data and explaining variables from **Statistics Finland** (stat.fi PxWeb API)
- Key variables to look for: price per m², location (municipality/region), building type, year of construction, floor area, number of rooms
- Save raw API responses to `data/raw/`
- Clean and process data to `data/processed/`

## Phase 2 — Modelling

- Exploratory data analysis in a Jupyter notebook (`notebooks/`)
- Build regression specifications to explain apartment prices in terms of the available explaining variables
- Start with OLS, then evaluate regularized regression or gradient boosting as needed
- Select the best model based on fit and interpretability
- Serialize the trained model to `models/`

## Phase 3 — UI

- Streamlit app (`app/app.py`) where the user inputs explaining variable values (location, size, building type, etc.)
- App loads the trained model and returns a price estimate

---

## Data Source

**Statistics Finland** — [stat.fi](https://stat.fi)
- Free public API (PxWeb)
- Covers the whole country including Helsinki and other municipalities
