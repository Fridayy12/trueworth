import os

# Try Streamlit secrets first, then environment variable
try:
    import streamlit as st
    AV_API_KEY = st.secrets.get("AV_API_KEY", "")
except:
    AV_API_KEY = os.environ.get("AV_API_KEY", "")

MIN_HISTORY_YEARS = 3
FINANCIAL_SECTORS = {"financial services", "banks", "insurance", "capital markets", "financial", "banking"}

RISK_FREE_RATE = 0.043
EQUITY_RISK_PREMIUM = 0.055
PERPETUITY_GROWTH_RATE = 0.025
FORECAST_YEARS = 5
TAX_RATE_DEFAULT = 0.21

MC_SIMULATIONS = 10_000
MC_CONFIDENCE_LEVEL = 0.90

STAR_MODEL = "Ridge"
RIDGE_ALPHA = 1.0
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.1

SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05