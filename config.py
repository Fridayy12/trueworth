"""
TrueWorth Configuration
"""
AV_API_KEY = "E022MJPOBT2UA5N1"
# Data Pipeline
MIN_HISTORY_YEARS = 3
FINANCIAL_SECTORS = {"financial services", "banks", "insurance", "capital markets", "financial", "banking"}

# DCF Parameters
RISK_FREE_RATE = 0.043
EQUITY_RISK_PREMIUM = 0.055
PERPETUITY_GROWTH_RATE = 0.025
FORECAST_YEARS = 5
TAX_RATE_DEFAULT = 0.21

# Monte Carlo
MC_SIMULATIONS = 10_000
MC_CONFIDENCE_LEVEL = 0.90

# ML — Star model selected via T1 testing
STAR_MODEL = "Ridge"  # Options: "Ridge", "XGBoost", "RandomForest"
RIDGE_ALPHA = 1.0
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.1

# Sentiment
SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05
