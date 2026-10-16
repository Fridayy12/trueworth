"""
TrueWorth ML Forecaster
========================
Trains multiple models, compares via LOO-CV, uses star model for DCF.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "total_revenue_growth", "gross_margin", "operating_margin",
    "net_margin", "fcf_margin", "ebitda_margin",
    "debt_to_equity", "roa", "roe", "capex_to_revenue",
]

def _get_models():
    return {
        "XGBoost": GradientBoostingRegressor(
            n_estimators=config.XGB_N_ESTIMATORS, max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE, subsample=0.8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS, max_features="sqrt", random_state=config.RF_RANDOM_STATE),
        "Ridge": Ridge(alpha=config.RIDGE_ALPHA),
    }

def _prepare_data(features_df):
    df = features_df.copy().sort_values("year").reset_index(drop=True)
    available = [c for c in FEATURE_COLS if c in df.columns]
    target = "free_cashflow"
    if target not in df.columns or not available: return None, None, [], None
    subset = df[available + [target, "year"]].dropna(subset=[target])
    subset[available] = subset[available].fillna(subset[available].median())
    return subset[available].values, subset[target].values, available, subset["year"].values


def compare_models(features_df):
    """Run all models through LOO-CV. Returns comparison dict."""
    result = _prepare_data(features_df)
    if result[0] is None: return {"error": "Insufficient data"}
    X, y, feat_names, years = result
    if len(X) < 3: return {"error": f"Only {len(X)} data points"}

    loo = LeaveOneOut()
    all_results = {}

    for name, model in _get_models().items():
        preds = np.zeros(len(y)); residuals = []
        scaler = StandardScaler()
        for tr, te in loo.split(X):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            if name == "Ridge":
                Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
                model.fit(Xtr_s, ytr); pred = model.predict(Xte_s)[0]
            else:
                model.fit(Xtr, ytr); pred = model.predict(Xte)[0]
            preds[te[0]] = pred; residuals.append(yte[0] - pred)

        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        all_results[name] = {
            "rmse": rmse, "mae": mae,
            "rmse_pct": round(rmse/np.mean(np.abs(y))*100, 2),
            "mae_pct": round(mae/np.mean(np.abs(y))*100, 2),
            "residuals": np.array(residuals), "predictions": preds,
        }

    # Try Prophet
    try:
        from prophet import Prophet
        prophet_df = pd.DataFrame({"ds": pd.to_datetime(years, format="%Y"), "y": y})
        pp = np.zeros(len(y)); pr = []
        for tr, te in loo.split(X):
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                        growth="linear", changepoint_prior_scale=0.5)
            m.fit(prophet_df.iloc[tr])
            fut = pd.DataFrame({"ds": [prophet_df.iloc[te[0]]["ds"]]})
            pred = m.predict(fut)["yhat"].values[0]
            pp[te[0]] = pred; pr.append(y[te[0]] - pred)
        rmse = float(np.sqrt(mean_squared_error(y, pp)))
        mae = float(mean_absolute_error(y, pp))
        all_results["Prophet"] = {"rmse": rmse, "mae": mae,
            "rmse_pct": round(rmse/np.mean(np.abs(y))*100,2),
            "mae_pct": round(mae/np.mean(np.abs(y))*100,2),
            "residuals": np.array(pr), "predictions": pp}
    except: pass

    # Find star
    star_name = min(all_results, key=lambda k: all_results[k]["rmse"])
    return {"models": all_results, "star": star_name, "years": years, "actuals": y, "feature_names": feat_names}


class StarForecaster:
    """Trains and forecasts using the star model (from config or auto-detected)."""

    def __init__(self):
        self.model = None; self.residuals = []; self.rmse = None; self.mae = None
        self.is_trained = False; self.model_name = ""; self.feature_importances = {}

    def train(self, features_df, star_name=None):
        result = _prepare_data(features_df)
        if result[0] is None: return {"error": "Insufficient data"}
        X, y, feat_names, years = result
        if len(X) < 3: return {"error": f"Only {len(X)} data points"}

        self.model_name = star_name or config.STAR_MODEL
        models = _get_models()
        if self.model_name not in models: self.model_name = "Ridge"

        # LOO-CV for metrics
        loo = LeaveOneOut(); preds = np.zeros(len(y)); self.residuals = []
        scaler = StandardScaler()
        for tr, te in loo.split(X):
            m = models[self.model_name]  # fresh instance each fold
            m = type(m)(**m.get_params())
            if self.model_name == "Ridge":
                Xs = scaler.fit_transform(X[tr]); m.fit(Xs, y[tr])
                preds[te[0]] = m.predict(scaler.transform(X[te]))[0]
            else:
                m.fit(X[tr], y[tr]); preds[te[0]] = m.predict(X[te])[0]
            self.residuals.append(y[te[0]] - preds[te[0]])
        self.residuals = np.array(self.residuals)
        self.rmse = float(np.sqrt(mean_squared_error(y, preds)))
        self.mae = float(mean_absolute_error(y, preds))

        # Final model on all data
        self.model = models[self.model_name]
        self._scaler = StandardScaler()
        if self.model_name == "Ridge":
            X_s = self._scaler.fit_transform(X); self.model.fit(X_s, y)
        else:
            self.model.fit(X, y)
        self.is_trained = True; self.feature_names = feat_names
        self._X = X; self._y = y

        # Feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = dict(zip(feat_names, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_); coefs = coefs / coefs.sum() if coefs.sum() > 0 else coefs
            self.feature_importances = dict(zip(feat_names, coefs))

        return {"model": self.model_name, "rmse": self.rmse, "mae": self.mae,
                "rmse_pct": round(self.rmse/np.mean(np.abs(y))*100, 1),
                "feature_importances": {k: round(v,4) for k,v in sorted(self.feature_importances.items(), key=lambda x:-x[1])},
                "residuals_std": round(float(np.std(self.residuals)),2), "n_samples": len(y)}

    def forecast(self, features_df, years_ahead=None):
        if not self.is_trained: return {"error": "Not trained"}
        years_ahead = years_ahead or config.FORECAST_YEARS
        df = features_df.copy().sort_values("year").reset_index(drop=True)
        available = [c for c in self.feature_names if c in df.columns]
        latest = df[available].iloc[-1:].fillna(df[available].median()).values[0]
        latest_year = int(df["year"].iloc[-1])
        forecasts = []; current = latest.copy()
        for i in range(1, years_ahead + 1):
            if self.model_name == "Ridge":
                pred = float(self.model.predict(self._scaler.transform(current.reshape(1,-1)))[0])
            else:
                pred = float(self.model.predict(current.reshape(1,-1))[0])
            forecasts.append({"year": latest_year+i, "predicted_fcf": pred})
            means = np.nanmean(self._X, axis=0)
            current = current * 0.9 + means * 0.1
        return {"forecasts": forecasts, "model": self.model_name,
                "residuals_std": float(np.std(self.residuals)), "base_year": latest_year}
