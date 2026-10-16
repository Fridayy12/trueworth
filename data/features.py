"""
TrueWorth Feature Engineering + Sentiment
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def compute_financial_features(df):
    df = df.copy().sort_values("year").reset_index(drop=True)
    for col in ["total_revenue","net_income","free_cashflow","operating_income","ebitda"]:
        if col in df.columns: df[f"{col}_growth"] = df[col].pct_change() * 100
    if "total_revenue" in df.columns:
        rev = df["total_revenue"].replace(0, np.nan)
        for src,dst in [("gross_profit","gross_margin"),("operating_income","operating_margin"),
                        ("net_income","net_margin"),("ebitda","ebitda_margin"),("free_cashflow","fcf_margin")]:
            if src in df.columns: df[dst] = (df[src]/rev)*100
        if "capex" in df.columns: df["capex_to_revenue"] = (df["capex"].abs()/rev)*100
    if "total_debt" in df.columns and "total_equity" in df.columns:
        df["debt_to_equity"] = df["total_debt"]/df["total_equity"].replace(0,np.nan)
    if "net_income" in df.columns:
        if "total_assets" in df.columns: df["roa"] = (df["net_income"]/df["total_assets"].replace(0,np.nan))*100
        if "total_equity" in df.columns: df["roe"] = (df["net_income"]/df["total_equity"].replace(0,np.nan))*100
    return df

def compute_feature_summary(df):
    cols = ["total_revenue_growth","net_income_growth","free_cashflow_growth","gross_margin",
            "operating_margin","net_margin","ebitda_margin","fcf_margin","debt_to_equity","roa","roe"]
    summary = {}
    for c in cols:
        if c in df.columns:
            v = df[c].dropna()
            if len(v): summary[c] = {"mean":round(v.mean(),2),"std":round(v.std(),2),"latest":round(v.iloc[-1],2),"n":len(v)}
    return summary

def compute_sentiment_scores(news):
    try:
        import nltk; nltk.download("vader_lexicon",quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError: return {"individual_scores":[],"avg_compound":0.0,"sentiment_label":"neutral","n_headlines":0}
    sia = SentimentIntensityAnalyzer(); scores = []
    for item in news:
        t = item.get("title","")
        if t:
            vs = sia.polarity_scores(t)
            scores.append({"title":t,"compound":round(vs["compound"],4),"pos":round(vs["pos"],4),"neg":round(vs["neg"],4)})
    avg = np.mean([s["compound"] for s in scores]) if scores else 0.0
    label = "positive" if avg >= config.SENTIMENT_POSITIVE_THRESHOLD else ("negative" if avg <= config.SENTIMENT_NEGATIVE_THRESHOLD else "neutral")
    return {"individual_scores":scores,"avg_compound":round(avg,4),"sentiment_label":label,"n_headlines":len(scores)}
