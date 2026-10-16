"""
TrueWorth Data Pipeline
Alpha Vantage (primary, 5-20yr financials) + yfinance (price, news, fallback).
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
warnings.filterwarnings("ignore")


class DataPipeline:
    def __init__(self, ticker):
        self.ticker_symbol = ticker.upper().strip()
        self.yf_ticker = yf.Ticker(self.ticker_symbol)
        self.info = {}; self.company_meta = {}
        self.income_stmt = pd.DataFrame(); self.balance_sheet = pd.DataFrame()
        self.cashflow = pd.DataFrame(); self.price_history = pd.DataFrame()
        self.news = []; self.log = []; self.data_source = "yfinance"
        self.financials_df = pd.DataFrame()

    def fetch_info(self):
        try: self.info = self.yf_ticker.info or {}
        except Exception as e: self._log(f"WARNING: {e}"); self.info = {}
        if not self.info: self._log("ERROR: No info returned"); return {}
        self.company_meta = {
            "name": self.info.get("longName") or self.info.get("shortName","N/A"),
            "sector": self.info.get("sector","N/A"), "industry": self.info.get("industry","N/A"),
            "market_cap": self.info.get("marketCap"), "currency": self.info.get("currency","USD"),
            "exchange": self.info.get("exchange","N/A"),
            "current_price": self.info.get("currentPrice") or self.info.get("regularMarketPrice") or self.info.get("previousClose"),
            "shares_outstanding": self.info.get("sharesOutstanding"),
            "beta": self.info.get("beta"), "total_debt": self.info.get("totalDebt"),
            "total_cash": self.info.get("totalCash"), "country": self.info.get("country","N/A"),
        }
        self._log(f"OK: {self.company_meta['name']} ({self.ticker_symbol})")
        return self.company_meta

    def fetch_financials(self):
        # ── Try Alpha Vantage ──
        api_key = getattr(config, "AV_API_KEY", "")
        if api_key and api_key != "your_av_key_here":
            try:
                from data.av_client import AlphaVantageClient
                av = AlphaVantageClient(api_key)
                av_df = av.get_financials(self.ticker_symbol)
                if not av_df.empty and len(av_df) >= 3:
                    self.financials_df = av_df
                    self.data_source = "Alpha Vantage"
                    self._log(f"OK: Financials via Alpha Vantage — {len(av_df)} years")
                    return
                else:
                    self._log("WARNING: AV returned insufficient data, trying yfinance")
            except Exception as e:
                self._log(f"WARNING: AV failed — {e}")

        # ── Fallback: yfinance ──
        self._log("INFO: Using yfinance for financials")
        for name, attr in [("income_stmt","income_stmt"),("balance_sheet","balance_sheet"),("cashflow","cashflow")]:
            try:
                df = getattr(self.yf_ticker, attr)
                if df is not None and not df.empty:
                    setattr(self, name, df)
                    self._log(f"OK: {name} — {len(df.columns)}yr")
                else:
                    setattr(self, name, pd.DataFrame())
            except Exception as e:
                setattr(self, name, pd.DataFrame())
                self._log(f"ERROR: {name} — {e}")
        self.data_source = "yfinance"

    def fetch_price_history(self, period="10y"):
        try:
            self.price_history = self.yf_ticker.history(period=period)
            if self.price_history is not None and not self.price_history.empty:
                self._log(f"OK: Price — {len(self.price_history)} days")
            else: self.price_history = pd.DataFrame()
        except Exception as e: self.price_history = pd.DataFrame()

    def fetch_news(self):
        try:
            raw = self.yf_ticker.news or []; self.news = []
            for item in raw:
                entry = {"title": item.get("title",""), "publisher": item.get("publisher",""),
                         "link": item.get("link",""), "published": ""}
                pt = item.get("providerPublishTime") or item.get("publishedDate","")
                if isinstance(pt,(int,float)): entry["published"] = datetime.fromtimestamp(pt).strftime("%Y-%m-%d %H:%M")
                elif isinstance(pt,str): entry["published"] = pt
                if entry["title"]: self.news.append(entry)
            self._log(f"OK: News — {len(self.news)} headlines")
        except Exception as e: self.news = []

    def validate(self):
        checks = {"valid_ticker": bool(self.info)}
        sector = (self.info.get("sector") or "").lower()
        industry = (self.info.get("industry") or "").lower()
        checks["not_financial"] = not any(kw in sector or kw in industry for kw in config.FINANCIAL_SECTORS)
        if not self.financials_df.empty: n = len(self.financials_df)
        elif not self.income_stmt.empty: n = len(self.income_stmt.columns)
        else: n = 0
        checks["sufficient_history"] = n >= config.MIN_HISTORY_YEARS
        checks["history_years"] = n
        checks["is_usd"] = self.info.get("currency","") == "USD"
        checks["data_source"] = self.data_source
        checks["overall_pass"] = all(checks.get(c,False) for c in ["valid_ticker","not_financial","sufficient_history"])
        return checks

    def get_financials_table(self):
        if not self.financials_df.empty: return self.financials_df
        if self.income_stmt.empty and self.cashflow.empty: return pd.DataFrame()
        records = []; years = self.income_stmt.columns if not self.income_stmt.empty else self.cashflow.columns
        for year in years:
            row = {"year": year.year if hasattr(year,"year") else str(year)}
            if not self.income_stmt.empty and year in self.income_stmt.columns:
                col = self.income_stmt[year]
                for f in ["Total Revenue","Operating Income","Net Income","EBITDA",
                          "Gross Profit","Interest Expense","Tax Provision"]:
                    row[f.lower().replace(" ","_")] = self._sg(col,f)
            if not self.cashflow.empty and year in self.cashflow.columns:
                cf = self.cashflow[year]
                row["operating_cashflow"] = self._sg(cf,"Operating Cash Flow")
                row["capex"] = self._sg(cf,"Capital Expenditure")
                row["free_cashflow"] = self._sg(cf,"Free Cash Flow")
                if pd.isna(row.get("free_cashflow")):
                    ocf,cx = row.get("operating_cashflow"),row.get("capex")
                    if pd.notna(ocf) and pd.notna(cx): row["free_cashflow"] = ocf + cx
            if not self.balance_sheet.empty and year in self.balance_sheet.columns:
                bs = self.balance_sheet[year]
                row["total_assets"] = self._sg(bs,"Total Assets")
                row["total_debt"] = self._sg(bs,"Total Debt")
                row["total_equity"] = self._sg(bs,"Stockholders Equity") or self._sg(bs,"Common Stock Equity")
                row["cash_and_equivalents"] = self._sg(bs,"Cash And Cash Equivalents") or self._sg(bs,"Cash Cash Equivalents And Short Term Investments")
            records.append(row)
        return pd.DataFrame(records).sort_values("year").reset_index(drop=True)

    def run(self):
        self._log(f"\n{'='*50}\nPIPELINE: {self.ticker_symbol}\n{'='*50}")
        self.fetch_info(); self.fetch_financials(); self.fetch_price_history(); self.fetch_news()
        checks = self.validate(); ft = self.get_financials_table()
        self._log(f"SOURCE: {self.data_source} | YEARS: {checks.get('history_years',0)}")
        self._log(f"RESULT: {'PASS' if checks['overall_pass'] else 'FAIL'}")
        return {"ticker": self.ticker_symbol, "company_meta": self.company_meta, "validation": checks,
                "financials_table": ft, "price_history_days": len(self.price_history),
                "news_count": len(self.news), "data_source": self.data_source, "log": self.log}

    def _sg(self, s, k):
        try: v = s.get(k,np.nan); return float(v) if v is not None else np.nan
        except: return np.nan
    def _log(self, m): self.log.append(m); print(m)