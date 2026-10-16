"""
TrueWorth — Alpha Vantage Client
Free tier: 25 calls/day, 5-20 years of annual financials.
"""
import requests
import pandas as pd
import numpy as np


class AlphaVantageClient:
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key):
        self.key = api_key

    def get_financials(self, ticker, limit=10):
        ticker = ticker.upper()
        try:
            inc = self._fetch("INCOME_STATEMENT", ticker)
            bs = self._fetch("BALANCE_SHEET", ticker)
            cf = self._fetch("CASH_FLOW", ticker)

            inc_annual = inc.get("annualReports", [])
            if not inc_annual:
                print("  AV: No income statement data")
                return pd.DataFrame()

            bs_map = {r.get("fiscalDateEnding","")[:4]: r for r in bs.get("annualReports",[])}
            cf_map = {r.get("fiscalDateEnding","")[:4]: r for r in cf.get("annualReports",[])}

            records = []
            for item in inc_annual[:limit]:
                yr = item.get("fiscalDateEnding","")[:4]
                if not yr: continue

                row = {"year": int(yr)}
                row["total_revenue"] = self._num(item, "totalRevenue")
                row["gross_profit"] = self._num(item, "grossProfit")
                row["operating_income"] = self._num(item, "operatingIncome")
                row["net_income"] = self._num(item, "netIncome")
                row["ebitda"] = self._num(item, "ebitda")
                row["interest_expense"] = self._num(item, "interestExpense")
                if row["interest_expense"] and row["interest_expense"] > 0:
                    row["interest_expense"] = -row["interest_expense"]
                row["tax_provision"] = self._num(item, "incomeTaxExpense")

                c = cf_map.get(yr, {})
                row["operating_cashflow"] = self._num(c, "operatingCashflow")
                row["capex"] = self._num(c, "capitalExpenditures")
                if row["capex"] and row["capex"] > 0:
                    row["capex"] = -row["capex"]
                fcf = None
                if row.get("operating_cashflow") and row.get("capex"):
                    fcf = row["operating_cashflow"] + row["capex"]
                row["free_cashflow"] = fcf if fcf is not None else np.nan

                b = bs_map.get(yr, {})
                row["total_assets"] = self._num(b, "totalAssets")
                row["total_debt"] = self._num(b, "shortLongTermDebtTotal") or self._num(b, "longTermDebt")
                row["total_equity"] = self._num(b, "totalShareholderEquity")
                row["cash_and_equivalents"] = self._num(b, "cashAndCashEquivalentsAtCarryingValue") or self._num(b, "cashAndShortTermInvestments")

                for k in row:
                    if row[k] is None: row[k] = np.nan
                records.append(row)

            if not records: return pd.DataFrame()
            df = pd.DataFrame(records).sort_values("year").drop_duplicates(subset=["year"]).reset_index(drop=True)
            print(f"  AV: {len(df)} years ({df['year'].min()}-{df['year'].max()})")
            return df

        except Exception as e:
            print(f"  AV: Error — {e}")
            return pd.DataFrame()

    def _fetch(self, function, symbol):
        try:
            resp = requests.get(self.BASE, params={
                "function": function, "symbol": symbol, "apikey": self.key
            }, timeout=15)
            if resp.status_code != 200:
                print(f"  AV: HTTP {resp.status_code} for {function}")
                return {}
            data = resp.json()
            if "Error Message" in data:
                print(f"  AV: {data['Error Message']}")
                return {}
            if "Note" in data:
                print(f"  AV: Rate limited — {data['Note'][:60]}")
                return {}
            return data
        except Exception as e:
            print(f"  AV: {e}")
            return {}

    def _num(self, d, key):
        val = d.get(key, "None")
        if val is None or val == "None" or val == "": return np.nan
        try: return float(val)
        except: return np.nan