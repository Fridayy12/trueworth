"""
TrueWorth — Alpha Vantage Client
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

            # Build lookup by fiscal year end date AND year
            bs_map = {}
            for r in bs.get("annualReports", []):
                fd = r.get("fiscalDateEnding", "")
                bs_map[fd] = r
                bs_map[fd[:4]] = r

            cf_map = {}
            for r in cf.get("annualReports", []):
                fd = r.get("fiscalDateEnding", "")
                cf_map[fd] = r
                cf_map[fd[:4]] = r

            records = []
            for item in inc_annual[:limit]:
                fd = item.get("fiscalDateEnding", "")
                yr = fd[:4]
                if not yr:
                    continue

                row = {"year": int(yr)}

                # Income statement
                row["total_revenue"] = self._num(item, "totalRevenue")
                row["gross_profit"] = self._num(item, "grossProfit")
                row["operating_income"] = self._num(item, "operatingIncome")
                row["net_income"] = self._num(item, "netIncome")
                row["ebitda"] = self._num(item, "ebitda")
                row["interest_expense"] = self._num(item, "interestExpense")
                if row["interest_expense"] and row["interest_expense"] > 0:
                    row["interest_expense"] = -row["interest_expense"]
                row["tax_provision"] = self._num(item, "incomeTaxExpense")

                # Cash flow — try exact date match first, then year
                c = cf_map.get(fd, cf_map.get(yr, {}))
                ocf = self._num(c, "operatingCashflow")
                capex = self._num(c, "capitalExpenditures")
                if capex is not None and not np.isnan(capex) and capex > 0:
                    capex = -capex

                row["operating_cashflow"] = ocf
                row["capex"] = capex

                # FCF = operating cash flow + capex (capex is negative)
                if pd.notna(ocf) and pd.notna(capex):
                    row["free_cashflow"] = ocf + capex
                elif pd.notna(ocf):
                    # Fallback: estimate FCF as 85% of operating cash flow
                    row["free_cashflow"] = ocf * 0.85
                elif pd.notna(row.get("net_income")):
                    # Last resort: use net income as rough FCF proxy
                    row["free_cashflow"] = row["net_income"] * 0.7
                else:
                    row["free_cashflow"] = np.nan

                # Balance sheet — try exact date match first, then year
                b = bs_map.get(fd, bs_map.get(yr, {}))
                row["total_assets"] = self._num(b, "totalAssets")
                row["total_debt"] = self._num(b, "shortLongTermDebtTotal") or self._num(b, "longTermDebt")
                row["total_equity"] = self._num(b, "totalShareholderEquity")
                row["cash_and_equivalents"] = (self._num(b, "cashAndCashEquivalentsAtCarryingValue")
                    or self._num(b, "cashAndShortTermInvestments"))

                # Replace None with NaN
                for k in row:
                    if row[k] is None:
                        row[k] = np.nan

                records.append(row)

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records).sort_values("year").drop_duplicates(subset=["year"]).reset_index(drop=True)
            fcf_valid = df["free_cashflow"].notna().sum()
            print(f"  AV: {len(df)} years ({df['year'].min()}-{df['year'].max()}) | FCF valid: {fcf_valid}/{len(df)}")
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
                print(f"  AV: Rate limited")
                return {}
            return data
        except Exception as e:
            print(f"  AV: {e}")
            return {}

    def _num(self, d, key):
        val = d.get(key, "None")
        if val is None or val == "None" or val == "" or val == "0":
            return np.nan
        try:
            return float(val)
        except:
            return np.nan