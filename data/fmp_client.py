"""
TrueWorth — Financial Modeling Prep Client
Free tier: 250 calls/day, 5-10 years of annual financials.
"""
import requests
import pandas as pd
import numpy as np


class FMPClient:
    BASE = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key):
        self.key = api_key

    def get_financials(self, ticker, limit=10):
        """Fetch annual financials. Returns DataFrame matching pipeline format."""
        ticker = ticker.upper()
        try:
            inc = self._get(f"/income-statement/{ticker}", limit)
            bs = self._get(f"/balance-sheet-statement/{ticker}", limit)
            cf = self._get(f"/cash-flow-statement/{ticker}", limit)

            if not inc:
                print("  FMP: No income statement data")
                return pd.DataFrame()

            # Build year-keyed lookup for BS and CF
            bs_map = {item.get("calendarYear") or str(item.get("date",""))[:4]: item for item in (bs or [])}
            cf_map = {item.get("calendarYear") or str(item.get("date",""))[:4]: item for item in (cf or [])}

            records = []
            for item in inc:
                yr = item.get("calendarYear") or str(item.get("date",""))[:4]
                if not yr: continue
                yr = str(yr)

                row = {"year": int(yr)}
                # Income statement
                row["total_revenue"] = item.get("revenue")
                row["gross_profit"] = item.get("grossProfit")
                row["operating_income"] = item.get("operatingIncome")
                row["net_income"] = item.get("netIncome")
                row["ebitda"] = item.get("ebitda")
                row["interest_expense"] = item.get("interestExpense")
                if row["interest_expense"]: row["interest_expense"] = -abs(row["interest_expense"])
                row["tax_provision"] = item.get("incomeTaxExpense")

                # Cash flow
                c = cf_map.get(yr, {})
                row["operating_cashflow"] = c.get("operatingCashFlow")
                row["capex"] = c.get("capitalExpenditure")
                if row["capex"] and row["capex"] > 0: row["capex"] = -row["capex"]
                fcf = c.get("freeCashFlow")
                if fcf is not None:
                    row["free_cashflow"] = fcf
                elif row.get("operating_cashflow") and row.get("capex"):
                    row["free_cashflow"] = row["operating_cashflow"] + row["capex"]
                else:
                    row["free_cashflow"] = np.nan

                # Balance sheet
                b = bs_map.get(yr, {})
                row["total_assets"] = b.get("totalAssets")
                row["total_debt"] = b.get("totalDebt") or b.get("longTermDebt")
                row["total_equity"] = b.get("totalStockholdersEquity") or b.get("totalEquity")
                row["cash_and_equivalents"] = b.get("cashAndCashEquivalents") or b.get("cashAndShortTermInvestments")

                # Replace None with NaN
                for k in row:
                    if row[k] is None: row[k] = np.nan

                records.append(row)

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records).sort_values("year").drop_duplicates(subset=["year"]).reset_index(drop=True)
            print(f"  FMP: {len(df)} years ({df['year'].min()}-{df['year'].max()})")
            return df

        except Exception as e:
            print(f"  FMP: Error — {e}")
            return pd.DataFrame()

    def _get(self, endpoint, limit=10):
        url = f"{self.BASE}{endpoint}"
        params = {"apikey": self.key, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"  FMP: HTTP {resp.status_code} for {endpoint}")
                return []
            data = resp.json()
            if isinstance(data, list): return data
            if isinstance(data, dict) and "Error Message" in data:
                print(f"  FMP: {data['Error Message']}")
                return []
            return []
        except Exception as e:
            print(f"  FMP: Request failed — {e}")
            return []