"""
TrueWorth DCF Engine — WACC, discounting, terminal value
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class DCFEngine:
    def __init__(self, company_meta, features_df=None):
        self.meta = company_meta; self.features_df = features_df; self.wacc = None; self.wacc_components = {}

    def compute_wacc(self):
        rf = config.RISK_FREE_RATE; beta = self.meta.get("beta") or 1.0
        coe = rf + beta * config.EQUITY_RISK_PREMIUM
        mc = self.meta.get("market_cap") or 0; td = self.meta.get("total_debt") or 0
        if mc <= 0:
            self.wacc = coe
            self.wacc_components = {"risk_free_rate":rf,"beta":beta,"equity_risk_premium":config.EQUITY_RISK_PREMIUM,
                "cost_of_equity":round(coe,4),"wacc":round(coe,4)}
            return self.wacc
        tv = mc + td; ew = mc/tv; dw = td/tv; tr = self._tax_rate(); cod = self._cost_of_debt()
        self.wacc = ew*coe + dw*cod*(1-tr)
        self.wacc_components = {"risk_free_rate":rf,"beta":beta,"equity_risk_premium":config.EQUITY_RISK_PREMIUM,
            "cost_of_equity":round(coe,4),"cost_of_debt":round(cod,4),"tax_rate":round(tr,4),
            "equity_weight":round(ew,4),"debt_weight":round(dw,4),"wacc":round(self.wacc,4)}
        return self.wacc

    def compute_intrinsic_value(self, forecasted_fcfs, shares_outstanding=None):
        if self.wacc is None: self.compute_wacc()
        shares = shares_outstanding or self.meta.get("shares_outstanding") or 1
        wacc = self.wacc; g = config.PERPETUITY_GROWTH_RATE
        if wacc <= g: wacc = g + 0.02

        # Guard 1: Fix negative FCF forecasts
        cleaned_fcfs = []
        has_negative = False
        for fc in forecasted_fcfs:
            fcf = fc["predicted_fcf"]
            if fcf <= 0:
                has_negative = True
                mc = self.meta.get("market_cap") or 1e9
                floor_val = max(abs(fcf) * 0.1, mc * 0.005)
                cleaned_fcfs.append({**fc, "predicted_fcf": floor_val, "was_negative": True})
            else:
                cleaned_fcfs.append({**fc, "predicted_fcf": fcf, "was_negative": False})

        # PV of forecasted FCFs
        pv_fcfs = []
        for i, fc in enumerate(cleaned_fcfs):
            yr = i+1; pv = fc["predicted_fcf"]/((1+wacc)**yr)
            pv_fcfs.append({"year":fc.get("year",yr),"fcf":fc["predicted_fcf"],"pv":pv,
                           "df":round(1/((1+wacc)**yr),4)})
        sum_pv = sum(p["pv"] for p in pv_fcfs)

        # Terminal value
        last_fcf = cleaned_fcfs[-1]["predicted_fcf"]
        tv_fcf = last_fcf*(1+g); terminal = tv_fcf/(wacc-g)
        n = len(cleaned_fcfs); pv_term = terminal/((1+wacc)**n)

        # Enterprise value
        ev = sum_pv + pv_term

        # Guard 2: EV floor = net cash
        td = self.meta.get("total_debt") or 0
        tc = self.meta.get("total_cash") or 0
        net_cash = tc - td
        if ev < net_cash and net_cash > 0:
            ev = net_cash

        # Equity value
        equity = ev - td + tc

        # Guard 3: Per-share floor at $0
        iv_ps = equity / shares
        valuation_note = None
        if iv_ps < 0:
            iv_ps = 0.01
            valuation_note = "Intrinsic value floored at ~$0 — company's projected cash flows do not support current valuation under DCF. This is common for high-growth companies that reinvest heavily."
        elif has_negative:
            valuation_note = "Some forecasted cash flows were negative and adjusted upward. Results may be less reliable for this company."

        cp = self.meta.get("current_price")
        upside = round((iv_ps-cp)/cp*100,1) if cp and cp > 0 and iv_ps > 0 else None

        return {"enterprise_value":ev,"equity_value":equity,
            "intrinsic_value_per_share":round(iv_ps,2),
            "current_price":cp,"upside_pct":upside,
            "sum_pv_fcfs":sum_pv,"terminal_value":terminal,
            "pv_terminal":pv_term,
            "terminal_pct_of_ev":round(pv_term/ev*100,1) if ev>0 else 0,
            "wacc":self.wacc,"wacc_components":self.wacc_components,
            "shares_outstanding":shares,"pv_details":pv_fcfs,
            "valuation_note":valuation_note,"had_negative_fcf":has_negative}

    def _tax_rate(self):
        if self.features_df is not None and not self.features_df.empty:
            df = self.features_df
            if "tax_provision" in df.columns and "operating_income" in df.columns:
                l = df.iloc[-1]; t,o = l.get("tax_provision"),l.get("operating_income")
                if t is not None and o is not None and o > 0:
                    r = abs(t)/abs(o)
                    if 0.05 < r < 0.5: return r
        return config.TAX_RATE_DEFAULT

    def _cost_of_debt(self):
        if self.features_df is not None and not self.features_df.empty:
            df = self.features_df
            if "interest_expense" in df.columns and "total_debt" in df.columns:
                l = df.iloc[-1]; ie,td = l.get("interest_expense"),l.get("total_debt")
                if ie is not None and td is not None and td > 0:
                    r = abs(ie)/td
                    if 0.01 < r < 0.15: return r
        return config.RISK_FREE_RATE + 0.015