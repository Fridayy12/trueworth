"""
TrueWorth Monte Carlo + Sensitivity Analysis
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class MonteCarloSimulator:
    def __init__(self, base_dcf_result, ml_residuals, forecasted_fcfs):
        self.base = base_dcf_result; self.residuals = ml_residuals
        self.base_fcfs = [f["predicted_fcf"] for f in forecasted_fcfs]
        self.wacc = base_dcf_result["wacc"]; self.shares = base_dcf_result["shares_outstanding"]
        self.net_debt = base_dcf_result["enterprise_value"] - base_dcf_result["equity_value"]

    def run(self, n=None):
        n = n or config.MC_SIMULATIONS
        res_std = float(np.std(self.residuals))
        if res_std < 1: res_std = abs(np.mean(self.base_fcfs)) * 0.10
        res_mean = float(np.mean(self.residuals))
        g = config.PERPETUITY_GROWTH_RATE; values = np.zeros(n)
        for i in range(n):
            fcfs = [f + np.random.normal(res_mean, res_std*(1+0.15*j)) for j,f in enumerate(self.base_fcfs)]
            w = max(self.wacc + np.random.normal(0,0.01), 0.03)
            sg = np.clip(g + np.random.normal(0,0.005), 0.005, 0.04)
            if w <= sg: w = sg + 0.02
            pv = sum(f/((1+w)**(j+1)) for j,f in enumerate(fcfs))
            tf = fcfs[-1]*(1+sg); tv = tf/(w-sg); pvt = tv/((1+w)**len(fcfs))
            values[i] = (pv + pvt - self.net_debt) / self.shares
        self.results = values
        ci = config.MC_CONFIDENCE_LEVEL; lp = (1-ci)/2*100; up = (1-(1-ci)/2)*100
        p1,p99 = np.percentile(values,[1,99]); cl = values[(values>=p1)&(values<=p99)]
        cp = self.base.get("current_price")
        pu = round(float(np.mean(values>cp))*100,1) if cp and cp>0 else None
        return {"n_simulations":n,"mean":round(float(np.mean(values)),2),"median":round(float(np.median(values)),2),
            "std":round(float(np.std(values)),2),"ci_lower":round(float(np.percentile(values,lp)),2),
            "ci_upper":round(float(np.percentile(values,up)),2),"ci_level":ci,
            "p10":round(float(np.percentile(values,10)),2),"p25":round(float(np.percentile(values,25)),2),
            "p50":round(float(np.percentile(values,50)),2),"p75":round(float(np.percentile(values,75)),2),
            "p90":round(float(np.percentile(values,90)),2),"current_price":cp,
            "prob_undervalued_pct":pu,"all_values":values}

class SensitivityAnalyzer:
    def __init__(self, base_dcf, forecasted_fcfs, shares, net_debt):
        self.base_fcfs = [f["predicted_fcf"] for f in forecasted_fcfs]
        self.wacc = base_dcf["wacc"]; self.g = config.PERPETUITY_GROWTH_RATE
        self.shares = shares; self.nd = net_debt; self.base_iv = base_dcf["intrinsic_value_per_share"]

    def run(self, v=0.15):
        def dcf(fcfs, w, g):
            if w<=g: w=g+0.02
            pv = sum(f/((1+w)**(j+1)) for j,f in enumerate(fcfs))
            tv = fcfs[-1]*(1+g)/(w-g); pvt = tv/((1+w)**len(fcfs))
            return (pv+pvt-self.nd)/self.shares
        r = []
        lo = [f*(1-v) for f in self.base_fcfs]; hi = [f*(1+v) for f in self.base_fcfs]
        r.append({"name":"FCF Forecast","base":round(self.base_iv,2),"low":round(dcf(lo,self.wacc,self.g),2),
            "high":round(dcf(hi,self.wacc,self.g),2)})
        r.append({"name":"WACC","base":round(self.base_iv,2),"low":round(dcf(self.base_fcfs,self.wacc*(1+v),self.g),2),
            "high":round(dcf(self.base_fcfs,self.wacc*(1-v),self.g),2)})
        gl = max(self.g*(1-v*2),0.005); gh = min(self.g*(1+v*2),self.wacc-0.01)
        r.append({"name":"Terminal Growth","base":round(self.base_iv,2),"low":round(dcf(self.base_fcfs,self.wacc,gl),2),
            "high":round(dcf(self.base_fcfs,self.wacc,gh),2)})
        for x in r: x["spread"] = round(abs(x["high"]-x["low"]),2)
        r.sort(key=lambda x: x["spread"], reverse=True)
        return r
