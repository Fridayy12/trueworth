"""
TrueWorth — Streamlit App
"""
import nltk
nltk.download('vader_lexicon', quiet=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.pipeline import DataPipeline
from data.features import compute_financial_features, compute_feature_summary, compute_sentiment_scores
from models.forecaster import StarForecaster
from valuation.dcf import DCFEngine
from valuation.monte_carlo import MonteCarloSimulator, SensitivityAnalyzer
import config

st.set_page_config(page_title="TrueWorth", page_icon="◈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* Force light theme everywhere */
    .stApp, .stApp > div, .main .block-container {
        background: linear-gradient(180deg, #F8FAFF 0%, #FFFFFF 40%) !important;
        color: #1E293B !important; font-family: 'DM Sans', sans-serif;
    }
    .main .block-container p, .main .block-container span, .main .block-container label { color: #1E293B !important; }
    .app-header h1 { color: #FFFFFF !important; }
    .app-header p { color: #93C5FD !important; }
    .verdict-box .vb-text { color: #1E293B !important; }
    .wacc-cell.wacc-main .wv { color: #FFFFFF !important; }
    .wacc-cell.wacc-main .wl { color: #93C5FD !important; }
    .disclaimer { color: #94A3B8 !important; }

    /* Sidebar — dark always */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important; }
    section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div { color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label { color: #94A3B8 !important; font-size: 0.78rem; }
    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white !important; border: none; border-radius: 10px;
        font-weight: 600; padding: 12px; font-size: 0.95rem;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35); transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        box-shadow: 0 6px 20px rgba(37,99,235,0.5); transform: translateY(-1px);
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 50%, #1E40AF 100%);
        border-radius: 16px; padding: 28px 32px; margin-bottom: 20px;
        position: relative; overflow: hidden;
    }
    .app-header::before {
        content: ''; position: absolute; top: -50%; right: -20%;
        width: 300px; height: 300px; border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%);
    }
    .app-header h1 { font-family: 'Space Grotesk'; font-size: 2rem; font-weight: 700;
        color: #FFFFFF !important; margin: 0; position: relative; }
    .app-header p { color: #93C5FD !important; font-size: 0.9rem; margin-top: 4px; position: relative; }

    /* Company cards */
    .info-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
    .info-card { background: #FFFFFF; border-radius: 12px; padding: 18px 20px;
        border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: all 0.25s; }
    .info-card:hover { box-shadow: 0 6px 16px rgba(0,0,0,0.1); transform: translateY(-2px); }
    .info-card .ic-label { font-size: 0.7rem; color: #64748B !important; text-transform: uppercase;
        letter-spacing: 0.8px; font-weight: 600; }
    .info-card .ic-value { font-size: 1.25rem; font-weight: 700; color: #0F172A !important; margin-top: 6px; }
    .info-card.accent { border-left: 3px solid #3B82F6; }

    /* Verdict */
    .verdict-box { border-radius: 14px; padding: 20px 28px; margin: 16px 0;
        display: flex; align-items: center; gap: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.25s; }
    .verdict-box:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); transform: translateY(-1px); }
    .vb-bull { background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); border: 1px solid #6EE7B7; }
    .vb-bear { background: linear-gradient(135deg, #FEF2F2 0%, #FECACA 100%); border: 1px solid #FCA5A5; }
    .vb-fair { background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); border: 1px solid #93C5FD; }
    .vb-icon { font-size: 2rem; } .vb-text { font-size: 0.95rem; color: #1E293B !important; line-height: 1.5; }

    /* Metric cards */
    .mrow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
    .mrow-3 { grid-template-columns: repeat(3, 1fr); }
    .mcard { background: #FFFFFF; border-radius: 12px; padding: 18px 22px;
        border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: all 0.25s; }
    .mcard:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); transform: translateY(-1px); }
    .mcard .mc-label { font-size: 0.7rem; color: #64748B !important; text-transform: uppercase;
        letter-spacing: 0.6px; font-weight: 600; }
    .mcard .mc-val { font-size: 1.5rem; font-weight: 700; color: #0F172A !important; margin-top: 6px; }
    .mcard .mc-delta { font-size: 0.82rem; font-weight: 600; margin-top: 4px; }
    .mcard .mc-up { color: #059669 !important; } .mcard .mc-down { color: #DC2626 !important; }
    .mcard.highlight { background: linear-gradient(135deg, #EFF6FF, #DBEAFE); border-color: #93C5FD; }

    /* Section title */
    .sec-title { font-family: 'Space Grotesk'; font-size: 1.05rem; font-weight: 600;
        color: #0F172A !important; margin: 28px 0 14px 0; display: flex; align-items: center; gap: 8px; }
    .sec-title .bar { width: 4px; height: 20px; border-radius: 2px; background: #3B82F6; }

    /* WACC grid */
    .wacc-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0; }
    .wacc-cell { background: #F8FAFC; border-radius: 10px; padding: 12px 16px; border: 1px solid #E2E8F0;
        transition: all 0.2s; }
    .wacc-cell:hover { background: #F1F5F9; }
    .wacc-cell .wl { font-size: 0.68rem; color: #64748B !important; text-transform: uppercase; }
    .wacc-cell .wv { font-size: 1.05rem; font-weight: 700; color: #0F172A !important; margin-top: 4px; }
    .wacc-cell.wacc-main { background: linear-gradient(135deg, #0F172A, #1E3A5F); }
    .wacc-cell.wacc-main .wl { color: #93C5FD !important; }
    .wacc-cell.wacc-main .wv { color: #FFFFFF !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #F1F5F9; border-radius: 12px; padding: 4px; border: none; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 22px; font-weight: 500;
        color: #334155 !important; border: none; background: transparent; }
    .stTabs [aria-selected="true"] { background: #FFFFFF !important; color: #0F172A !important;
        font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Disclaimer */
    .disclaimer {
        text-align: center; padding: 12px 20px; margin-top: 24px;
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
        font-size: 0.78rem; color: #94A3B8 !important; letter-spacing: 0.3px;
    }

    /* Hide unwanted elements */
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] .stNumberInput input { color: #0F172A !important; background: #FFFFFF !important; }
    footer { display: none !important; }
    [data-testid="stMetric"] { background: #FFFFFF !important; border: 1px solid #E2E8F0;
        border-radius: 10px; padding: 12px; }
    /* Hide copy/link buttons on all elements */
    button[title="Copy to clipboard"], [data-testid="stHeaderActionElements"],
    .stCopyButton, [data-testid="StyledLinkIconContainer"],
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="app-header"><h1>◈ TrueWorth</h1><p>Intrinsic Valuation Platform</p></div>', unsafe_allow_html=True)

POPULAR = {"— Select or type below —":"","Apple (AAPL)":"AAPL","Microsoft (MSFT)":"MSFT",
    "Google (GOOGL)":"GOOGL","Amazon (AMZN)":"AMZN","NVIDIA (NVDA)":"NVDA","Tesla (TSLA)":"TSLA",
    "Meta (META)":"META","Johnson & Johnson (JNJ)":"JNJ","Walmart (WMT)":"WMT",
    "Coca-Cola (KO)":"KO","Procter & Gamble (PG)":"PG","Disney (DIS)":"DIS",
    "Netflix (NFLX)":"NFLX","Adobe (ADBE)":"ADBE","Salesforce (CRM)":"CRM"}

with st.sidebar:
    st.markdown("### ◈ Configuration")
    selected = st.selectbox("Company", list(POPULAR.keys()), label_visibility="collapsed")
    ticker = POPULAR[selected] if POPULAR[selected] else st.text_input("Enter ticker", placeholder="e.g. AAPL").upper().strip()
    if ticker: st.caption(f"→ `{ticker}`")
    st.markdown("---")
    ca,cb = st.columns(2)
    with ca: forecast_years = st.selectbox("Forecast", [3,4,5,6,7], index=2)
    with cb:
        sm={"1K":1000,"5K":5000,"10K":10000,"25K":25000,"50K":50000}
        n_sims = sm[st.selectbox("Simulations", list(sm.keys()), index=2)]
    st.markdown("##### Rates")
    risk_free = st.number_input("Risk-Free %", value=config.RISK_FREE_RATE*100, min_value=0.5, max_value=15.0, step=0.1)/100
    erp = st.number_input("Equity Premium %", value=config.EQUITY_RISK_PREMIUM*100, min_value=2.0, max_value=10.0, step=0.1)/100
    tg = st.number_input("Terminal Growth %", value=config.PERPETUITY_GROWTH_RATE*100, min_value=0.5, max_value=4.0, step=0.1)/100
    st.markdown("---")
    run_btn = st.button("Run Valuation", type="primary", use_container_width=True)

config.RISK_FREE_RATE=risk_free; config.EQUITY_RISK_PREMIUM=erp; config.PERPETUITY_GROWTH_RATE=tg
PL=dict(template="plotly_white", font=dict(family="DM Sans, sans-serif", size=12, color="#1E293B"),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", margin=dict(l=40,r=20,t=45,b=35))
COLORS={"blue":"#3B82F6","green":"#22C55E","red":"#EF4444","purple":"#8B5CF6",
    "amber":"#F59E0B","navy":"#0F172A","slate":"#64748B","teal":"#14B8A6"}

if run_btn and ticker:
    with st.spinner(f"Fetching {ticker}..."):
        pipe=DataPipeline(ticker); result=pipe.run()
    if not result["validation"]["overall_pass"]:
        st.error(f"**{ticker}** failed validation.")
        for m in result["log"]:
            if "ERROR" in m or "EXCLUDED" in m: st.warning(m)
        st.stop()
    meta=result["company_meta"]; ft=result["financials_table"]
    if ft.empty: st.error("No data."); st.stop()

    mc_v=meta.get("market_cap"); mc_s=f"${mc_v/1e9:,.1f}B" if mc_v else "N/A"
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-card accent"><div class="ic-label">Company</div><div class="ic-value">{meta.get("name",ticker)}</div></div>
        <div class="info-card"><div class="ic-label">Sector</div><div class="ic-value">{meta.get("sector","N/A")}</div></div>
        <div class="info-card"><div class="ic-label">Current Price</div><div class="ic-value">${meta.get("current_price",0):,.2f}</div></div>
        <div class="info-card"><div class="ic-label">Market Cap</div><div class="ic-value">{mc_s}</div></div>
    </div>""", unsafe_allow_html=True)

    features_df=compute_financial_features(ft); sentiment=compute_sentiment_scores(pipe.news)
    with st.spinner("Analyzing..."):
        forecaster=StarForecaster(); train_r=forecaster.train(features_df)
        if "error" in train_r: st.error(train_r["error"]); st.stop()
        fc_r=forecaster.forecast(features_df, years_ahead=forecast_years)
        if "error" in fc_r: st.error(fc_r["error"]); st.stop()
        forecasts=fc_r["forecasts"]
    dcf=DCFEngine(meta, features_df); dcf.compute_wacc(); dcf_r=dcf.compute_intrinsic_value(forecasts)
    iv=dcf_r["intrinsic_value_per_share"]; cp=dcf_r["current_price"] or 0; up=dcf_r["upside_pct"]
    with st.spinner(f"Simulating ({n_sims:,}x)..."):
        mc_sim=MonteCarloSimulator(dcf_r, forecaster.residuals, forecasts); mc_r=mc_sim.run(n=n_sims)
    nd=dcf_r["enterprise_value"]-dcf_r["equity_value"]
    sa=SensitivityAnalyzer(dcf_r, forecasts, dcf_r["shares_outstanding"], nd); sa_r=sa.run()

    # Verdict
    if up is not None:
        if up>10: cls,icon,label="vb-bull","▲","Potentially Undervalued"
        elif up<-10: cls,icon,label="vb-bear","▼","Potentially Overvalued"
        else: cls,icon,label="vb-fair","◆","Near Fair Value"
        st.markdown(f'<div class="verdict-box {cls}"><span class="vb-icon">{icon}</span><span class="vb-text"><strong>{label}</strong> — Intrinsic value <strong>${iv:,.2f}</strong> vs market <strong>${cp:,.2f}</strong> ({up:+.1f}%)</span></div>', unsafe_allow_html=True)
    
        if dcf_r.get("valuation_note"):
            st.caption(f"⚠ {dcf_r['valuation_note']}")
    # Triple Gauge
    gauge_min = max(0, min(iv, cp) * 0.3); gauge_max = max(iv, cp) * 1.8
    if gauge_max <= gauge_min: gauge_max = gauge_min + 100
    rmse_pct = train_r.get("rmse_pct", 20)
    confidence = max(0, min(100, 100 - rmse_pct * 3))
    mc_spread = mc_r["ci_upper"] - mc_r["ci_lower"]
    mc_med = mc_r["median"] if mc_r["median"] != 0 else 1
    precision = max(0, min(100, 100 - (mc_spread / abs(mc_med)) * 100))

    fig_g = make_subplots(rows=1, cols=3, specs=[[{"type":"indicator"},{"type":"indicator"},{"type":"indicator"}]],
        column_widths=[0.4, 0.3, 0.3])
    fig_g.add_trace(go.Indicator(mode="gauge+number+delta", value=iv,
        number={"prefix":"$","font":{"size":32,"family":"Space Grotesk","color":"#0F172A"}},
        delta={"reference":cp,"relative":True,"valueformat":".1%",
               "increasing":{"color":"#22C55E"},"decreasing":{"color":"#EF4444"}},
        gauge={"axis":{"range":[gauge_min,gauge_max],"tickprefix":"$","tickfont":{"size":9,"color":"#64748B"}},
            "bar":{"color":COLORS["blue"],"thickness":0.25},"bgcolor":"#F1F5F9","borderwidth":0,
            "steps":[{"range":[gauge_min,cp*0.7],"color":"#FEE2E2"},{"range":[cp*0.7,cp*0.9],"color":"#FEF3C7"},
                {"range":[cp*0.9,cp*1.1],"color":"#E0F2FE"},{"range":[cp*1.1,cp*1.4],"color":"#DCFCE7"},
                {"range":[cp*1.4,gauge_max],"color":"#BBF7D0"}],
            "threshold":{"line":{"color":"#0F172A","width":3},"thickness":0.8,"value":cp}},
        title={"text":"Valuation<br><span style='font-size:10px;color:#64748B'>▎= market price</span>",
               "font":{"size":13,"color":"#334155"}}), row=1, col=1)
    fig_g.add_trace(go.Indicator(mode="gauge+number", value=round(confidence),
        number={"suffix":"%","font":{"size":28,"family":"Space Grotesk","color":"#0F172A"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"size":9,"color":"#64748B"}},
            "bar":{"color":COLORS["teal"],"thickness":0.3},"bgcolor":"#F1F5F9","borderwidth":0,
            "steps":[{"range":[0,40],"color":"#FEE2E2"},{"range":[40,70],"color":"#FEF3C7"},{"range":[70,100],"color":"#DCFCE7"}]},
        title={"text":"Model Confidence<br><span style='font-size:10px;color:#64748B'>Based on RMSE</span>",
               "font":{"size":13,"color":"#334155"}}), row=1, col=2)
    fig_g.add_trace(go.Indicator(mode="gauge+number", value=round(precision),
        number={"suffix":"%","font":{"size":28,"family":"Space Grotesk","color":"#0F172A"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"size":9,"color":"#64748B"}},
            "bar":{"color":COLORS["purple"],"thickness":0.3},"bgcolor":"#F1F5F9","borderwidth":0,
            "steps":[{"range":[0,40],"color":"#FEE2E2"},{"range":[40,70],"color":"#FEF3C7"},{"range":[70,100],"color":"#DCFCE7"}]},
        title={"text":"Estimate Precision<br><span style='font-size:10px;color:#64748B'>Based on MC spread</span>",
               "font":{"size":13,"color":"#334155"}}), row=1, col=3)
    fig_g.update_layout(height=270, margin=dict(l=20,r=20,t=70,b=10),
        paper_bgcolor="#FFFFFF", font=dict(family="DM Sans", color="#1E293B"))
    st.plotly_chart(fig_g, use_container_width=True)

    # Tabs
    tab1,tab2,tab3,tab4 = st.tabs(["◈ DCF Valuation","⟐ Monte Carlo","▦ Financials","⟡ Forecast"])

    with tab1:
        d_str = f"{'▲ Undervalued' if up>0 else '▼ Overvalued'}" if up else ""
        d_cls = "mc-up" if up and up>0 else "mc-down"
        st.markdown(f"""<div class="mrow mrow-3">
            <div class="mcard highlight"><div class="mc-label">Intrinsic Value</div><div class="mc-val">${iv:,.2f}</div></div>
            <div class="mcard"><div class="mc-label">Market Price</div><div class="mc-val">${cp:,.2f}</div></div>
            <div class="mcard"><div class="mc-label">Upside / Downside</div><div class="mc-val">{up:+.1f}%</div><div class="mc-delta {d_cls}">{d_str}</div></div>
        </div>""", unsafe_allow_html=True)
        wc=dcf_r["wacc_components"]
        st.markdown('<div class="sec-title"><span class="bar"></span>Discount Rate (WACC)</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="wacc-row">
            <div class="wacc-cell"><div class="wl">Risk-Free</div><div class="wv">{wc.get('risk_free_rate',0)*100:.2f}%</div></div>
            <div class="wacc-cell"><div class="wl">Beta</div><div class="wv">{wc.get('beta',0):.2f}</div></div>
            <div class="wacc-cell"><div class="wl">Cost of Equity</div><div class="wv">{wc.get('cost_of_equity',0)*100:.2f}%</div></div>
            <div class="wacc-cell"><div class="wl">Cost of Debt</div><div class="wv">{wc.get('cost_of_debt',0)*100:.2f}%</div></div>
            <div class="wacc-cell"><div class="wl">Tax Rate</div><div class="wv">{wc.get('tax_rate',0)*100:.1f}%</div></div>
            <div class="wacc-cell wacc-main"><div class="wl">WACC</div><div class="wv">{wc.get('wacc',0)*100:.2f}%</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="bar"></span>Enterprise Value</div>', unsafe_allow_html=True)
        pv_f=dcf_r["sum_pv_fcfs"]/1e9; pv_t=dcf_r["pv_terminal"]/1e9; ev_b=pv_f+pv_t
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Pie(values=[abs(pv_f),abs(pv_t)], labels=["PV of Cash Flows","PV of Terminal Value"],
            hole=0.65, marker=dict(colors=[COLORS["blue"],COLORS["navy"]]),
            textinfo="label+percent", textfont=dict(size=12, color="#334155"), textposition="outside", pull=[0,0.03],
            hovertemplate="<b>%{label}</b><br>$%{value:,.1f}B<br>%{percent}<extra></extra>"))
        fig_ev.add_annotation(text=f"${ev_b:,.1f}B", x=0.5, y=0.55, font_size=22,
            font_color=COLORS["navy"], font_family="Space Grotesk", showarrow=False)
        fig_ev.add_annotation(text="Enterprise Value", x=0.5, y=0.42, font_size=11,
            font_color="#64748B", showarrow=False)
        fig_ev.update_layout(showlegend=False, height=340, margin=dict(l=60,r=60,t=20,b=20),
            paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig_ev, use_container_width=True)

    with tab2:
        ci_pct=int(config.MC_CONFIDENCE_LEVEL*100)
        st.markdown(f"""<div class="mrow">
            <div class="mcard"><div class="mc-label">Mean</div><div class="mc-val">${mc_r['mean']:,.2f}</div></div>
            <div class="mcard"><div class="mc-label">Median</div><div class="mc-val">${mc_r['median']:,.2f}</div></div>
            <div class="mcard"><div class="mc-label">{ci_pct}% CI Low</div><div class="mc-val">${mc_r['ci_lower']:,.2f}</div></div>
            <div class="mcard highlight"><div class="mc-label">{ci_pct}% CI High</div><div class="mc-val">${mc_r['ci_upper']:,.2f}</div></div>
        </div>""", unsafe_allow_html=True)
        vals=mc_r["all_values"]; p2,p98=np.percentile(vals,[2,98]); dv=vals[(vals>=p2)&(vals<=p98)]
        fig_h=go.Figure()
        fig_h.add_trace(go.Histogram(x=dv, nbinsx=100, marker=dict(color=COLORS["blue"],line=dict(width=0)), opacity=0.75,
            hovertemplate="$%{x:,.2f}<br>Count: %{y}<extra></extra>"))
        fig_h.add_vline(x=mc_r["median"], line_dash="solid", line_color=COLORS["red"], line_width=2,
            annotation_text="Median", annotation_font=dict(size=11,color=COLORS["red"]))
        if cp>0: fig_h.add_vline(x=cp, line_dash="dash", line_color=COLORS["green"], line_width=2,
            annotation_text="Market", annotation_font=dict(size=11,color=COLORS["green"]))
        fig_h.update_layout(**PL, height=380, xaxis_title="Intrinsic Value ($/share)",
            yaxis_title="Frequency", showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)
        pc1,pc2=st.columns(2)
        with pc1:
            st.markdown('<div class="sec-title"><span class="bar"></span>Percentiles</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Percentile":["10th","25th","50th","75th","90th"],
                "Value":[f"${mc_r[k]:,.2f}" for k in ["p10","p25","p50","p75","p90"]]}),
                use_container_width=True, hide_index=True)
        with pc2:
            st.markdown('<div class="sec-title"><span class="bar"></span>Sensitivity</div>', unsafe_allow_html=True)
            fig_t=go.Figure(); names=[r["name"] for r in sa_r]; base=sa_r[0]["base"]
            fig_t.add_trace(go.Bar(y=names, x=[r["low"]-base for r in sa_r], base=base,
                orientation="h", marker_color=COLORS["red"], opacity=0.85, name="−15%",
                hovertemplate="%{y}: $%{x:,.2f}<extra></extra>"))
            fig_t.add_trace(go.Bar(y=names, x=[r["high"]-base for r in sa_r], base=base,
                orientation="h", marker_color=COLORS["green"], opacity=0.85, name="+15%",
                hovertemplate="%{y}: $%{x:,.2f}<extra></extra>"))
            fig_t.add_vline(x=base, line_color=COLORS["navy"], line_width=2)
            fig_t.update_layout(**PL, height=240, barmode="overlay", xaxis_title="$/share")
            st.plotly_chart(fig_t, use_container_width=True)

    with tab3:
        fig_fin=go.Figure()
        fig_fin.add_trace(go.Bar(x=features_df["year"], y=features_df["total_revenue"]/1e9,
            name="Revenue", marker=dict(color=COLORS["blue"],cornerradius=4),
            hovertemplate="FY%{x}<br>$%{y:,.1f}B<extra>Revenue</extra>"))
        if "free_cashflow" in features_df.columns:
            fig_fin.add_trace(go.Bar(x=features_df["year"], y=features_df["free_cashflow"]/1e9,
                name="FCF", marker=dict(color=COLORS["green"],cornerradius=4),
                hovertemplate="FY%{x}<br>$%{y:,.1f}B<extra>FCF</extra>"))
        fig_fin.update_layout(**PL, barmode="group", height=360, yaxis_title="$B")
        st.plotly_chart(fig_fin, use_container_width=True)

        if not pipe.price_history.empty:
            ph = pipe.price_history.copy(); ph.index = pd.to_datetime(ph.index)
            monthly = ph["Close"].resample("ME").last().dropna()
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=monthly.index, y=monthly.values,
                name="Price", mode="lines", line=dict(color=COLORS["blue"], width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
                hovertemplate="%{x|%b %Y}: $%{y:,.2f}<extra></extra>"))
            fig_price.add_hline(y=iv, line_dash="dash", line_color=COLORS["red"], line_width=2,
                annotation_text=f"Intrinsic: ${iv:,.2f}", annotation_position="top left",
                annotation_font=dict(size=11, color=COLORS["red"]))
            if cp>0: fig_price.add_hline(y=cp, line_dash="dot", line_color=COLORS["green"], line_width=1,
                annotation_text=f"Current: ${cp:,.2f}", annotation_position="bottom right",
                annotation_font=dict(size=10, color=COLORS["green"]))
            if mc_r["ci_lower"]>0 and mc_r["ci_upper"]>0:
                fig_price.add_hrect(y0=mc_r["ci_lower"], y1=mc_r["ci_upper"],
                    fillcolor="rgba(239,68,68,0.06)", line_width=0)
            fig_price.update_layout(**PL, height=380, title="Historical Price vs Intrinsic Value",
                yaxis_title="Price ($)", showlegend=False)
            st.plotly_chart(fig_price, use_container_width=True)

        mcols=[c for c in ["gross_margin","operating_margin","net_margin","fcf_margin"] if c in features_df.columns]
        if mcols:
            clrs={"gross_margin":COLORS["blue"],"operating_margin":COLORS["amber"],
                  "net_margin":COLORS["green"],"fcf_margin":COLORS["purple"]}
            fig_m=go.Figure()
            for c in mcols:
                fig_m.add_trace(go.Scatter(x=features_df["year"], y=features_df[c],
                    name=c.replace("_"," ").title(), mode="lines+markers",
                    line=dict(color=clrs.get(c,"#333"),width=2.5), marker=dict(size=7),
                    hovertemplate="%{x}: %{y:.1f}%<extra>"+c.replace("_"," ").title()+"</extra>"))
            fig_m.update_layout(**PL, height=300, yaxis_title="%")
            st.plotly_chart(fig_m, use_container_width=True)
        summary=compute_feature_summary(features_df)
        st.dataframe(pd.DataFrame([{"Metric":k.replace("_"," ").title(),
            "Latest":f"{v['latest']:.2f}","Avg":f"{v['mean']:.2f}"} for k,v in summary.items()]),
            use_container_width=True, hide_index=True)
        if sentiment["individual_scores"]:
            sl=sentiment["sentiment_label"].upper(); sc=sentiment["avg_compound"]
            st.markdown(f'<div class="sec-title"><span class="bar"></span>Sentiment: {sl} ({sc:+.3f})</div>', unsafe_allow_html=True)
            for s in sentiment["individual_scores"][:5]:
                e="●" if s["compound"]>0.05 else ("○" if s["compound"]<-0.05 else "◌")
                clr = "#059669" if s["compound"]>0.05 else ("#DC2626" if s["compound"]<-0.05 else "#94A3B8")
                st.markdown(f'<span style="color:{clr}">{e}</span> {s["compound"]:+.3f} — {s["title"]}', unsafe_allow_html=True)

    with tab4:
        hy=features_df["year"].tolist(); hf=(features_df["free_cashflow"]/1e9).tolist()
        fy=[f["year"] for f in forecasts]; ff=[f["predicted_fcf"]/1e9 for f in forecasts]
        rs=fc_r["residuals_std"]/1e9
        fig_fc=go.Figure()
        fig_fc.add_trace(go.Scatter(x=hy, y=hf, name="Historical", mode="lines+markers",
            line=dict(color=COLORS["navy"],width=3), marker=dict(size=8),
            hovertemplate="FY%{x}<br>$%{y:,.1f}B<extra>Historical</extra>"))
        fig_fc.add_trace(go.Scatter(x=[hy[-1]]+fy, y=[hf[-1]]+ff, name="Projected",
            mode="lines+markers", line=dict(color=COLORS["red"],width=3,dash="dash"), marker=dict(size=8),
            hovertemplate="FY%{x}<br>$%{y:,.1f}B<extra>Projected</extra>"))
        upper=[hf[-1]]+[f+rs*(1+.15*i) for i,f in enumerate(ff)]
        lower=[hf[-1]]+[f-rs*(1+.15*i) for i,f in enumerate(ff)]; xb=[hy[-1]]+fy
        fig_fc.add_trace(go.Scatter(x=xb+xb[::-1], y=upper+lower[::-1], fill="toself",
            fillcolor="rgba(239,68,68,0.08)", line=dict(color="rgba(0,0,0,0)"), name="Confidence", hoverinfo="skip"))
        fig_fc.update_layout(**PL, height=400, yaxis_title="FCF ($B)",
            title=f"{forecast_years}-Year Projection")
        st.plotly_chart(fig_fc, use_container_width=True)
        st.dataframe(pd.DataFrame([{"Year":f["year"],"Projected FCF":f"${f['predicted_fcf']/1e9:,.2f}B"}
            for f in forecasts]), use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown('<div class="disclaimer">◈ TrueWorth is built for educational and analytical exploration only. It does not constitute financial advice, and should not be used as the sole basis for investment decisions.</div>', unsafe_allow_html=True)

elif not run_btn:
    st.markdown("")
    st.markdown("Select a company or enter a US ticker, then click **Run Valuation**.")