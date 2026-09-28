"""Valuation Lab — scenario-based equity valuator.

What it does:
  1. Auto-pulls a company's live fundamentals (price, shares, TTM revenue /
     earnings / free cash flow, margins, P/E) via yfinance — no API keys.
  2. Lets you build bull / base / bear scenarios: revenue growth, net-margin
     path, FCF margin, buybacks, exit P/E range, discount rate.
  3. Projects revenue -> net income -> EPS -> price range (EPS x exit PE),
     discounts back to present value, and probability-weights the scenarios
     into one expected fair value.
  4. Cross-checks your assumptions with a reverse DCF ("what growth is the
     current price already implying?") and a sensitivity grid.

Method: scenario-based earnings-power valuation (EPS x exit multiple), the
same family of models professional fundamental shops run. See
VALUATION_METHOD.md for the research behind every choice.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Valuation Lab", page_icon="🔬", layout="wide")

# --------------------------------------------------------------------------
# styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .scen-bull {background:#0e5c2f;color:#fff;font-weight:700;text-align:center;
        padding:8px;border-radius:6px;letter-spacing:2px}
    .scen-base {background:#1d4ed8;color:#fff;font-weight:700;text-align:center;
        padding:8px;border-radius:6px;letter-spacing:2px}
    .scen-bear {background:#991b1b;color:#fff;font-weight:700;text-align:center;
        padding:8px;border-radius:6px;letter-spacing:2px}
    div[data-testid="stMetricValue"] {font-size:1.35rem}
    </style>
    """,
    unsafe_allow_html=True,
)

HORIZON = 5  # projection years


# --------------------------------------------------------------------------
# data layer — everything from free public sources, no keys
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company(ticker: str) -> dict:
    """Pull live fundamentals + history. Never raises — returns {'error': msg}
    on failure so the UI can fall back to manual entry."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        try:
            price = float(t.fast_info.get("last_price"))
        except Exception:
            price = None
    if not price:
        return {"error": f"Could not fetch data for '{ticker}'. Check the ticker or enter data manually."}

    out = {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "price": float(price),
        "market_cap": info.get("marketCap"),
        "shares": info.get("sharesOutstanding"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg": info.get("pegRatio"),
        "ttm_margin": (info.get("profitMargins") or 0) * 100,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }

    # --- annual history: revenue, net income, FCF, margins -----------------
    try:
        fin = t.financials
        rev = fin.loc["Total Revenue"].sort_index() if "Total Revenue" in fin.index else None
        ni = None
        for key in ("Net Income", "Net Income Common Stockholders"):
            if key in fin.index:
                ni = fin.loc[key].sort_index()
                break
    except Exception:
        rev, ni = None, None
    try:
        cf = t.cashflow
        ocf = cf.loc["Operating Cash Flow"].sort_index() if "Operating Cash Flow" in cf.index else None
        capex = cf.loc["Capital Expenditure"].sort_index() if "Capital Expenditure" in cf.index else None
        fcf = (ocf + capex) if (ocf is not None and capex is not None) else None  # capex is negative
    except Exception:
        fcf = None

    hist = {}
    if rev is not None and len(rev) >= 2:
        rev = rev.dropna()
        yrs = len(rev) - 1
        hist["rev_cagr_5y"] = (rev.iloc[-1] / rev.iloc[0]) ** (1 / yrs) - 1 if yrs and rev.iloc[0] else None
        hist["rev_cagr_3y"] = (rev.iloc[-1] / rev.iloc[-3]) ** (1 / 2) - 1 if len(rev) >= 3 and rev.iloc[-3] else None
        hist["revenue_ttm"] = float(rev.iloc[-1])
    if ni is not None and rev is not None:
        common = pd.DataFrame({"rev": rev, "ni": ni}).dropna()
        if len(common):
            margins = (common["ni"] / common["rev"] * 100)
            hist["margin_min_5y"] = float(margins.min())
            hist["margin_max_5y"] = float(margins.max())
            hist["margin_avg_5y"] = float(margins.mean())
            hist["ni_ttm"] = float(common["ni"].iloc[-1])
    if fcf is not None and rev is not None:
        common = pd.DataFrame({"rev": rev, "fcf": fcf}).dropna()
        if len(common):
            fm = common["fcf"] / common["rev"] * 100
            hist["fcf_margin_avg"] = float(fm.mean())
            hist["fcf_ttm"] = float(common["fcf"].iloc[-1])
    out["hist"] = hist
    return out


# --------------------------------------------------------------------------
# valuation engine
# --------------------------------------------------------------------------
def project(sc: dict, base: dict, years: int = HORIZON) -> pd.DataFrame:
    """Scenario projection. Returns one row per year with revenue, net income,
    EPS, FCF/share, target price range, present values and CAGRs."""
    rows = []
    rev = base["revenue_ttm"]
    shares = base["shares"]
    r = sc["discount"] / 100.0
    for t in range(1, years + 1):
        rev *= 1 + sc["rev_growth"] / 100.0
        margin = sc["margin_start"] + (sc["margin_end"] - sc["margin_start"]) * t / years
        ni = rev * margin / 100.0
        shares *= 1 + sc["share_change"] / 100.0
        eps = ni / shares if shares else np.nan
        fcf = rev * sc["fcf_margin"] / 100.0
        fcf_ps = fcf / shares if shares else np.nan
        p_lo, p_hi = eps * sc["pe_lo"], eps * sc["pe_hi"]
        rows.append({
            "Year": t,
            "Revenue": rev,
            "Net income": ni,
            "Net margin %": margin,
            "EPS": eps,
            "FCF/share": fcf_ps,
            "Price lo": p_lo,
            "Price hi": p_hi,
            "PV lo": p_lo / (1 + r) ** t,
            "PV hi": p_hi / (1 + r) ** t,
            "CAGR lo %": ((p_lo / base["price"]) ** (1 / t) - 1) * 100,
            "CAGR hi %": ((p_hi / base["price"]) ** (1 / t) - 1) * 100,
        })
    return pd.DataFrame(rows)


def implied_growth(mktcap: float, fcf0: float, r: float = 0.10,
                   n: int = HORIZON, gt: float = 0.025):
    """Reverse DCF: what constant FCF growth rate g over n years justifies the
    current market cap? Returns None if FCF is non-positive (nothing sane to
    solve for) or the market cap is absurd."""
    if not fcf0 or fcf0 <= 0 or r <= gt:
        return None

    def pv(g):
        v, f = 0.0, fcf0
        for t in range(1, n + 1):
            f *= 1 + g
            v += f / (1 + r) ** t
        v += (f * (1 + gt) / (r - gt)) / (1 + r) ** n
        return v

    lo, hi = -0.30, 0.60
    if pv(lo) > mktcap:
        return lo  # priced for shrinkage — deep value or broken model
    if pv(hi) < mktcap:
        return hi  # priced for >60% growth — mania or model miss
    for _ in range(60):
        mid = (lo + hi) / 2
        if pv(mid) < mktcap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# --------------------------------------------------------------------------
# sidebar: ticker + company snapshot
# --------------------------------------------------------------------------
st.title("🔬 Valuation Lab")
st.caption("Scenario-based equity valuation — bull / base / bear projections, "
           "present values and a probability-weighted fair value. Research tooling, not advice.")

with st.sidebar:
    st.header("Company")
    ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "AAPL")).strip().upper()
    if st.button("Load company", type="primary") or "co" not in st.session_state:
        st.session_state["ticker"] = ticker
        with st.spinner(f"Fetching {ticker}…"):
            st.session_state["co"] = fetch_company(ticker)
    co = st.session_state.get("co", {})

if co.get("error"):
    st.error(co["error"])
    st.info("Tip: check the ticker symbol (e.g. AAPL, MSFT, NVDA). US listings work best.")
    st.stop()

# company snapshot bar
h = co.get("hist", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"${co['price']:,.2f}")
c2.metric("Market cap", f"${co['market_cap'] / 1e9:,.1f}B" if co.get("market_cap") else "n/a")
c3.metric("Trailing P/E", f"{co['trailing_pe']:.1f}×" if co.get("trailing_pe") else "n/a")
c4.metric("TTM net margin", f"{co['ttm_margin']:.1f}%" if co.get("ttm_margin") else "n/a")
cagr5 = h.get("rev_cagr_5y")
c5.metric("5-yr revenue CAGR", f"{cagr5 * 100:.1f}%" if cagr5 is not None else "n/a")
st.caption(f"**{co['name']}**" + (f" · {co['sector']} — {co['industry']}" if co.get("sector") else ""))

# --- auto-derived anchors ---------------------------------------------------
anchor_rev = (h.get("rev_cagr_5y") or 0.08) * 100          # default base growth
anchor_margin = co.get("ttm_margin") or 15.0
anchor_fcfm = h.get("fcf_margin_avg") or (anchor_margin * 0.9)
anchor_pe = co.get("trailing_pe") or 20.0

with st.expander("📊 Auto-derived anchors — sanity checks for your inputs", expanded=False):
    a1, a2, a3 = st.columns(3)
    with a1:
      with st.container(border=True):
          st.markdown("**Growth history**")
          st.write(f"5-yr revenue CAGR: **{cagr5 * 100:.1f}%**" if cagr5 is not None else "n/a")
          c3y = h.get("rev_cagr_3y")
          st.write(f"3-yr revenue CAGR: **{c3y * 100:.1f}%**" if c3y is not None else "")
          st.caption("Base case near or below the 5-yr CAGR is the disciplined default — "
                     "sustained acceleration is rare (see methodology).")
    with a2:
      with st.container(border=True):
          st.markdown("**Margin history**")
          if h.get("margin_avg_5y") is not None:
              st.write(f"5-yr avg: **{h['margin_avg_5y']:.1f}%** · range {h['margin_min_5y']:.1f}–{h['margin_max_5y']:.1f}%")
          st.write(f"TTM: **{anchor_margin:.1f}%** · FCF margin avg: **{anchor_fcfm:.1f}%**")
          st.caption("Margins mean-revert. A target far above the historical max needs a named reason "
                     "(pricing power, mix shift, scale).")
    with a3:
      with st.container(border=True):
          st.markdown("**What the price already implies**")
          ig = implied_growth(co["market_cap"], h.get("fcf_ttm"), r=0.10) if co.get("market_cap") else None
          if ig is not None:
              st.write(f"Reverse DCF: **{ig * 100:.1f}%** annual FCF growth for 5 yrs, then 2.5%, at 10% discount")
              st.caption("If your base-case growth is *below* this, you're implicitly saying the stock is overvalued. "
                         "If above, you see something the market doesn't — write down what.")
          else:
              st.write("n/a (needs positive TTM free cash flow)")

# manual overrides when auto data is thin
base = {
    "price": co["price"],
    "revenue_ttm": h.get("revenue_ttm"),
    "shares": co.get("shares"),
    "market_cap": co.get("marketCap") or co.get("market_cap"),
}
needs_manual = not base["revenue_ttm"] or not base["shares"]
if needs_manual:
    st.warning("Auto-fetch missed revenue or share count — enter them manually:")
    m1, m2 = st.columns(2)
    base["revenue_ttm"] = m1.number_input("TTM revenue ($B)", value=100.0, min_value=0.01) * 1e9
    base["shares"] = m2.number_input("Shares outstanding (B)", value=1.0, min_value=0.001) * 1e9
base["market_cap"] = base["market_cap"] or base["price"] * base["shares"]


# --------------------------------------------------------------------------
# scenario inputs
# --------------------------------------------------------------------------
def scenario_inputs(key: str, title: str, css: str, defaults: dict) -> dict:
    st.markdown(f'<div class="scen-{css}">{title}</div>', unsafe_allow_html=True)
    sc = {}
    sc["rev_growth"] = st.number_input("Revenue growth %/yr", -50.0, 100.0,
                                       defaults["rev_growth"], 0.5, key=f"{key}_g",
                                       help="Annual revenue compounding over the horizon.")
    cm1, cm2 = st.columns(2)
    sc["margin_start"] = cm1.number_input("Margin now %", -50.0, 80.0, defaults["margin_start"], 0.5,
                                          key=f"{key}_ms", help="Defaults to TTM net margin.")
    sc["margin_end"] = cm2.number_input(f"Margin yr{HORIZON} %", -50.0, 80.0, defaults["margin_end"], 0.5,
                                        key=f"{key}_me", help="Where margins settle. Glides linearly.")
    sc["fcf_margin"] = st.number_input("FCF margin %", -50.0, 100.0, defaults["fcf_margin"], 0.5,
                                       key=f"{key}_fcf",
                                       help="Free cash flow as % of revenue. Defaults to 5-yr average.")
    sc["share_change"] = st.number_input("Share change %/yr", -15.0, 15.0, defaults["share_change"], 0.25,
                                         key=f"{key}_sh",
                                         help="Negative = buybacks shrink the count (boosts EPS). -2% ≈ steady buyback.")
    p1, p2 = st.columns(2)
    sc["pe_lo"] = p1.number_input("Exit P/E low", 1.0, 100.0, defaults["pe_lo"], 1.0, key=f"{key}_plo")
    sc["pe_hi"] = p2.number_input("Exit P/E high", 1.0, 100.0, defaults["pe_hi"], 1.0, key=f"{key}_phi")
    sc["discount"] = st.number_input("Discount rate %", 1.0, 30.0, defaults["discount"], 0.5,
                                     key=f"{key}_dr",
                                     help="Your required return. 10% is a standard equity hurdle.")
    sc["prob"] = st.slider("Probability %", 0, 100, defaults["prob"], 5, key=f"{key}_pr")
    return sc


st.subheader("Scenario assumptions")
s1, s2, s3 = st.columns(3)
with s1:
    bull = scenario_inputs("bull", "BULL CASE", "bull", {
        "rev_growth": round(anchor_rev * 1.5, 1), "margin_start": round(anchor_margin, 1),
        "margin_end": round(min(anchor_margin * 1.2, 45), 1), "fcf_margin": round(anchor_fcfm, 1),
        "share_change": -2.0, "pe_lo": round(anchor_pe * 0.9, 0), "pe_hi": round(anchor_pe * 1.2, 0),
        "discount": 10.0, "prob": 25})
with s2:
    base_sc = scenario_inputs("base", "BASE CASE", "base", {
        "rev_growth": round(anchor_rev, 1), "margin_start": round(anchor_margin, 1),
        "margin_end": round(anchor_margin, 1), "fcf_margin": round(anchor_fcfm, 1),
        "share_change": -1.0, "pe_lo": round(anchor_pe * 0.8, 0), "pe_hi": round(anchor_pe, 0),
        "discount": 10.0, "prob": 50})
with s3:
    bear = scenario_inputs("bear", "BEAR CASE", "bear", {
        "rev_growth": round(anchor_rev * 0.3, 1), "margin_start": round(anchor_margin, 1),
        "margin_end": round(anchor_margin * 0.8, 1), "fcf_margin": round(anchor_fcfm * 0.8, 1),
        "share_change": 0.0, "pe_lo": round(anchor_pe * 0.55, 0), "pe_hi": round(anchor_pe * 0.75, 0),
        "discount": 10.0, "prob": 25})

for sc in (bull, base_sc, bear):
    if sc["pe_lo"] > sc["pe_hi"]:
        st.warning("Exit P/E low exceeds high in one scenario — swap them.")
        st.stop()

# --------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------
proj = {"Bull": project(bull, base), "Base": project(base_sc, base), "Bear": project(bear, base)}
years = [str(pd.Timestamp.now().year + int(y)) for y in proj["Base"]["Year"]]

t1, t2, t3 = st.tabs(["📈 Projections", "⚖️ Verdict", "📚 Methodology"])


def fmt_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Screenshot-style grid: metrics as rows, years as columns."""
    g = pd.DataFrame(index=[
        "REVENUE", "REV GROWTH", "NET INCOME", "NET INC. GROWTH", "NET INC. MARGINS",
        "EPS", "FCF / SHARE", "PE LOW EST", "PE HIGH EST",
        "SHARE PRICE LOW", "SHARE PRICE HIGH", "PV LOW (today's $)", "PV HIGH (today's $)",
        "CAGR LOW", "CAGR HIGH",
    ])
    rev_g = df["Revenue"].pct_change() * 100
    ni_g = df["Net income"].pct_change() * 100
    for i, y in enumerate(years):
        r = df.iloc[i]
        g[y] = [
            f"${r['Revenue'] / 1e9:,.1f}B",
            f"{rev_g.iloc[i]:.0f}%" if i else "—",
            f"${r['Net income'] / 1e9:,.1f}B",
            f"{ni_g.iloc[i]:.0f}%" if i else "—",
            f"{r['Net margin %']:.0f}%",
            f"${r['EPS']:.2f}",
            f"${r['FCF/share']:.2f}",
            f"{sc_map[name]['pe_lo']:.0f}",
            f"{sc_map[name]['pe_hi']:.0f}",
            f"${r['Price lo']:,.0f}",
            f"${r['Price hi']:,.0f}",
            f"${r['PV lo']:,.0f}",
            f"${r['PV hi']:,.0f}",
            f"{r['CAGR lo %']:+.0f}%",
            f"{r['CAGR hi %']:+.0f}%",
        ]
    return g


sc_map = {"Bull": bull, "Base": base_sc, "Bear": bear}

with t1:
    st.subheader(f"Multi-year projections — {co['ticker']} @ ${co['price']:,.2f}")
    for name in ("Bull", "Base", "Bear"):
        css = {"Bull": "bull", "Base": "base", "Bear": "bear"}[name]
        st.markdown(f'<div class="scen-{css}">{name.upper()} CASE</div>', unsafe_allow_html=True)
        st.dataframe(fmt_grid(proj[name]), use_container_width=True)

    # fan chart
    st.subheader("Price fan chart")
    fig = go.Figure()
    colors = {"Bull": ("#22c55e", "rgba(34,197,94,0.15)"),
              "Base": ("#3b82f6", "rgba(59,130,246,0.15)"),
              "Bear": ("#ef4444", "rgba(239,68,68,0.15)")}
    xs = [int(y) for y in years]
    for name in ("Bear", "Base", "Bull"):  # bear first so bull draws on top
        df = proj[name]
        line, fill = colors[name]
        mid = (df["Price lo"] + df["Price hi"]) / 2
        fig.add_trace(go.Scatter(x=xs, y=df["Price hi"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=df["Price lo"], mode="lines", name=f"{name} range",
                                 line=dict(width=0), fill="tonexty", fillcolor=fill, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=mid, mode="lines+markers", name=f"{name} mid",
                                 line=dict(color=line, width=2.5)))
    fig.add_hline(y=co["price"], line_dash="dash", line_color="#f0b429",
                  annotation_text=f"Today ${co['price']:,.2f}")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=420,
                      xaxis_title="Year", yaxis_title="Share price ($)",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bands = EPS × exit P/E range each year. The dashed line is today's price — "
               "everything above it is your required growth story.")

with t2:
    st.subheader("Probability-weighted verdict")
    probs = np.array([bull["prob"], base_sc["prob"], bear["prob"]], dtype=float)
    if probs.sum() == 0:
        st.warning("Set at least one scenario probability above zero.")
        st.stop()
    probs /= probs.sum()

    # expected present value using each scenario's midpoint PV at horizon
    mids, los, his = {}, {}, {}
    for name, p in zip(("Bull", "Base", "Bear"), probs):
        df = proj[name]
        lo, hi = df["PV lo"].iloc[-1], df["PV hi"].iloc[-1]
        los[name], his[name], mids[name] = lo, hi, (lo + hi) / 2
    ev = sum(mids[n] * p for n, p in zip(("Bull", "Base", "Bear"), probs))
    exp_ret = ev / co["price"] - 1
    mos = 1 - co["price"] / ev if ev > 0 else np.nan

    with st.container(border=True):
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Expected fair value (today's $)", f"${ev:,.0f}")
        v2.metric("Expected return", f"{exp_ret * 100:+.0f}%",
                  f"{((1 + exp_ret) ** (1 / HORIZON) - 1) * 100:+.1f}% annualized")
        v3.metric("Margin of safety", f"{mos * 100:.0f}%" if mos == mos else "n/a",
                  "positive = discount to fair value" if (mos == mos and mos > 0) else "negative = paying a premium")
        v4.metric("Bull / bear PV spread", f"${his['Bull']:,.0f} / ${los['Bear']:,.0f}")

    s = pd.DataFrame([{
        "Scenario": n,
        "Probability": f"{p * 100:.0f}%",
        f"PV range yr{HORIZON}": f"${los[n]:,.0f} – ${his[n]:,.0f}",
        "Implied CAGR": f"{proj[n]['CAGR lo %'].iloc[-1]:+.0f}% to {proj[n]['CAGR hi %'].iloc[-1]:+.0f}%",
    } for n, p in zip(("Bull", "Base", "Bear"), probs)])
    st.dataframe(s, use_container_width=True, hide_index=True)

    st.subheader("Sensitivity — base case fair value (PV midpoint) in today's $")
    pe_range = np.linspace(max(5, base_sc["pe_lo"] - 5), base_sc["pe_hi"] + 5, 5)
    g_range = np.array([base_sc["rev_growth"] - 4, base_sc["rev_growth"] - 2,
                        base_sc["rev_growth"], base_sc["rev_growth"] + 2,
                        base_sc["rev_growth"] + 4])
    sens = pd.DataFrame(index=[f"PE {p:.0f}" for p in pe_range],
                        columns=[f"g {g:+.0f}%" for g in g_range], dtype=float)
    for i, pe in enumerate(pe_range):
        for j, g in enumerate(g_range):
            sc2 = dict(base_sc, rev_growth=float(g), pe_lo=float(pe), pe_hi=float(pe))
            d2 = project(sc2, base)
            sens.iloc[i, j] = (d2["PV lo"].iloc[-1] + d2["PV hi"].iloc[-1]) / 2
    st.dataframe(sens.style.format("${:,.0f}").background_gradient(cmap="RdYlGn"), use_container_width=True)
    st.caption("Green = above today's price. If most of the grid is red, the thesis needs heroic assumptions.")

    st.subheader("Reality check")
    ig = implied_growth(co["market_cap"], h.get("fcf_ttm"), r=0.10) if co.get("market_cap") else None
    if ig is not None:
        st.write(f"The current price implies **{ig * 100:.1f}%** annual FCF growth for {HORIZON} years "
                 f"(10% discount, 2.5% terminal). Your scenarios assume revenue growth of "
                 f"**{bear['rev_growth']:.0f}% / {base_sc['rev_growth']:.0f}% / {bull['rev_growth']:.0f}%** (bear/base/bull).")
        if base_sc["rev_growth"] / 100 < ig - 0.02:
            st.warning("Your base case grows *slower* than what the price already implies — "
                       "on your own numbers this stock is overvalued. Either the market knows something "
                       "you don't, or your margin/exit-multiple assumptions are doing heavy lifting.")
        elif base_sc["rev_growth"] / 100 > ig + 0.05:
            st.info("Your base case is more optimistic than the market's implied growth — "
                    "make sure you can name the specific edge (product cycle, share gains, margin expansion).")
        else:
            st.success("Your base case sits near the market's implied expectations — the valuation "
                       "hinges on execution and the exit multiple, which is the honest place for it.")

with t3:
    st.subheader("How this valuator works — and how to use it well")
    st.markdown("""
**The engine: scenario-based earnings-power valuation.** For each scenario and each year:
1. **Revenue** compounds at your assumed growth rate.
2. **Net income** = revenue × net margin, with margin gliding linearly from today's TTM margin to your year-5 target.
3. **EPS** = net income ÷ shares, with shares shrinking (buybacks) or growing (dilution) at your assumed rate.
4. **Target price** = EPS × your exit P/E range. This is the same "what will it earn, what will people pay" logic behind most fundamental price targets.
5. **Present value** discounts that future price back at your required return — because a dollar in 2030 is not a dollar today. (Many retail tools skip this; it flatters every projection.)
6. **Expected value** = Σ probability × scenario midpoint PV. One number, with the uncertainty preserved in the bull/bear spread.

**How to set inputs that don't lie to you:**
- **Growth — start from base rates, not hopes.** Look at the 5-yr revenue CAGR anchor above. Sustained acceleration beyond a company's own history is the exception; research on corporate growth (Mauboussin) shows high growth fades as companies scale. Your *bear* case should be genuinely bad (growth stalls, margins compress) — otherwise the weighting is theater.
- **Margins — respect mean reversion.** The anchor shows the 5-yr margin range. A year-5 target above the historical max needs a named mechanism: pricing power, mix shift toward software/services, operating leverage with a credible path. "They'll figure it out" is not a mechanism.
- **Exit P/E — tie it to growth (PEG discipline).** A useful check: exit P/E ÷ (year-5 earnings growth %) ≈ 1 is fair, >1.5 is rich, <0.7 is cheap (Lynch's PEG). Also sanity-check against the stock's own history and today's market: with the S&P 500 Shiller CAPE above 40 (Sep 2026), exit multiples near historical medians are already a *bearish* assumption for most large caps.
- **Buybacks — don't ignore them.** A steady -2%/yr share shrink adds ~2 points to annual EPS growth for free. Check the 10-K: has the share count actually been falling?
- **Discount rate — your hurdle, not theirs.** 10% is the standard equity hurdle; use higher for leveraged, cyclical, or story stocks. This is where risk lives in the model.
- **Probabilities — the base shouldn't be 80%.** If you can't imagine the bear case, you haven't thought about the business. 25/50/25 is a disciplined default.

**What this model deliberately does NOT do:** no terminal-value DCF with a perpetual growth guess (the terminal value usually drives 70%+ of a DCF and is the least knowable input — here the exit multiple carries that weight transparently); no quarterly precision theater; no Monte Carlo (three honest scenarios beat 10,000 draws from made-up distributions).

**Cross-checks built in:** the reverse-DCF "priced-in growth" anchor, FCF/share alongside EPS (cash is harder to fake than earnings), and the sensitivity grid. If the model says "buy" but the sensitivity grid is mostly red, the model is telling you the answer depends on everything going right.
""")
    st.caption("Methodology notes: Damodaran (investment valuation), Mauboussin — *Expectations Investing* "
               "(reverse DCF) and corporate longevity / base-rate research, Lynch (PEG). "
               "Free data via Yahoo Finance; no API keys.")
