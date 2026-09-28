# 🔬 Valuation Lab

Scenario-based equity valuation — the tool from that screenshot, rebuilt with
professional-grade methods.

**What it does:** pull any stock's live fundamentals (price, shares, TTM
revenue/earnings/free cash flow, margins, P/E — no API keys), build bull/base/
bear scenarios (revenue growth, margin path, buybacks, exit P/E, discount
rate), and get multi-year price projections, present values, and one
probability-weighted fair value.

**How it's better than the screenshot tool:**
- Discounts future prices to **present value** (a dollar in 2030 ≠ a dollar today)
- **Buybacks/dilution** flow through to EPS (a -2%/yr shrink ≈ +2pts annual EPS growth)
- Margins **glide** from today's level to your target instead of assuming a flat number
- **Reverse DCF anchor**: shows what FCF growth the current price already implies,
  so you can see when your base case is braver (or more timid) than the market
- **Sensitivity grid** and probability weighting with an honest default (25/50/25)
- FCF/share alongside EPS — cash is harder to fake than earnings

## Run it

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Then enter a ticker (e.g. AAPL) and click **Load company**.

## Files

- `app.py` — the whole app
- `VALUATION_METHOD.md` — the research behind the methods: how to calibrate
  growth, margins, exit multiples and discount rates without lying to yourself
- `.streamlit/config.toml` — dark terminal theme

## Method in one paragraph

For each scenario and year: revenue compounds at your growth rate → net income
= revenue × margin (gliding to your target) → EPS = net income ÷ shrinking/
growing share count → target price = EPS × exit P/E → present value at your
discount rate. Expected fair value = Σ probability × scenario PV. See the
Methodology tab in the app for the full write-up (base rates, PEG discipline,
margin mean reversion, and the model's deliberate limitations).

Research tooling, not investment advice.
