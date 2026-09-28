# Valuation method — research notes

How to use the available data to get honest valuations out of a
scenario model. This is the thinking built into Valuation Lab.

## 1. The core engine (and why EPS × exit multiple)

The model projects **revenue → net income → EPS → price (EPS × P/E)** per
scenario, then discounts to present value. This is the earnings-power family
of valuation: transparent, and every assumption is visible and arguable.

A full DCF was deliberately *not* the primary engine. In a standard DCF the
terminal value routinely drives 70%+ of the answer, and the terminal growth
rate is the least knowable input in the model — precision theater. Here the
exit P/E carries the "what is this worth in year 5" weight, and it's an
explicit, debatable number sitting right in front of you.

## 2. Calibrating growth — base rates beat narratives

The single biggest error in retail models is extrapolating recent growth.
Empirical work on corporate performance (Mauboussin's research on corporate
longevity and the fade of high growth) shows:

- High revenue growth **mean-reverts** as companies scale; the larger the
  base, the harder each incremental point of growth.
- Sustained >15% revenue growth over 5+ years is rare outside genuine
  platform shifts.
- Analyst 5-year growth estimates skew optimistic on average.

Practical calibration, in order:
1. Start the **base case** at or below the company's own 5-yr revenue CAGR
   (the app shows this anchor). Acceleration needs a named catalyst.
2. Make the **bear case genuinely bad** — growth stalls, margins compress.
   If your bear case still shows +40% upside, your scenarios are theater.
3. Cross-check against the **reverse DCF**: the app computes the FCF growth
   rate the current price already implies. If your base growth is below it,
   your own numbers say the stock is overvalued — sit with that before
   overriding it with a generous exit multiple.

## 3. Calibrating margins — respect mean reversion

Net margins revert toward industry norms over time as competition arbitrages
excess returns (the "measuring the moat" literature). The app shows the 5-yr
margin range as an anchor.

- A year-5 target **above the historical max** needs a mechanism: pricing
  power, mix shift toward software/services, or operating leverage with a
  credible path — not "they'll figure it out."
- Margin expansion + revenue acceleration + multiple expansion in the same
  scenario is triple-counting optimism. Pick at most two.
- For cyclicals, use **mid-cycle** margins, not peak margins — valuing a
  peak-earnings year at a full multiple is the classic value trap.

## 4. Calibrating the exit multiple — PEG discipline

The exit P/E is where most price targets are secretly decided. Discipline it:

- **PEG check**: exit P/E ÷ expected earnings growth (%) ≈ 1 is fair,
  >1.5 rich, <0.7 cheap (Lynch). If your growth is 8% and your exit P/E is
  30, you're paying 3.75× growth — write down why.
- **Own history**: a stock's 5–10 yr P/E range is the best anchor for where
  multiples can go. Multiples are cyclical too.
- **Market context**: with the S&P 500 Shiller CAPE above 40 (Sep 2026),
  assuming exit multiples near historical medians is already a *cautious*
  stance for large caps — don't reflexively project today's multiple
  forward forever.
- Growth stocks deserve higher multiples *only while the growth lasts*;
  the exit multiple should reflect year-5 maturity, not year-1 excitement.

## 5. The discount rate — your hurdle, not theirs

10% is the standard equity hurdle rate (long-run equity returns ≈ 10%
nominal). Adjust up for leverage, cyclicality, binary outcomes, or weak
governance — this single input is where risk belongs in the model. Note the
model discounts the *price target*, which is equivalent to requiring your
hurdle on the whole investment, dividends aside.

## 6. Shares outstanding — the forgotten lever

A steady -2%/yr buyback adds roughly 2 points to annual EPS growth without
any business improvement. Check the 10-K: has the share count actually been
falling, or is buyback cash offset by stock comp issuance? For heavy
issuers (many small caps), model positive share change — dilution is a
silent tax the screenshot-style tools ignore entirely.

## 7. FCF as the lie detector

Net income can be managed; free cash flow is harder to fake. The model shows
FCF/share next to EPS every year. If projected earnings soar while FCF
margins collapse, the earnings are low quality — revisit the margin and
capex assumptions. (The app uses a FCF-margin input defaulting to the 5-yr
average; for asset-light businesses it tracks net margin closely.)

## 8. Probability weighting — keep it honest

25/50/25 (bull/base/bear) is a disciplined default. Two rules:

- The probabilities must sum to a real distribution — if you catch yourself
  at 10/80/10, you haven't done scenario analysis, you've done a point
  estimate with decorations.
- Read the **bull/bear PV spread** as the uncertainty gauge. A tight spread
  with a big expected return is a genuinely attractive setup; a wide spread
  means the answer is "it depends" — size accordingly or pass.

## 9. Known limitations (stated plainly)

- Constant annual growth per scenario — real growth fades; the base-rate
  guidance compensates, but the model won't fade it for you.
- No balance-sheet distress modeling — for leveraged companies, equity can
  go to zero while the "earnings power" math looks fine. Check leverage
  separately.
- Exit multiples are exogenous — the model doesn't predict multiple
  compression; you do, via the bear case.
- Garbage in, gospel out — the sensitivity grid exists so you can see how
  much of the answer depends on everything going right.

## Sources

- Damodaran, *Investment Valuation* — DCF and relative-valuation frameworks
- Mauboussin, *Expectations Investing* — reverse DCF / priced-in expectations
- Mauboussin & Callahan — corporate longevity, growth fade, base rates
- Lynch, *One Up on Wall Street* — PEG heuristic
- Free data: Yahoo Finance (price, shares, financials, cash flows)
