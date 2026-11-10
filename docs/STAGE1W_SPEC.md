# Stage 1W: the Stage 1 calendar-time test re-run on the widened flag (registered 2026-11-09, before any Stage 1W result was computed)

**Status:** REGISTERED, **exploratory by construction.** The widened flag was adopted *after* the pre-registered strict flag (42 firms: guided within 180 days, sold after the latest guidance, abrupt exit) returned a null in Stage 1. What I had seen before writing this: the strict-flag Stage 1 result and its eight variants; the Stage 2 wider-definition results (all eight alphas between +0.2% and +2.8% a year); the Stage 15 large-loss results for this exact flag (14.2% against 17.6%); and Stages 11 and 12. **What I had not seen:** the calendar-time Carhart alpha for the widened flag. The supplement therefore presents this as the lead test but states that the flag definition was chosen after the strict version was null.

## 1. Why

The strict flag needs three conditions at once (guidance in the 180 days before the announcement, a sale after the *latest* guidance, and an abrupt exit judged from announcement text). That gives 42 firms and little power, excludes the motivating case (Fiserv's CEO sold before the latest guidance), and depends on text labels that are noisy (the program's precision against hand labels is 59%). The widened flag keeps the hypothesis (guide, sell, leave before the guidance comes due) but needs no text classification and has about ten times the sample.

## 2. The flag (fixed now; the Stage 15 definition, `gg_lossrate.flags`)

A CEO or CFO departure announced 2004 to 2025 (announcement date identified, not a death) is flagged when **(i)** the firm issued at least one EPS guidance release in the 365 days before the announcement, **(ii)** the executive made an open-market sale after the first of those releases (Form 4, name matched), and **(iii)** at least one of those releases covers a period that ends after the announcement. No abruptness requirement. Comparison (placebo) pool: eligible departures that are not flagged.

## 3. The test (identical to Stage 1 except the flag)

Hold each flagged firm from day +2 to +253 after the announcement (delisting returns applied); each day form the equal-weighted portfolio of firms held; regress its excess return on the market, size, value and momentum factors (weighted least squares, weights = number of firms held, Newey-West 10 lags); the intercept is the annualised alpha. **Primary:** one-sided (alpha below zero) placebo p-value from 2,000 random groups of non-flagged eligible departures with the same number per announcement year, seed 7. Windows with fewer than 20 observed days are dropped (Stage 15 used the same minimum).

## 4. Variants (eight, fixed now; Holm-adjusted together; 1,000 placebo draws each)

V1 CEOs only; V2 six-month hold; V3 six-factor model (adds profitability and investment); V4 excluding 2008 to 2009 and 2020; V5 announcements 2004 to 2014; V6 announcements 2015 to 2025; V7 the latest guidance within 180 days (instead of 365); V8 sale after the *latest* guidance release (instead of the first).

## 5. Interpretation rules (fixed now)

- **Supported** only if the primary alpha is negative and the placebo p is below 0.05. Otherwise not supported, reported with the placebo 5th and 95th percentiles.
- Because the flag was chosen after the strict-flag null, even a supported result would be exploratory and would need new data.
- Reported whatever the sign. Also reported: the screening funnel (counts at each condition), and whether the Fiserv departure (announced 2025-05-07) is in the flagged sample and has a usable window.
- Known limits fixed now: the flag windows use the first-identified announcement dates in `events.csv` (as in Stage 15, no leak guard); the Stage 1 caveat that the portfolio holds many firms at once, so overlapping windows make firm-level inference rest on the placebo; windows of firms announced late in 2025 are partial.

## 5b. Amendment 1 (2026-11-09; after the 2004-2024 run below, before any extended-data result)

**Why.** The first run used the legacy CRSP daily file, which stops on 2024-12-31 on this account, so every 2025 announcement (including the Fiserv departure that motivates the project) had no return window. The CRSP v2 daily file (`gg_cache/ciz_dsf_2023_2025.pkl`, already pulled for Stage 9, through 2025-12-31, delisting returns included) removes that limit.

**Change (only the data).** For dates after 2024-12-31 the legacy returns are extended with the v2 returns (the project's `gg_crsp_ciz` conversion; on the 2023 to 2024 overlap the two files agree on 98% of 464,825 stock-days); the factor file already runs through 2025. Flag, windows (day +2 to +253, at least 20 observed days), model, placebo (2,000 draws, seed 7) and the eight variants are unchanged, so 2025 announcements enter with partial windows (at most about 160 days). **The extended run replaces the 2004-2024 run as the primary test** because it uses the same definition on all the data, and the 2004-2024 result stays reported (Section 6) so nothing is hidden. I knew the 2004-2024 result (alpha +2.7%) when I made this change.

## 6. Results

Run 2026-11-09 (`python scripts/run_stage1w.py --part primary` and `--part variants`; `outputs/results/stage1w_primary.json`, `stage1w_variants.json`).

**Verdict: not supported. The widened-flag firms do not underperform; their alpha is positive.**

**Funnel.** 5,163 departures; 3,650 with an identified announcement and not a death; 1,855 guided in the prior 365 days; 660 sold after the first of those releases; **501 flagged** (a guided period ends after the exit); **480 with a usable return window** (477 CEOs, 3 CFOs; 131 in technology).

**Primary.** Four-factor alpha **+2.7% a year** (t = +1.75, Newey-West two-sided p = 0.08), 5,281 trading days, 22 firms held on average (loadings: market 0.94, size 0.45, value 0.03, momentum -0.04). One-sided placebo p (alpha below zero) **0.81**: 81% of 2,000 random groups of non-flagged departures did as badly or worse (placebo mean +0.8%, sd 2.1%, 5th to 95th percentile -2.7% to +4.3%). With a placebo sd of 2.1%, an underperformance of about 5% a year would be detected with 80% power.

**Variants (1,000 draws each; Holm-adjusted p = 1.0 for all).** CEOs only +3.0% (n 477); six-month hold +1.8%; six-factor model +2.0%; excluding 2008-09 and 2020 +2.7% (n 427); 2004-2014 +2.8% (n 266); 2015-2025 +2.7% (n 214); guidance within 180 days +3.0% (n 469); sale after the latest release +0.1% (n 184). No variant is negative; none is significant.

**The Fiserv check.** The Bisignano departure (announced 2025-05-07) is in the eligible sample and **is flagged** (five releases in the prior year, a $25.7M sale on 2024-09-20 after the first release of 2024-07-24, and three guided periods ending after the exit), but it has **no return window**: the CRSP cache ends on 2024-12-31, so every 2025 announcement is outside the return tests (flagged: 501; with returns: 480). Extending the return data through 2025 would add Fiserv's own post-announcement return and the other 2025 events.

**Limits.** The flag was chosen after the strict-flag null, so this is exploratory; windows use the first-identified announcement dates (no leak guard, as in Stage 15); the portfolio holds about 22 firms at a time, so the placebo (not the t-statistic) is the inference; no 2025 events.

### 6b. Extended-data results (Amendment 1; returns through 2025-12-31; `outputs/results/stage1w_ext_primary.json`, `stage1w_ext_variants.json`). These are the primary results.

**Verdict: not supported, as before.** 501 flagged; **492 with return data** (489 CEOs, 3 CFOs; 135 in technology). Four-factor alpha **+2.6% a year** (t = +1.65, Newey-West two-sided p = 0.10), 5,531 trading days, 22 firms held on average (loadings: market 0.94, size 0.46, value 0.02, momentum -0.05). One-sided placebo p **0.80** (placebo mean +0.85%, sd 2.1%, 5th to 95th percentile -2.6% to +4.3%); an underperformance of about 5% a year would be detected with 80% power. Variants (Holm-adjusted p = 1.0 for all): CEOs only +2.9%; six-month hold +1.1%; six-factor +1.9%; excluding 2008-09 and 2020 +2.6%; 2004-2014 +2.8% (n 266); 2015-2025 +2.2% (n 226); guidance within 180 days +2.8%; sale after the latest release +0.1% (n 188).

**The Fiserv departure is now in the return tests** (flagged, with a partial window of 168 trading days ending 2025-12-31). Nine flagged events still have no return data. **Latest data:** the CRSP v2 daily file on this account ends on 2025-12-31 (checked against WRDS for four large permnos on 2026-11-09: no 2026 rows), so 2025 announcements have partial windows and nothing later can be added from CRSP.
