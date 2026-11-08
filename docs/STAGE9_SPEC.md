# Stage 9: a held-out confirmation of the Stage 8 opacity-mechanism test, on 2025 insider sales

Status: **written 2026-11-07, before any 2025 return, avoided loss or regression was computed.** Author: Nayrb (requested by the author on 2026-11-07: use the latest period as a test set, earlier years as training).

What I looked at before writing it, none of it an outcome: counts of 2025 insider-days and of how many have a transparency score; the CUSIP-to-permno link rate by year (which exposed and fixed a name-table bug); an overlap check of the new CRSP v2 returns against the old file on 2023 to 2024 (98.2% identical at six decimals, largest gap 0.0015); the end dates of the new price and factor files (CRSP 2025-12-31, factors 2026-09-30). **No 2025 return has been used in any calculation.** The 2025 prices are on disk but untouched.

## 1. What this tests

Stage 8 found no sign that the avoided loss after insider sales depends on transparency (primary Z = -0.129, standard error 0.152, one-sided p 0.199, 2004 to 2024). Every year in that sample has been seen. Stage 9 repeats the **same single test, unchanged**, on sale dates not seen: 2025-01-01 to 2025-09-30 (the last date that still has 63 trading days of CRSP prices before 2025-12-31).

It is a confirmation, not a search. The primary specification is Stage 8's, fixed in `docs/STAGE8_SPEC.md` Section 4, with one change forced by the sample (Section 4 below).

## 2. Sample (fixed now)

- Officers and directors by role code, open-market sales (code S), summed per person, issuer and day, value at least $100,000, sale dates 2025-01-01 to 2025-09-30, exactly as Stage 8 (`gg_stage8.insider_sale_days`), linked to permno with the CRSP name table valid on the sale date (v1 and v2 name rows merged, the v2 row winning) and to gvkey with the primary CRSP-Compustat link.
- Returns: CRSP daily v2 (CIZ) for 2025, spliced to the v1 file for earlier dates (the 252-day beta window for a January 2025 sale reaches into 2024). Factors: the Fama-French five factors and momentum, same file as before, extended.
- $T$: the last 10-K filed strictly before the sale, within 550 days, scored with the **frozen** Stage 5 configuration (hash `72fb2dc9d4c9…`, rubric r3, extraction ex8). $Z=(T-\mu)/\sigma$ with $\mu,\sigma$ from the **Stage 5 scores as they were**, so Z keeps its scale. Newly scored filings (the 2025 10-Ks and the gap filings for firms whose chains stopped earlier) were added without changing any previously existing score or phase (checked in code).
- Sale-days with no $T$ are dropped; the share lost and the reasons are reported.

## 3. Outcome and variables

Identical to Stage 8: $Y_s=-100\times CAR_s(+1,+63)$, four-factor, betas from the 252 trading days before the sale (at least 120 observations), winsorised at 1% and 99% (the cut-offs are computed on the 2025 sample); controls log market cap, 12-month return, log sale value, pre-sale volatility $V$ (standardised on the 2025 sample) and role indicators.

## 4. The test (one test)

$$ Y_s = \lambda_{j(i)} + \tau_q + \theta\,Z_s + \gamma' W_s + u_s , $$

SIC two-digit industry effects, year-quarter effects (three quarters). **One change from Stage 8:** standard errors **clustered by firm only**, because clustering by quarter with three quarters is not reliable. **H5-M: $\theta<0$**, one-sided, $p<0.05$. This is the only confirmatory test; the project-wide count of formal tests becomes 37.

**Power, printed before the estimate:** the standard error of $\theta$, the minimum detectable effect (about 2.49 times the standard error) and a firm-cluster bootstrap plant $-\delta Z_s$ into $Y_s$ at several $\delta$. My rough expectation, from scaling Stage 8, is a minimum detectable effect near 0.8 points per standard deviation of T, perhaps worse.

## 5. Interpretation rules (fixed now)

- **Supported:** $\theta<0$ with one-sided $p<0.05$.
- **Not supported:** anything else, reported with the minimum detectable effect. Because the design cannot detect effects under about 0.8, a non-rejection does **not** rule out the Stage 8 estimate of -0.13 or anything of that size. What it can do is say whether the 2025 point estimate and its 90% interval are consistent with Stage 8 (interval contains -0.13) or not.
- A significant $\theta$ with a magnitude far above Stage 8's (for example below -1) would be reported as such and treated as a sign of a small-sample overestimate, not as a good estimate.
- A pooled estimate (Stage 8 plus Stage 9) is reported as a descriptive summary only.

## 6. Secondary and exploratory (outside the count; none can rescue a failed primary)

1. **Other officers only, and large firms only** (above the median market cap of the 2025 sample): the two Stage 8 near-misses (one-sided p 0.052 and 0.055). They are the only pre-named subgroups; no other subgroup will be run on 2025.
2. **Guidance interaction** $Z\times G$ as in Stage 8 S3.
3. **Walk-forward exploration on 2004 to 2024 (exploratory; the years have been seen):** for each test year 2009 to 2024, estimate the primary regression on the five preceding years and on the test year, and report $\hat\theta$ by test year, plus a selection exercise: pick on each five-year training window whichever of {all, other officers, large firms, opportunistic only} has the most negative t-statistic, and report its out-of-window estimate. This shows how stable $\theta$ is over time and whether selecting on training years would have worked; it does not feed the primary test and carries no weight as confirmation because every year was seen in Stage 8.

## 7. Known limits fixed now

Three quarters of one year; one market regime; the sample leans toward firms with a scored 10-K (about 63% of linked insider-days had a score before the additions); scheduled (10b5-1) and tax sales cannot be separated; the effect is identified from differences between firms; and the leave condition is still dropped, so a null here, like Stage 8, says nothing direct about guide-sell-leave. If the result is null, the honest summary is that this project could not detect an opacity effect on insider sales and could not rule out a small one.

## 8. Results (added after the run; Sections 1 to 7 above were not changed)

Run once on 2026-11-07 with `scripts/run_stage9.py`. Output: `outputs/results/stage9_results.json`.

- **Sample.** 6,921 linked insider-days (2025-01 to 2025-09, at least $100,000); 6,690 with an avoided loss and a transparency score, from 839 firms and 3,283 people, three quarters. Missing T: 110 no prior filing, 78 not scored, 39 extraction failed. 64% of sale-days follow guidance.
- **Power (printed before the estimate).** Standard error of Z 0.730; minimum detectable effect **1.82** pp per sd of T (the spec's rough pre-run guess of 0.8 was too optimistic: I scaled by firm count and under-counted the cost of firm clustering). Simulated rejection at planted 0, 0.91, 1.82, 3.63: 7%, 50%, 93%, 100% (30 replications).
- **Primary.** Z = -0.202 (se 0.730), one-sided p 0.391, two-sided 0.782, 90% interval [-1.40, +1.00]. **Not supported.** The Stage 8 estimate (-0.129) lies inside the interval.
- **Secondary (outside the count, per Section 6).** Other officers +0.029 (se 0.815, p 0.51); large firms -2.003 (se 1.068, one-sided p 0.031); Z x G -2.264 (se 1.212, one-sided p 0.031), with Z alone +1.171 in that column. Two of three reach one-sided 5%; under Section 6 none can rescue the primary, and at this power a significant estimate is likely inflated.
- **Walk-forward (exploratory, years already seen).** Coefficient on Z negative in 10 of 16 test years 2009 to 2024, mean -0.033; selecting the subset on the previous five years gives a mean out-of-window estimate of -0.11 with the chosen subset changing year to year.
- **Reading under Section 5.** Not supported, minimum detectable effect 1.82; consistent with Stage 8's -0.13 but cannot confirm or refute it. The project-wide count of formal tests is 37.
