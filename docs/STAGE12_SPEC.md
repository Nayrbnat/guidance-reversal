# Stage 12: does a model trained on the past predict the future? A rolling-window out-of-sample test (registered 2026-11-08, before any Stage 12 outcome was computed)

**Numbering.** The held-out 2025 opacity confirmation is Stage 9, the large-loss-rate test is Stage 15 (earlier numbered 10) and the low-power methods are Stage 11 (`STAGE11_SPEC.md`); this is Stage 12.

**Status:** REGISTERED. What I looked at before writing it: how often the first-identified announcement date differs from the corrected one (98.4% equal; for 6% of flagged events the sale and guidance windows can extend past the corrected date), and the number of eligible events per year (140 to 212). No Stage 12 prediction, AUC or gap was computed. Exploratory: it reuses events from Stages 1 to 11.

## 1. What this tests

A relationship that is real should survive being **estimated on the past and tested on the future**; noise will not. Stages 1 to 11 test hypotheses on the full sample. This stage asks a different question, with a machine-learning model: **do the Guidance Reversal ingredients (guided, sold, optimism, left before the guidance came due, abrupt exit) carry information about large losses that generic firm characteristics do not, out of sample?**

## 2. Sample (fixed now)

Eligible departures (announcement identified, not death) announced 2004 to 2024, with a complete 12-month window (at least 200 observed days from day +2) so that every label is a full-year outcome. **Leak guard:** events where the first-identified announcement date is more than one day later than the corrected date are excluded, because their sale and guidance features can include information after the corrected announcement (2% of events, 6% of flagged ones).

## 3. Outcome

**Primary:** large loss = 12-month market-adjusted buy-and-hold return (days +2 to +253) below -30 points, exactly as in the large-loss test. **Secondary:** the return itself, winsorised at 1% and 99% inside each training window.

## 4. Features (all measured before day +2 of the return window)

- **B, baseline (generic):** log market cap and 12-month return and volatility over the 252 trading days ending two trading days before the corrected announcement, CEO or CFO, SIC sector group.
- **R, the Guidance Reversal ingredients:** number of guidance releases in 12 months, days since the latest, number whose period ends after the exit, the latest optimism and the maximum optimism against consensus, whether the latest guidance was a raise, the value sold after guidance and over 12 months as a share of market cap, number of sales, opportunistic-sale value, the rule-based abrupt flag, the exit type from the rule classifier, and the flag itself.
- **O, opacity (secondary only):** the transparency score of the last 10-K filed before the announcement.
- No feature uses the announcement-window return, the outcome, or any date after day +1.

## 5. Models (fixed now)

- **ML model (primary): gradient boosting**, scikit-learn `HistGradientBoostingClassifier`, depth 3, learning rate 0.05, 100 iterations, at least 40 observations per leaf, L2 regularisation 1.0, seed 0, no tuning.
- **Benchmark: L2-regularised logistic regression** on standardised features, C = 1.
- No hyperparameter is chosen on test data.

## 6. Rolling window (fixed now)

For each test year t = 2010 to 2024: **train on announcement years t-6 to t-2** (five years) and **test on year t**. Year t-1 is left out as an **embargo** so that every training label (a 12-month window) is complete before the test year starts. Variant: an expanding window from 2004 to t-2. Predictions are made once per event and never refit on test data.

## 7. Tests

**P12a (the one confirmatory test).** For the gradient-boosting model, the out-of-sample **mean within-year AUC** of B+R minus that of B alone (years with at least 5 large losses), pooled over the 15 test years. **H: positive**, one-sided. Inference: 200 permutations that shuffle the R rows across events within each announcement year (so R is broken from the outcome and from B, and the whole rolling procedure is re-run), p = (1 + number at least as large) / 201. Reported with: the AUC of B, of B+R, the difference by test year, a 95% interval from a bootstrap over announcement quarters (predictions fixed), and the logistic benchmark (1,000 permutations).

**Secondary (outside the count, none can rescue a failed P12a):** the flag alone as a score; B+R+O on the events with a score; the return as the outcome (out-of-sample R-squared against the training mean); the expanding window; the logistic model.

**P12b (descriptive).** Rolling five-year windows of announcement years (2004-08, 2005-09, ..., 2020-24): in each, the large-loss share of flagged firms minus that of guided firms with outstanding guidance and no sale, the 3-day reaction gap to the first post-exit guidance, and the mean 12-month return gap, each with a bootstrap standard error over quarters. Reported in full, no window selected. A heterogeneity test (Cochran's Q) over the four non-overlapping blocks 2004-08, 2009-13, 2014-18, 2019-24.

**Count.** P12a adds one formal test. With Stage 9 (held-out 2025), Stage 15 (large-loss) and Stage 11 (low-power methods) the project-wide count becomes **41**.

## 8. Interpretation rules (fixed now)

- **Supported:** a positive difference in AUC with permutation p < 0.05 **and** a positive difference for the logistic benchmark.
- **Not supported:** anything else, reported with the interval. An AUC difference below about 0.02 is not economically meaningful even if significant.
- A positive result here would be exploratory: every year was seen in earlier stages.
- Known limits fixed now: about 700 training events and about 25 large losses per test year among flagged firms, so the model has little to learn from; the first-identified announcement date is used for the features, which the leak guard handles only for later dates; the return is market-adjusted, not risk-adjusted; the sector grouping is coarse.

## 9. Results (run 2026-11-08, `python scripts/run_stage12.py`, 200 permutations for gradient boosting and 1,000 for logistic, seed 0; `outputs/results/stage12_results.json`)

Two bugs in the runner script (a column that was never built, and a duplicated column name) stopped the first two attempts before any model was fitted; they were fixed and nothing in Sections 1 to 8 was changed.

**Sample.** 3,334 eligible events with a window, 3,053 after the year, window-length and leak-guard rules (the guard dropped 40, 30 of them flagged), 3,010 with complete baseline features; 430 flagged; large-loss share 21.0%. 15 test years (2010 to 2024), 2,096 out-of-sample predictions.

**Verdict: P12a is not supported. Adding the Guidance Reversal ingredients does not improve out-of-sample prediction; it makes it slightly worse. Nothing in the rolling windows shows a stable relationship.**

| | AUC, baseline only | AUC, baseline + reversal ingredients | Difference | Permutation p |
|---|---|---|---|---|
| **Gradient boosting (primary)** | 0.620 | 0.602 | **-0.018** (95% interval -0.040 to +0.010) | **0.945** |
| Logistic (benchmark) | 0.619 | 0.602 | -0.017 (-0.034 to +0.004) | 0.438 |

The baseline (size, return, volatility, role, sector) reaches AUC 0.62, so generic characteristics do predict large losses. For the logistic model the shuffled features also lower AUC (mean change -0.018), so the real ingredients do no better than noise; for gradient boosting the real ingredients do worse than shuffled ones (null mean -0.006), the signature of a pattern learned in the training years that does not hold in the test years. The test is sensitive: the 95th percentile of the gradient-boosting null is +0.0075, so a gain of about 0.01 AUC would have been detected. By test year the difference is positive in 5 years and negative in 10 (range -0.21 in 2021 to +0.06 in 2016).

**Secondary (outside the count; none can rescue the primary).** The flag alone as a score: mean within-year AUC **0.480** over 15 years (below 0.5: flagged firms are slightly less likely to lose heavily). Expanding window: logistic -0.016 (p 0.90), gradient boosting +0.009 (p 0.12, 100 permutations), the only positive difference and not significant. Adding the transparency score to the reversal ingredients (2,923 events with a score): -0.002 (p 0.66) logistic, -0.002 (p 0.75) boosting. The return as the outcome: out-of-sample R-squared is negative for the baseline (-0.03 ridge, -0.08 boosting), so no model here beats the training-window mean at predicting 12-month returns, and adding the ingredients lowers it further (-0.06, -0.09).

**P12b, rolling five-year windows (17 windows, 2004-08 to 2020-24).** Large-loss share, flagged minus comparator: positive (the hypothesised direction) in **5 of 17** windows; by non-overlapping block -3.3 (SE 4.5), +4.5 (3.8), **-7.9 (3.2)**, -5.4 (5.7) points; Cochran's Q p = 0.09. First-guidance reaction gap (hypothesised sign negative): 8 of 17 windows; blocks +1.0, +1.2, -0.6, -0.5 points (SEs 0.8 to 1.1); Q p = 0.32. 12-month return gap: 9 of 17; blocks -2.3, -5.8, +5.7, +3.1 (SEs 4 to 7); Q p = 0.41. No window is selected; none shows an effect that persists.

**Limits.** About 700 training events a window and about 100 flagged events a window (430 over 21 announcement years), so the model has little to learn from; depth-3 boosting cannot be ruled out as too simple to find an interaction, although the larger expanding window does no better (+0.009, not significant); the first-identified announcement date is used for the features; the return is market-adjusted; the leak guard removes 30 flagged events. The project-wide count of formal tests becomes 41 counting Stages 9 to 11.

## Amendment 1 (2026-11-09): returns through 2025-12-31

**Data only.** The legacy CRSP daily file and the market index in the main cache stop on 2024-12-31 on this account (WRDS has no 2026 CRSP rows). The CRSP v2 daily file (`gg_cache/ciz_dsf_2023_2025.pkl`, through 2025-12-31) is appended for later dates (`src/gg_extend.py`); the market return after 2024 is the Fama-French market (mktrf + rf), which over 2022 to 2024 correlates 0.998 with the cached CRSP index and runs about 1.4% a year higher. Nothing else changes: the sample rule (announced 2004 to 2024, at least 200 observed days from day +2, leak guard), features, models, windows and test years (2010 to 2024). The longer returns complete the 12-month windows of late-2024 announcements, which the old data truncated and the 200-day rule dropped, so the sample grows slightly. Only P12a is rerun (`python scripts/run_stage12.py --extend --primary-only`; `stage12_results_ext.json`). The original (2004 to 2024 returns) results stay in this document. I knew those results when I made this change; the change is made because the 2025 returns exist, not because of any result.
