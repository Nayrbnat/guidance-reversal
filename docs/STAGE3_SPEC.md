# Stage 3 spec (APPROVED by the project owner 2026-10-24; registered in PREREGISTRATION.md before any outcome was computed)

Decisions confirmed by the project owner: (1) **CEO only** for the primary tests; (2) **any open-market sale** counts (no size threshold); (3) optimism means **guidance above consensus** (binary), as in Stage 2. The standardised continuous version is robustness R1.
So far I have looked only at **counts** (episodes, coverage, sample sizes in the feasibility table at the bottom), never at misses, returns or hazard rates by group.

## 0. Why Stage 3

Stages 1-2 compare leavers with other leavers using 12-month alpha. They cannot say whether *leaving* adds anything, and returns are a noisy proxy for the actual claim:
**"optimistic guidance, the executive sells, leaves, and the company misses."** Stage 3 adds stayers as a control group and tests the miss directly.

## 1. Three questions and the one statistic that answers each

| Test | Question | Unit | Statistic (one-sided H1) |
|---|---|---|---|
| **P1 hazard** | Is a CEO more likely to leave in a quarter when the latest guidance was optimistic **and** they sold after it? | CEO x quarter, guiders only | coefficient on OPT x SALEPOST > 0 |
| **P2 miss** | Is the realised miss worse when guidance was optimistic, the CEO sold after it, **and** left before the period ended? | one guidance release | coefficient on OPT x SALE x EXIT < 0 |
| **P3 leavers vs stayers** | Do firms where the CEO guided, sold and **left** underperform firms with the same pattern where the CEO **stayed**? | calendar-time portfolios | alpha(leavers) - alpha(stayers) < 0 |

Holm correction over these three. Everything else is robustness, reported but not in the family.

## 2. Definitions

- **CEO episode** (executive, firm): from `becameceo` (or 1 Jan of their first CEO-flagged ExecuComp year) to `leftofc` (else 31 Dec of their last CEO-flagged year, right-censored). If two overlap on a date, the later start is CEO of record.
- **Exit (E1, primary):** `leftofc` falls in the window; ExecuComp reason DECEASED dropped. Disappearing without `leftofc` = **censored**, not an exit.
- **OPT (primary):** 1 if the guided EPS midpoint is strictly above consensus at the guidance date, else 0. (R1 uses the continuous gap, (midpoint - consensus)/price, standardised to sd 1 after winsorising 1/99.)
- **Miss:** (actual - guided midpoint) / price, in % of price points, winsorised 1/99; gaps over 100% of price dropped. (Robustness: actual minus consensus.)
- **Sale:** Form 4 open-market sale by an owner matching the CEO on last name and full first name/nickname (score 3), value > 0. Always timed after the guidance, and before the exit if there is one.

### P1 model (linear probability, percentage points per quarter)
`exit(q) = firm FE + year FE + c1 OPT + c2 SALEPOST + c3 OPT*SALEPOST + SALE12 + log mcap + 12m return + log tenure + age`, all measured at the **start** of quarter q. Sample: CEO-quarters with guidance in the previous 365 days. SEs clustered by firm.

### P2 model
`miss = firm FE + year FE + b1 OPT + b2 SALE + b3 EXIT + b4 OPT*SALE + b5 OPT*EXIT + b6 SALE*EXIT + b7 OPT*SALE*EXIT + log mcap + 12m return + log days to period end + quarterly dummy + log tenure + age`.
SALE = sale in (guidance date, min(period end, exit)]; EXIT = exit in (guidance date, period end]. Executive of record = CEO. SEs clustered by firm. **Only b7 is the claim.**
Why the triple interaction: b1 is negative almost mechanically (guide above the street, land near consensus = "miss"); b3 and b5 soak up a new CEO's "big bath" write-offs.

### P3 design
Pattern L1 = guidance within 180 days before date d and a CEO sale between that guidance and d. Leavers = Stage-2 L1 x U2 events at the announcement date. Stayers = each quarter start at which a CEO satisfies L1, does not exit within 365 days and is observed for 365 days afterwards, thinned to one pseudo-event per executive per 4 quarters. Same 12-month Carhart calendar-time alpha as before; labels permuted within year, 5,000 draws.

## 3. Assumptions I am making, and what breaks if they are wrong

| # | Assumption | If wrong |
|---|---|---|
| A1 | `leftofc` = date they left the CEO office, including moves to chair/advisor | Role changes counted as exits dilute any "left under a cloud" effect |
| A2 | Executives who vanish from ExecuComp without `leftofc` have not left (censored) | If many are real exits, hazards are too low and "stayers" include leavers (biases P3 toward 0) |
| A3 | CFO exits cannot use `leftofc` (only 6% of CFO episodes have it vs 68% for CEOs), so **Stage 3 analyses CEOs only**; CFOs are future work | Nothing is said about CFOs, who are central to the guidance story |
| A4 | **Optimism vs consensus proxies intent** | We test a pattern, not intent; the street may simply have been stale or the CEO confident |
| A5 | I/B/E/S guidance and actuals are comparable after split rebasing (guidance often non-GAAP) | Misses carry measurement error: costs power, can bias if non-GAAP guiders also sell more (checks: R4, R5, the 100%-of-price cut) |
| A6 | Strict name matching finds the CEO's own open-market sales; 10b5-1 status, gifts, hedges, pledges unobserved | Sales undercounted (attenuation); planned and discretionary sales mixed |
| A7 | SALE and EXIT are defined **after** the guidance (post-treatment), so P2 is predictive, not causal. Window lengths differ (annual vs quarterly guidance) | Longer windows mechanically hold more sales/exits; controls for log days and quarterly dummy reduce, not remove |
| A8 | A new CEO may take a big bath regardless of intent | Handled by net-of-b3/b5 design, falsification R11 and P3; not fully removable |
| A9 | Stayer pseudo-date (quarter start) is comparable to leaver date (announcement) | Different guidance-to-date gaps could bias P3; I report days-since-guidance for both groups |
| A10 | Firm fixed effects absorb stable firm guidance style, so identification is within firm over time | Firms with one CEO throughout contribute nothing to EXIT effects (effective n reported) |
| A11 | Firm-clustered SEs with the usual G/(G-1)*(N-1)/(N-K) correction (firm effects nested in clusters) and t(G-1) | Cross-sectional correlation by calendar time would make SEs too small (R10 year-quarter FE) |
| A12 | Population = S&P 1500 guiders (stayers are guiders too) | Nothing said about non-guiders, micro-caps, non-U.S. |
| A13 | Firms that fail or are acquired often lack a final actual and drop out of P2 | Works against finding misses; failure is covered in P3 via delisting returns |

## 4. Rules

- **Minimum sizes:** P1 needs >= 100 exits; P2 needs >= 100 releases with SALE=1 and EXIT=1; P3 needs >= 40 leavers and >= 40 stayers with windows. Otherwise reported as underpowered, outside the family.
- I report the **minimum detectable coefficient** (about 2.5 x the standard error) for P1 and P2, so a null is read correctly.
- A finding is claimed only if Holm-adjusted p < 0.05. For P2 the sign must also survive R1, R4, R5, and R11 must be null; otherwise it is called fragile.
- Everything is reported regardless of outcome. (Checks R3, R7 and R8, which needed holdings-scaled sales and CFO exits, were dropped before any outcome was computed to keep the set small.)

## 5. Robustness (not in the family)

R1 continuous standardised optimism in place of the binary flag · R2 sale >= $1M · R4 outcome = actual minus consensus · R5 first release per (firm, period) only ·
R6 logit with year FE and no firm FE for P1 · R9 exits excluding reason RETIRED · R10 year-quarter FE in P2 ·
**R11 falsification: sales in the 180 days *before* the guidance instead of after (should be null if the story is guidance-contingent selling)**.

## 6. Disclosures

I had already seen all of Stage 1-2, including that the guide-sell-leave group's *post-exit guidance misses were not worse* than other departures' (an exploratory result). P2 is a regression version of that, so it is **not blind**. Project-wide formal tests would be 28 + 3 = 31.

## 7. Feasible sample sizes (counts only, from `gg_panel_data.py feasibility`)

Built from 7,950 CEO episodes (3,315 firms; 5,432 with `leftofc`, 14 deceased), 135,276 linked EPS guidance rows, 478,792 matched CEO sales over 4,154 episodes. A CEO of record was found for 91% of guidance releases.

| Panel | Sample | Treatment cells | Verdict |
|---|---|---|---|
| **A (P1)** | 77,414 CEO-quarters, 4,648 CEOs, 2,290 firms | **1,943 exits** (2.5% per quarter); SALEPOST=1 in 13,479 (17%); **SALEPOST and exit: 324**; optimistic (OPT>0) 42%; 1,172 firms show exit variation (hold 1,939 of the exits) | Feasible. Min-size rule (>=100 exits) met easily |
| **B (P2)** | 98,178 releases, 2,082 firms, 4,119 CEOs (63,372 annual, 34,806 quarterly) | SALE=1: 26,466 (27%); EXIT=1: 4,067 (4.1%); **SALE and EXIT: 1,006 releases from 359 firms**; of these OPT>0: 435; firms with exit variation: 930 | Feasible. Min-size rule (>=100) met. Effective sample is closer to the **359 firms** than to 1,006 (releases cluster within firm) |
| **C (P3)** | **157 leavers** (Stage 2 L1 x U2, with windows) vs **5,281 stayer pseudo-events** (1,938 CEOs, 1,380 firms, 172-349 per year) | n/a | Feasible. Min-size rule (>=40 each) met. Leaver side is the bottleneck, so the difference has the same noise as the Stage 2 L1 cell (placebo sd about 4%/yr) |

Placebo R11 (sale in the 180 days before the guidance): 29,579 releases have one, so the falsification test has plenty of cells.

What the counts imply:
- **P1 and P2 are far better powered than Stages 1-2** because they use all guiders and stayers, not only leavers. P2's triple-interaction cell (1,006 releases, 359 firms) is about 20x Stage 1's 45 events.
- **P3 is not better powered on the leaver side** (157). It answers "does leaving matter?" but cannot detect small differences.
- Stayer events are 34x the leavers, so P3's within-year permutation compares a small group with a large one. Treat its p-value as describing the leaver arm's noise.
- Only 435 of the 1,006 SALE and EXIT releases have OPT>0, so any optimism-specific effect rests on 435 releases from a few hundred firms.

## 8. Amendment 1 (2026-10-24, after an assumptions audit, BEFORE any Stage-3 outcome was computed)

The project owner asked for a recheck of the econometric assumptions (stationarity, time-series weaknesses, look-ahead). No outcome had been computed, so these changes are made to the registered design. They replace the conflicting text above.

**Changes to the primary design**
1. **Fixed effects and standard errors.** Time fixed effects are **year-quarter** (of the guidance date for P2, of the quarter for P1), not year. Standard errors are **two-way clustered by firm and year-quarter** (Cameron-Gelbach-Miller, eigenvalue-clipped), with t(min(G)-1). Firm-only clustering becomes robustness (R10 replaced). Reason: a simulation in `test_gg.py` (firms with different exposure to common time shocks, true effect zero, 40 quarters) shows firm-only clustering rejects **69%** of true nulls at the 5% level; two-way clustering rejects 10% (conservative when there are no shocks). Earnings misses and CEO turnover cluster in 2008-09 and 2020, so the shock is real. Even two-way is somewhat liberal with few time clusters, so a p-value just under 0.05 is not to be trusted.
2. **Look-ahead in measurement dates.** The share price used to scale optimism and misses, and the market cap and 12-month return controls, are taken at the close of the trading day **before** the guidance date (before the quarter start for P1), not on the release date, which already contains the market's reaction.
3. **Survivorship in the stayer control (P3).** The registered stayer rule required the executive's episode to be observed in ExecuComp for 365 more days. That conditions on the *firm* surviving (acquired, delisted or de-listed firms drop out) and applies only to stayers, so it inflates stayer returns and would bias P3 toward the hypothesis. **Replacement rule:** a stayer is a CEO with no `leftofc` in (d, d+365 days], with **no requirement to remain in ExecuComp**; returns come from CRSP including delisting returns, exactly as for leavers. Events after 2024-12-31 are excluded for both groups so every window can be 12 months.

**New assumptions (A14-A19)**

| # | Assumption / weakness | Handling |
|---|---|---|
| A14 | **Stationarity.** Outcomes (exit indicator, miss as % of price) are bounded and stationary; regressors are indicators or ratios. Log market cap and tenure are trending, which would matter for I(1) levels but not for a stationary dependent variable with year-quarter FE. Guidance practice has shifted (Reg FD, post-2010 decline, 2020 withdrawals), so slope **stability** is not assumed | Year-quarter FE absorb level shifts; **R12 repeats P1 and P2 for 2004-2014 and 2015-2025** and reports whether the key coefficient keeps its sign |
| A15 | **Cross-sectional dependence in time.** Misses, optimism and exits co-move with the business cycle and firms differ in exposure | Two-way clustering (above). Residual risk stated: only ~88 quarters |
| A16 | **Strict exogeneity with firm FE.** A CEO's exit ends the episode and the next CEO starts, so within-firm demeaned regressors are mechanically related to time since the last exit (Nickell-type bias, order 1/T, T about 30-88 quarters) | Log tenure is a control; **R6 (logit with no firm FE)** compares |
| A17 | **Announcement contamination.** `leftofc` is the *effective* date. A departure is often announced weeks earlier, and sales after the announcement but before the effective date are not guidance-contingent. Guidance released after the announcement is not written to enable the exit | **R13:** P1 sale window ends one quarter before q; P2 sale window ends 90 days before the exit. **R14:** P2 drops releases within 60 days before the exit |
| A18 | **Selection on a future outcome.** P2 keeps releases for which an actual was later reported. If exits cause coverage to stop, attrition correlates with EXIT | Diagnostic reported: share of releases with a missing actual by EXIT status, before dropping |
| A19 | **P3 comparability.** Leavers require an identified announcement (annotator or rule-based); stayers do not. Stayers are far more numerous (about 5,000 vs 157) and their portfolio holds many more positions, so the leaver arm drives the noise | Reported; days since guidance shown for both groups; no inference beyond the permutation p-value |

**Look-ahead audit (what is and is not look-ahead).** P1 covariates use only dates strictly before the quarter start; `SALEPOST` runs from the latest guidance to the quarter start. P2's SALE and EXIT windows begin after the guidance and are the *treatment* definitions, so P2 is predictive, not causal. Ex-post classification of leavers and stayers (by whether an exit follows) is the treatment definition, not look-ahead, as in any event study. ExecuComp role and tenure dates are known later but describe who held the office then. Winsorising and standardising use the whole sample, which affects scaling, not inference. The run asserts these timing rules in code (a "look-ahead audit" printed with the results).

Robustness list now: R1, R2, R4, R5, R6, R9, **R11, R12, R13, R14** (R10 dropped; firm-only clustering is reported alongside the primary).

## 9. Null calibration (2026-10-24, before the registered run; outcomes randomised, no real outcome seen)

`gg_stage3_calibrate.py` permuted the outcome (exit indicator, miss) within year-quarter on the real panels and re-ran P1 and P2 with the same random outcome for both standard-error types (60 draws).
- One-sided 5% rejection rate: **P1 6.7%, P2 6.7%** (two-way and firm-only identical); share with p < 0.10: 13% and 18% (nominal 10%). Uncertainty on these rates is about +/-3 points.
- Mean standard error vs the spread of the coefficient across draws: P1 0.311 vs 0.326; P2 0.143 vs 0.158. The standard errors may be **5-10% too small**, so the tests are mildly liberal.
- Two-way and firm-only errors are nearly identical here because permuting within quarter removes any common time shock by construction. This is therefore **not** evidence that firm-only clustering is safe on real data (see S3.8, simulation with firm-specific exposure to time shocks).
- An earlier unpaired run (different random outcomes for each call) showed P2 rejecting 0 of 100 draws; that comparison was invalid and is not used.

**Consequence (declared now):** no change to the primary design. Because the tests are mildly liberal, a one-sided p between **0.03 and 0.07 is called borderline** and is not claimed as a finding even if it passes Holm at 0.05.

## 10. Result (added after the single registered run, 5,000 permutations; `gg_output/stage3_results.json`)

Look-ahead audit passed (4,000 sampled rows re-verified against raw sales, zero timing violations). No design change was made after the outcomes were seen.

| Test | Estimate | One-sided p | Holm p | 95% interval | Detectable at 80% power |
|---|---|---|---|---|---|
| **P1** hazard, OPT x SALEPOST (pp per quarter; baseline exit rate 2.5%) | **-0.32** (se 0.30; 75,016 CEO-quarters, 1,943 exits, 324 after a sale) | 0.85 (wrong sign) | 0.85 | -0.91 to +0.27 | 0.75 (30% of baseline) |
| **P2** miss, OPT x SALE x EXIT (pp of price) | **-0.03** (se 0.17; 94,742 releases, 1,006 with sale and exit, 435 optimistic) | 0.43 | 0.85 | -0.37 to +0.31 | 0.43 |
| **P3** leavers minus stayers, 12m Carhart alpha | **-2.1%** (leavers +0.2%, n=157; stayers +2.3%, n=5,388) | 0.26 (permutation sd 2.4%) | 0.77 | about -7% to +3% | about 6% |

**H3a, H3b and H3c are not supported.** All 17 robustness checks are null: P1 has a negative coefficient in all seven variants, P2 has no consistent sign, and the falsification (sales before the guidance, R11) is null. Subperiods 2004-2014 and 2015-2025 agree. Attrition: 95.9% of releases followed by an exit have an actual vs 99.5% without an exit.

**Non-primary coefficients (descriptive, not tested hypotheses, not corrected for multiplicity):** in P2, exit alone is followed by a worse miss (X: -0.42, t -4.7), optimism alone by a worse miss (O: -0.07, t -3.5, partly mechanical), and a sale after guidance by a *better* outcome (S: +0.14, t +4.2); the sale-and-exit term is +0.47 (t +3.4), so CEOs who sold and then left show **no worse** miss than CEOs who left without selling (the opposite of the hypothesis). In P1, a sale since the latest guidance goes with a higher exit hazard (S: +0.44 pp, t 1.78, two-sided p 0.08), but not more so when guidance was optimistic. The exit effect is consistent with new-CEO write-offs (A8) or deterioration that precedes an exit, and does not show intent.

**Limits that remain:** power is modest for the interaction terms (the detectable P2 effect, 0.43 pp of price, is about the size of the whole exit main effect); CEOs only; optimism relative to consensus is a proxy for intent; the calibration showed the tests are mildly liberal (S3.9), so a null is, if anything, firmer than its p-value suggests; the tests are not independent of Stages 1-2 (same firms and sales data); project-wide formal tests are now 31.
