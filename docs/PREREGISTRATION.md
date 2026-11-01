# Pre-specified test plan: Guidance Reversal

Written 2026-10-24, **before any returns were computed for the rebuilt flag.** If anything below is changed after the
returns have been seen, the change is listed under "Deviations" at the bottom, with the reason.

## What had already been seen (exposure disclosure)

- Early versions of this project (old text labels, outcome-conditioned flag, 2010-2025) were run and discussed. Their
  headline gap (-21% vs -4% BHAR) was circular. A day-0 version of the flag showed **no** gap (+1.2%, p = 0.71).
- Those versions had large measurement errors (the old "abrupt" rule had 23% precision against hand labels; 12% of
  executive names were mis-parsed). The measurement has since been rebuilt using hand-labelled data that contain **no
  return information**. The returns of the rebuilt flag had **not** been computed when this file was written.
- So this is not a blind first look at the idea, but it is a pre-specified test of a re-measured flag.

## Hypothesis (H1)

Firms whose departing CEO or CFO (1) issued EPS guidance in the 180 days before the departure announcement,
(2) sold shares between that most recent guidance and the announcement, and (3) left abruptly, earn **negative
risk-adjusted returns over the following 12 months**.

## Definitions (fixed)

- **Departure and announcement date:** ExecuComp CEO/CFO departures, 2004-2025; announcement date = date of the earliest
  Capital IQ executive-change item in which that executive is the person leaving. Where blind annotators
  (validation/) identified the item, their choice overrides the rule-based one.
- **Recent guidance:** an I/B/E/S EPS guidance release within 180 days before the announcement.
- **Sale:** Thomson Form 4 sale (trancode S, acqdisp D) by an owner matching the executive by last name AND full first
  name or nickname (match score 3), dated from the most recent guidance to the announcement date.
- **Abrupt:** judged from the announcement text by independent annotators who never see returns, using the written
  rubric in validation/ (effective immediately / within about 30 days with no orderly transition, interim named,
  terminated or forced out, resignation without announced succession). Planned retirements and succession plans are
  not abrupt. Annotator reliability on this item was measured (92% agreement on 40 duplicated tasks).

## Primary test

1. Hold every flagged firm from trading day +2 to +253 after its announcement (skip the 2-day reaction).
2. Each calendar day the portfolio is the equal-weighted average of open positions (delisting returns included).
3. Regress its excess return (minus the risk-free rate) on **Mkt-RF, SMB, HML, UMD**, weighting each day by the number of
   open positions, with Newey-West (10 lags) standard errors. The test statistic is the **annualised alpha** (daily x 252).
4. Inference by a **placebo test**: 2,000 random portfolios of other departures, with the same number of events in each
   announcement year as the flagged set, run through the identical procedure. The placebo pool is departures whose
   announcement was identified in the text (so flagged and placebo events have comparably accurate dates); the pool of
   all other departures is shown as a sensitivity check. The one-sided p-value is the share of
   placebo alphas at or below the observed alpha.
5. Reject the null only if **p < 0.05** and the flagged set has at least 40 events with usable windows. With fewer, the
   result is reported as underpowered and no inference is drawn.

## Secondary analyses (one family, Holm-adjusted together)

S1 CEO departures only; S2 6-month holding period; S3 lenient insider matching (initial-only matches allowed);
S4 rule-based "abrupt" instead of annotator judgement; S5 six-factor model (adds RMW, CMA); S6 guidance + sale
**without** the abruptness condition (tests whether abruptness matters at all); S7 excluding 2008-09 and 2020;
S8 sample split: announcements 2004-2014 vs 2015-2025 (same sign in both?).

Everything else (subgroups, event-time BHARs, matched-control differences, machine-learning prediction) is
**exploratory** and labelled as such. It is not used to claim a finding.

## Power (from simulation in test_gg.py, clean synthetic market)

With about 66 flagged events the placebo test detects an alpha of about -10% per year roughly 90% of the time and a
-5% alpha much less often. Real data are noisier and events cluster, so power is lower. A null result therefore rules out
**large** effects, not small ones, and the write-up must say so.

## What would change my mind

Whatever the sign. The flag definition above is not to be tuned after seeing returns. Alternative definitions may be
shown only as exploratory sensitivity, clearly labelled.

## Result (added after the single pre-specified run)

Primary: 42 flagged events with usable windows (>= 40, so inference allowed); alpha -0.6% per year, t = -0.11, placebo p = 0.39 -> **H0 not rejected**.
All eight secondary tests null (Holm-adjusted p = 1.0). Details: `MEMO.md`, `gg_output/study_results.json`, `guide_and_go_analysis.ipynb`.
The flag definition and test were not changed after the returns were seen.

## Deviations and further exposure (added 2026-10-24, before the primary test was run)

1. **Dry-run leak (my error).** The mechanics dry-run of `gg_study.py` printed secondary-test results for the **real**
   subsets, not just the random flag. Two results were therefore seen before the primary test:
   - S4 (rule-based abrupt flag, n = 43 events with windows): alpha +1.5%/yr, placebo p = 0.49.
   - S6 (all guidance + sale candidates regardless of exit type, n = 267): alpha +2.5%/yr, placebo p = 0.63.
   Both are null. The primary flag (annotator-confirmed abrupt) is a subset of S6 and overlaps heavily with S4, so the
   primary result is **not independent** of these two. The dry run has since been changed so that it only ever computes the
   random flag.
2. **Data repair before any primary return was computed.** The WRDS REST API had silently truncated large pulls (about half of all
   firms had no price history). All return data were re-pulled completely via Postgres. This changes the data, not the
   flag definition or test.
3. **Realistic power.** The placebo distribution on the real data has a standard deviation of about 6.7% per year
   (mean open positions about 3). That implies the minimum alpha detectable at 80% power (one-sided 5%) is roughly
   **-17% per year**, much larger than the -10% in the clean simulation. A null result rules out only large effects.

---

# Stage 2 (REVISED 2026-10-24, before any Stage-2 returns were computed)

*This replaces an earlier draft of Stage 2 that was built around an abruptness grid. Before any of its returns were computed the objective was clarified
by the project owner: the question is whether executives **intentionally gave optimistic guidance, sold stock, and then left**. Abruptness is not part of the
hypothesis, and screening on it cut 286 candidates to 45 without a substantive reason. The abruptness work stays in Stage 1 and as an exploratory split.*

**Hypothesis H2.** Departing CEOs/CFOs who (1) issued EPS guidance, (2) sold shares, and (3) then left, earn negative risk-adjusted returns afterwards; and the effect is stronger
when the guidance was **optimistic** (the proxy for intent).

**Sample.** CEO/CFO departures announced 2004-2025 with an identified announcement (rule, or annotator where available), excluding exits identified as **death** (cannot be intentional).
Any other exit type counts. No abruptness condition.

**Two fixes found after Stage 1 (declared here, not applied retroactively to Stage 1).** (a) "CEO" had been assigned to anyone who held that role in any year; Stage 2 universe **U2** uses the role in the
executive's *last* ExecuComp row; **U1** keeps the Stage-1 definition. (b) Deaths are excluded (see above).

**Link between guidance and sale (L):**
L1 guidance <= 180 days before the announcement and a sale between the latest guidance and the announcement (the Stage-1 link) · L2 as L1 with guidance <= 365 days ·
L3 guidance <= 365 days and **any** sale in the prior 365 days · L4 **any** sale in the prior 365 days, no guidance requirement.
(Sale = strict name match, score 3. L4 does not involve guidance, so it is a pure "sold then left" test and is not part of F2.)

**Family F1: does the pattern predict negative alpha?** 4 links x 2 universes = 8 tests. Calendar-time portfolio exactly as in Stage 1 (12-month hold from day +2, equal-weighted, delisting returns,
Fama-French-Carhart alpha with weights = open positions); placebo test with the same number of events per announcement year drawn from other departures **in the same universe**
(identified announcement, death excluded); 5,000 draws; one-sided, H1: alpha < 0.

**Family F2: does optimistic guidance matter?** For L1, L2, L3 x 2 universes = 6 tests. Among the F1 events, **O** = the latest guidance midpoint is above consensus at the guidance date (`opt_last` > 0);
**N** = at or below consensus. Events with no consensus are dropped from F2. Statistic: alpha(O) minus alpha(N), annualised. Inference: permute the O/N labels within announcement year 5,000 times;
one-sided, H1: difference < 0 (optimistic guiders do worse).

**Correction.** Holm-Bonferroni across the 14 tests (8 + 6). F1 cells with fewer than 40 events with windows, or F2 cells with fewer than 20 in either arm, are reported but excluded from the family (declared now).
A Stage-2 finding is claimed only if Holm-adjusted p < 0.05; every test is reported regardless.

**Exploratory (not in the family, reported as such):** guidance *raised* versus the firm's own previous guidance for the same period as the intent split; ex-post guidance misses (for periods ending after the exit) among
guide-sell-leave events versus other departures; continuous optimism (Spearman with 12-month BHAR); excluding health/personal exits; splitting by the Stage-1 abruptness judgement.

**Exposure and honesty.** (1) Stage 1 was null; Stage 2 was designed *because* of that, which is a forking path. Holm correction covers Stage 2 only, so across the whole project 28 formal tests have now been
looked at (Stage 1: 1 primary + 8 secondary; Stage 2: 14 + any added). (2) The Stage-1 secondary S6 (guidance + sale, any exit, n = 267: alpha +3.0%, placebo p = 0.63) is nearly the same sample as F1's L1 x U1 cell
(differences: deaths excluded, an identified announcement required). (3) No Stage-2 returns, optimism splits or L2-L4 samples had been examined when this was written.

**Result (added after the single Stage-2 run, 5,000 draws; `gg_output/stage2_results.json`).** 3,650 eligible events (35 deaths excluded).
- **F1:** all 8 cells null, none negative: L1xU1 +2.0% (n=235, p=0.52), L1xU2 +0.2% (157, 0.38), L2xU1 +0.9% (277, 0.34), L2xU2 +0.5% (187, 0.35), L3xU1 +2.2% (670, 0.73), L3xU2 +2.8% (483, 0.80), L4xU1 +1.3% (1,114, 0.41), L4xU2 +1.7% (821, 0.62). Holm p = 1.0 throughout. Placebo sd 1.2-4.2%/yr.
- **F2:** diff O-N: L1xU1 -4.9% (p=0.066), L1xU2 -3.2% (0.20), L2xU1 -1.1% (0.28), L2xU2 -0.7% (0.42), **L3xU1 -6.1% (O 221 events -1.9%, N 400 events +4.2%; p=0.0034, Holm 0.0476)**, L3xU2 -5.4% (p=0.046, Holm 0.60).
- **Exploratory:** guidance *raise* split goes the opposite way (raisers higher alpha, p 0.66-0.97); post-exit guidance misses are not more negative in the group (median +0.0002 to +0.0009 vs -0.0011 to -0.0018); Spearman(optimism, 12m BHAR) = -0.03 (p=0.47, n=592); excluding health exits changes nothing.
- **Reading, per the interpretation rule below:** the one Holm-significant test is in the looser link only (L3), not L1/L2; the optimistic arm itself is not clearly negative (-1.9%); the contrast is driven by the non-optimistic arm's unusually high alpha; the intent proxy is contradicted by the raise-based split; and it is one of ~28 project-wide tests after a Stage-1 null. Described as weak, suggestive evidence needing replication. U2 "CEO" events are ~all CEOs (CFO coverage is near zero because ExecuComp rarely populates `leftofc` for CFOs).

**Interpretation.** A significant result in a looser link (L3/L4) with null in the strict link (L1) is weaker evidence and will be described that way. If everything is null with the larger samples, the conclusion is that
guide-sell-leave is not associated with underperformance, to the precision shown by the interval widths, which will be reported.

---

# Stage 3 (registered 2026-10-24, BEFORE any Stage-3 outcome was computed)

Source of truth: `STAGE3_SPEC.md` (identical text). Decisions confirmed by the project owner: CEO only; any open-market sale; optimism = guidance above consensus.
Before registering I looked only at counts (episodes, coverage, sample and treatment-cell sizes in section S3.7), never at misses, returns or exit rates by group.

### S3.0 Why Stage 3

Stages 1-2 compare leavers with other leavers using 12-month alpha. They cannot say whether *leaving* adds anything, and returns are a noisy proxy for the actual claim:
**"optimistic guidance, the executive sells, leaves, and the company misses."** Stage 3 adds stayers as a control group and tests the miss directly.

### S3.1 Three questions and the one statistic that answers each

| Test | Question | Unit | Statistic (one-sided H1) |
|---|---|---|---|
| **P1 hazard** | Is a CEO more likely to leave in a quarter when the latest guidance was optimistic **and** they sold after it? | CEO x quarter, guiders only | coefficient on OPT x SALEPOST > 0 |
| **P2 miss** | Is the realised miss worse when guidance was optimistic, the CEO sold after it, **and** left before the period ended? | one guidance release | coefficient on OPT x SALE x EXIT < 0 |
| **P3 leavers vs stayers** | Do firms where the CEO guided, sold and **left** underperform firms with the same pattern where the CEO **stayed**? | calendar-time portfolios | alpha(leavers) - alpha(stayers) < 0 |

Holm correction over these three. Everything else is robustness, reported but not in the family.

### S3.2 Definitions

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

### S3.3 Assumptions I am making, and what breaks if they are wrong

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

### S3.4 Rules

- **Minimum sizes:** P1 needs >= 100 exits; P2 needs >= 100 releases with SALE=1 and EXIT=1; P3 needs >= 40 leavers and >= 40 stayers with windows. Otherwise reported as underpowered, outside the family.
- I report the **minimum detectable coefficient** (about 2.5 x the standard error) for P1 and P2, so a null is read correctly.
- A finding is claimed only if Holm-adjusted p < 0.05. For P2 the sign must also survive R1, R4, R5, and R11 must be null; otherwise it is called fragile.
- Everything is reported regardless of outcome. (Checks R3, R7 and R8, which needed holdings-scaled sales and CFO exits, were dropped before any outcome was computed to keep the set small.)

### S3.5 Robustness (not in the family)

R1 continuous standardised optimism in place of the binary flag · R2 sale >= $1M · R4 outcome = actual minus consensus · R5 first release per (firm, period) only ·
R6 logit with year FE and no firm FE for P1 · R9 exits excluding reason RETIRED · R10 year-quarter FE in P2 ·
**R11 falsification: sales in the 180 days *before* the guidance instead of after (should be null if the story is guidance-contingent selling)**.

### S3.6 Disclosures

I had already seen all of Stage 1-2, including that the guide-sell-leave group's *post-exit guidance misses were not worse* than other departures' (an exploratory result). P2 is a regression version of that, so it is **not blind**. Project-wide formal tests would be 28 + 3 = 31.

### S3.7 Feasible sample sizes (counts only, from `gg_panel_data.py feasibility`)

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

### S3.8 Amendment 1 (2026-10-24, after an assumptions audit, BEFORE any Stage-3 outcome was computed)

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

### S3.9 Null calibration (2026-10-24, before the registered run; outcomes randomised, no real outcome seen)

`gg_stage3_calibrate.py` permuted the outcome (exit indicator, miss) within year-quarter on the real panels and re-ran P1 and P2 with the same random outcome for both standard-error types (60 draws).
- One-sided 5% rejection rate: **P1 6.7%, P2 6.7%** (two-way and firm-only identical); share with p < 0.10: 13% and 18% (nominal 10%). Uncertainty on these rates is about +/-3 points.
- Mean standard error vs the spread of the coefficient across draws: P1 0.311 vs 0.326; P2 0.143 vs 0.158. The standard errors may be **5-10% too small**, so the tests are mildly liberal.
- Two-way and firm-only errors are nearly identical here because permuting within quarter removes any common time shock by construction. This is therefore **not** evidence that firm-only clustering is safe on real data (see S3.8, simulation with firm-specific exposure to time shocks).
- An earlier unpaired run (different random outcomes for each call) showed P2 rejecting 0 of 100 draws; that comparison was invalid and is not used.

**Consequence (declared now):** no change to the primary design. Because the tests are mildly liberal, a one-sided p between **0.03 and 0.07 is called borderline** and is not claimed as a finding even if it passes Holm at 0.05.

### S3.10 Result (added after the single registered run, 5,000 permutations; `gg_output/stage3_results.json`)

Look-ahead audit passed (4,000 sampled rows re-verified against raw sales, zero timing violations). No design change was made after the outcomes were seen.

| Test | Estimate | One-sided p | Holm p | 95% interval | Detectable at 80% power |
|---|---|---|---|---|---|
| **P1** hazard, OPT x SALEPOST (pp per quarter; baseline exit rate 2.5%) | **-0.32** (se 0.30; 75,016 CEO-quarters, 1,943 exits, 324 after a sale) | 0.85 (wrong sign) | 0.85 | -0.91 to +0.27 | 0.75 (30% of baseline) |
| **P2** miss, OPT x SALE x EXIT (pp of price) | **-0.03** (se 0.17; 94,742 releases, 1,006 with sale and exit, 435 optimistic) | 0.43 | 0.85 | -0.37 to +0.31 | 0.43 |
| **P3** leavers minus stayers, 12m Carhart alpha | **-2.1%** (leavers +0.2%, n=157; stayers +2.3%, n=5,388) | 0.26 (permutation sd 2.4%) | 0.77 | about -7% to +3% | about 6% |

**H3a, H3b and H3c are not supported.** All 17 robustness checks are null: P1 has a negative coefficient in all seven variants, P2 has no consistent sign, and the falsification (sales before the guidance, R11) is null. Subperiods 2004-2014 and 2015-2025 agree. Attrition: 95.9% of releases followed by an exit have an actual vs 99.5% without an exit.

**Non-primary coefficients (descriptive, not tested hypotheses, not corrected for multiplicity):** in P2, exit alone is followed by a worse miss (X: -0.42, t -4.7), optimism alone by a worse miss (O: -0.07, t -3.5, partly mechanical), and a sale after guidance by a *better* outcome (S: +0.14, t +4.2); the sale-and-exit term is +0.47 (t +3.4), so CEOs who sold and then left show **no worse** miss than CEOs who left without selling (the opposite of the hypothesis). In P1, a sale since the latest guidance goes with a higher exit hazard (S: +0.44 pp, t 1.78, two-sided p 0.08), but not more so when guidance was optimistic. The exit effect is consistent with new-CEO write-offs (A8) or deterioration that precedes an exit, and does not show intent.

**Limits that remain:** power is modest for the interaction terms (the detectable P2 effect, 0.43 pp of price, is about the size of the whole exit main effect); CEOs only; optimism relative to consensus is a proxy for intent; the calibration showed the tests are mildly liberal (S3.9), so a null is, if anything, firmer than its p-value suggests; the tests are not independent of Stages 1-2 (same firms and sales data); project-wide formal tests are now 31.

---

# Stage 4 (EXPLORATORY ITERATION; plan written 2026-10-24 after Stage 3 results were seen, before any Stage-4 estimate)

**Status.** Stage 3 was null. The project owner asked me to examine the direction of the signs, what controls are missing, and whether the data give any insight. Everything below is a **forking path by construction**: I have seen the Stage-3 coefficients. To keep it honest, Stage 4 uses a **hold-out protocol**: specifications are explored on 2004-2014 only; a fixed rule selects at most a few candidate coefficients; only those are tested once on 2015-2025. Stage 3 did print the registered coefficients (and R12) for both halves, so the halves are not pristine for *those* coefficients; the new controls, sale types, exit types and event-time profile have not been seen in either half.

## What the Stage-3 signs suggest (diagnosis)

1. **Planned retirement is a first-order omitted factor.** RETIRED is the most common exit reason, and both selling and leaving rise sharply with age. A linear age control cannot separate "sold because retiring" from "sold then left under a cloud". The registered sale-and-exit term was **positive** (CEOs who sold and left show no worse miss than those who left without selling), which fits planned, orderly exits.
2. **"Any sale" mixes routine diversification with informed selling.** 27% of guidance releases have a CEO sale. The standard Cohen-Malloy-Pomorski rule (routine = sold in the same calendar month in each of the previous three years) separates them.
3. **Past performance and guidance style are uncontrolled.** Poor results cause both forced exits and misses; some firms habitually guide high. Controls: the firm's last realised miss, the share of its past guidance that was above consensus, 12-month return volatility, CEO ownership.
4. **The exit main effect (-0.42 pp of price) has two readings with different timing:** deterioration *before* the exit (forced turnover) or a new CEO's write-offs *after* it. An event-time profile separates them.
5. **P1 sign.** The negative OPT x SALEPOST coefficient may reflect confident optimists who sell and stay, or sellers who are planning retirement (an exit type outside the hypothesis). Splitting exits into retirement and other (competing risks) tests this.

## Specifications (all with firm and year-quarter fixed effects, two-way clustered SEs, as in Stage 3)

- **M1 augmented controls.** Replace linear age by age bins (55-59, 60-64, 65+) and add tenure squared, last realised miss (plus missing flag), share of past guidance above consensus (plus flag), log 12-month daily return volatility, CEO ownership % of shares (prior fiscal year; plus flag). Re-estimate P1 (OS) and P2 (OSX).
- **M2 sale type.** Replace "any sale" by **non-routine sale** (not routine by the 3-year same-month rule; includes unclassifiable early-tenure sales), keeping routine sales as a separate control. Terms: OS_nr (P1) and OSX_nr (P2).
- **M3 exit type.** P1 as competing risks (separate outcomes: exit with reason RETIRED vs other exits; term OS for each); P2 with EXIT split into retired and other, term OSX for each.
- **Event-time profile (descriptive, both halves, no hold-out).** Miss by quarters between the period end and the nearest CEO exit of the firm (-4..+4), separately for exits preceded by a CEO sale in the prior 365 days and those not; reference = releases more than 4 quarters from any exit.

## Hold-out protocol and selection rule (fixed now)

- **Exploration sample:** guidance dates (P2) or quarters (P1) in 2004-2014. **Confirmation sample:** 2015-2025, not estimated until the candidate list is written to `gg_output/stage4_candidates.json`.
- **Candidate terms (8):** P1 M1 OS; P1 M2 OS_nr; P1 M3 OS_retired; P1 M3 OS_other; P2 M1 OSX; P2 M2 OSX_nr; P2 M3 OSX_retired; P2 M3 OSX_other.
- **Selection rule:** a term is carried to confirmation if its exploration |t| >= 2.5 (two-sided). The direction is whatever the exploration estimate shows (a coefficient opposite to the hypothesis counts; it would be reported as such).
- **Confirmation:** each carried term is tested once on 2015-2025 for the *same sign*, one-sided, Holm-corrected over the number carried. Success = adjusted p < 0.05 **and** the mildly-liberal band of Stage 3 (0.03-0.07 = borderline) is respected.
- If no term passes the exploration rule, the conclusion is that the augmented controls, sale types and exit types do not reveal a pattern. Nothing is "found" from exploration alone.
- Project-wide formal tests: 31 (Stage 3) + up to 8 candidates tested in confirmation.

### S4 Amendment 1 (2026-10-24; after the Stage-4 exploration run on 2004-2014 and the failed confirmation attempt, before any M3b estimate)

**What happened.** The M1 and M2 explorations were null (augmented controls leave OS at -0.31 and OSX at +0.08; only 4.8% of sales are routine, so M2 is almost M1). The one candidate selected by the rule was **P2 M3 OSXr** (+0.97, t 3.5; 56 releases from 25 firms), a positive sign, opposite to the hypothesis.
**Confirmation could not be run:** the 2015-2025 half has **zero** releases with a sale followed by a retirement exit. Cause, found afterwards: the ExecuComp exit `reason` field is **missing for 87% of exits (4,713 of 5,432) and for essentially every exit after 2009**. The "retired" label therefore exists only for a small set of pre-2010 exits, and "other" mixes unlabelled retirees with everything else. **The reason-based exit type is an invalid measurement; M3 is withdrawn, and OSXr is recorded as untestable (not confirmed), not as a finding.**
Consequences for Stage 3: R9 (excluding RETIRED) was effectively a null operation, and deaths after 2008 are not excluded (the text classifier finds 29 death exits against 14 flagged by `reason`; about 1% of exits).

**Replacement M3b (exit type from the Capital IQ text classification, available for 87-90% of exits in both halves).** Planned = `retirement_planned`; forced = `resigned_abrupt` or `terminated_forced`; other = everything else, including exits with no classified text. Terms: P1 competing risks (separate hazards of planned, forced and other exits; term OS in each) and P2 with EXIT split into planned/forced/other (terms OSXp and OSXf). Same fixed effects, clustering, hold-out and selection rule (|t| >= 2.5 in 2004-2014 carries a term to a single test on 2015-2025, Holm over the number carried). Candidate terms: P1 M3b planned OS, P1 M3b forced OS, P2 M3b OSXp, P2 M3b OSXf.

**Descriptive evidence already seen (counts only, no outcomes):** a CEO sale in the 365 days before the exit is more common before planned retirements (35.4% of 1,644) than before abrupt resignations (20.5% of 1,188) or terminations (18.2% of 88); overall 26.3%. The event-time profile (both halves) shows misses of about -0.2 to -0.6 pp of price in the four quarters **before** and after exits not preceded by a sale, and roughly zero before exits preceded by a sale (negative only after, in 2004-2014). The interpretation (sales mark planned, orderly exits; non-sale exits follow deterioration) is a hypothesis generated from these data, tested only by M3b.

### S4 Result (exploration 2004-2014, hold-out 2015-2025; `gg_output/stage4_*.json`)

**Registered iteration (M1, M2, M3, M3b): nothing carried to confirmation that could be confirmed, and no pattern was found.**
- **M1 augmented controls** (age bins, tenure squared, last realised miss, share of past guidance above consensus, volatility, CEO ownership): the registered terms do not move. P1 OS -0.31 (t -0.76), P2 OSX +0.08 (t +0.51), against -0.31 and +0.07 with the Stage-3 controls (2004-2014).
- **M2 sale type:** only 4.8% of matched CEO sales are routine under the three-year same-month rule, so "non-routine" is almost "any sale"; OS_nr -0.38 (t -0.95), OSX_nr +0.07 (t +0.42). Uninformative rather than negative.
- **M3 (reason-based exit type): withdrawn** (S4 Amendment 1). The single candidate (OSXr +0.97, t 3.5; 56 releases, 25 firms) is a measurement artifact of an exit-reason field that is missing for 87% of exits and for essentially all exits after 2009; it is untestable in 2015-2025 (zero such releases).
- **M3b (text-based exit type): no term reached |t| >= 2.5, so none was carried to confirmation.** P1 planned-exit OS -0.36 (t -1.46); forced-exit OS -0.13 (t -0.85); P2 OSXp +0.25 (t 0.78); OSXf -0.58 (t -1.12).

**Post-hoc replication of exploration observations on 2015-2025 (descriptive, not a registered test, no selection rule):**
- Exits followed by worse misses: **forced exits** -1.05 (t -3.3) -> -0.57 (t -2.8) (same sign, smaller). **Planned exits** +0.01 -> **-0.55 (t -3.2)** (did not replicate; the planned/forced distinction is not stable).
- Sale-and-exit offsets flip sign across halves (SXp -0.25 -> +0.47; SXf +0.92 -> +0.30, not significant in 2015-2025), and so do the triple terms (OSXp +0.25 -> -0.62; OSXf -0.58 -> +0.60; OSXo +0.24 -> +0.62). Opposite signs across halves are the signature of noise.
- **Event-time profile (both halves, same sign pattern):** in the four quarters **before** a CEO exit not preceded by a sale, misses are negative (2004-2014: -0.25, -0.64, -0.22, -0.31 pp of price; 2015-2025: -0.04, -0.19, -0.25, -0.23); before exits preceded by a sale in the prior 365 days they are about zero or slightly positive (+0.08, +0.11, +0.03, +0.04; +0.11, +0.11, -0.04, +0.02). 22,736 releases fall within four quarters of an exit (9,961 after a sale). This is the only pattern that appears in both halves.

**What this teaches.** (1) There is no sign convention under which the registered tests succeed: the triple interaction is positive or negative depending on the half and the exit type. (2) Missing controls do not explain the null: prior misses, age, volatility and ownership leave the interaction terms unchanged. (3) Measurement matters: the ExecuComp exit-reason field cannot be used after 2009, and routine-sale classification is too strict for CEO data. (4) The robust descriptive fact is about timing: **exits not preceded by a CEO sale follow quarters of misses (deterioration, probably performance-driven turnover); exits preceded by a sale do not.** Read with the sale-then-exit hazard (a sale since the latest guidance raises the quarterly exit probability by about 0.4 pp, two-sided p about 0.08), CEO sales before an exit look like markers of planned, orderly departures rather than of an exit ahead of bad news. A legal channel is also plausible (executives holding adverse information are restricted from selling), but I have not tested it.
Project-wide formal tests remain 31, plus the one candidate that could not be tested. Stage 4 is exploratory throughout.
