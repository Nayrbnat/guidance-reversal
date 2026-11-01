# Memo: Guidance Reversal, results after rebuilding the measurement

**Date:** 2026-10-24
**Data:** WRDS (ExecuComp, CRSP incl. delisting returns, I/B/E/S guidance, Thomson Form 4, Capital IQ text, Fama-French factors); S&P 1500 CEO/CFO departures announced 2004-2025
**Files:** `guide_and_go_analysis.ipynb` (results), `PREREGISTRATION.md`, `limitations.md`, code in `guide_and_go.py`, `gg_labels.py`, `gg_tests.py`, `gg_study.py`, tests in `test_gg.py`

---

## Bottom line

**The Guidance Reversal pattern does not predict underperformance in this sample.** The pre-specified primary test finds a risk-adjusted return of
**-0.6% per year (t = -0.11, placebo p = 0.39)** for the 42 flagged firms. All eight pre-specified secondary tests are also null (Holm-adjusted p = 1.0).
This is a credible null, not a proof of absence: with 42 events the test could reliably detect an alpha of about **-20% a year**, but not -5% to -10%.
A better-powered version of the question (any firm where the executive issued guidance, sold, then left, 267 firms) shows **+3.0% a year (p = 0.63)**
and excludes underperformance larger than about 4% a year.

## Stage 4: iterating on the null (exploratory, with a hold-out)

After the Stage 3 null I asked three questions: is the sign wrong, are controls missing, is there an insight in the pattern? To avoid fishing, I explored on 2004-2014, fixed a selection rule, and kept 2015-2025 for confirmation.

* **Sign:** no sign convention rescues it. The key optimism x sale x exit term is +0.07 (2004-2014) and -0.03 (full sample) with Stage 3 controls, positive or negative depending on the half and the exit type.
* **Controls:** adding age bands, last realised miss, guidance style, volatility and CEO ownership changes nothing (P1 -0.31 -> -0.31; P2 +0.07 -> +0.08).
* **Data problem found:** ExecuComp's exit-reason field is blank for 87% of exits and for nearly all exits after 2009. The one exploratory "hit" (a positive sale-optimism-retirement effect, t 3.5, 56 releases from 25 firms) came from that field and cannot be tested out of sample. Exit type is now taken from the Capital IQ text classifier. No M3b term passed the exploration rule.
* **What replicates in both halves:** CEO exits not preceded by a sale follow quarters of earnings misses (about -0.2 to -0.6 pp of price in the four quarters before and after the exit); exits preceded by a sale in the prior year do not (about zero before the exit). A sale since the latest guidance also raises the quarterly exit probability a little (about +0.4 pp on a 2.5% base, two-sided p about 0.08).
* **Reading:** CEO sales before an exit look like markers of orderly, planned departures, not of an exit ahead of bad news. Exits that follow deterioration are not preceded by sales (plausibly because executives with adverse information are restricted from selling; this is not tested).
* **What does not replicate:** the planned-versus-forced split and every sale-by-exit-type term flip sign between the two halves.

## Stage 3 update: stayers as controls and a direct test on earnings misses

Pre-registered (`STAGE3_SPEC.md`, amended before any outcome was computed) with firm and year-quarter fixed effects and two-way clustered errors; null-calibrated; look-ahead audited in code.
Population: S&P 1500 CEOs of firms that issue EPS guidance (77,414 CEO-quarters, 98,169 guidance releases, 5,388 stayer events).

| Test | Question | Estimate | One-sided p |
|---|---|---|---|
| P1 | Do CEOs leave more when guidance was optimistic **and** they sold after it? | -0.32 pp per quarter (baseline 2.5%) | 0.85 |
| P2 | Is the realised miss worse when optimistic guidance, a sale and an exit all occur? | -0.03 pp of price | 0.43 |
| P3 | Do guide-sell-**leave** firms underperform guide-sell-**stay** firms? | -2.1%/yr (leavers +0.2%, stayers +2.3%) | 0.26 |

**None is significant (Holm 0.77-0.85), and all 17 robustness checks agree**, including the falsification test with sales *before* the guidance. This answers the earlier objection that there was no stayer control group: the control group is now in, and the pattern does not predict leaving, misses, or underperformance relative to it.
Descriptively (not tested hypotheses): exits are followed by worse misses, but CEOs who sold before leaving show **no worse** misses than those who left without selling, which is the opposite of the intent story.
Power: the interaction effects are detectable only if they are about the size of the whole exit effect (0.43 pp of price); a small effect cannot be ruled out. Two caveats on method: the tests were mildly liberal in calibration, and the stayer rule was changed before running to avoid survivorship bias.

## Stage 2 update: dropping abruptness (wider sample, intent proxied by optimistic guidance)

Abruptness cut 286 candidates to 45 without a substantive reason, so Stage 2 (pre-specified in `PREREGISTRATION.md` before its returns) drops it: any non-death CEO/CFO exit where an insider sale is linked to guidance (L1-L3) or to the prior year (L4).
Intent is proxied by guidance **above consensus**. Holm over 14 tests.

* **The group is not underperforming.** All eight cells have alpha between +0.2% and +2.8% a year (n = 157 to 1,114; placebo p 0.34-0.80). Intervals are roughly +/-2% (n=1,114) to +/-8% (n=157), far tighter than Stage 1.
* **Optimism split:** optimistic guiders minus the rest = -0.7% to -6.1% (all negative). **One test clears Holm: L3 x all CEOs/CFOs, -6.1%/yr, p = 0.003, Holm-adjusted 0.048.** Why I do not call it a finding:
  (1) the optimistic arm is only -1.9%; the gap comes mostly from the other arm at +4.2%; (2) it vanishes in the strict links (L1 p=0.07, L2 p=0.28); (3) the alternative intent split (guidance *raised* vs the firm's prior guidance) goes the opposite way;
  (4) after-exit guidance misses are not worse; (5) continuous optimism vs 12m BHAR: Spearman -0.03 (p=0.47); (6) Stage 2 followed a Stage 1 null (forking path), ~28 project-wide tests; (7) its "CEO" subset is nearly all CEOs.
* **Takeaway:** the overall pattern is still unsupported; "optimistic guidance, then sell, then leave" is a lead worth a replication with out-of-sample years or a stayer control group, not a result.

## The idea tested

A CEO or CFO issues earnings guidance, sells stock, then leaves abruptly before the guidance comes due; the company then underperforms.
**Guidance Reversal** is the name for firms where all three conditions hold (recent guidance, a sale after it, an abrupt exit).

## What was rebuilt (and why earlier results were wrong)

| Problem in the earlier version | What it did to the results | Fix |
|---|---|---|
| WRDS REST API silently cut off large downloads | **About half of all firms had no price history**, so every return number rested on half the sample | Pull via Postgres (complete: 10.2M vs 5.1M CRSP rows in the 2004 pull) |
| Old "abrupt" rule | 23% precision against hand labels; called 57% of departures abrupt (true rate about 21%) | Sentence-level classifier, tested on 280 hand labels; every flagged event read twice by independent annotators (94% agreement) |
| Old "government destination" label | 0 of 40 sampled cases were real (keywords like "Secretary of the Company") | Removed |
| 12% of executive names mis-parsed ("J.D." taken as surname "D") | Broke insider and text matching | Fixed and unit-tested |
| Flag required a later bad outcome | The -21% vs -4% gap was circular | Flag uses only announcement-date information |
| Guidance and actuals on different split bases | "Misses" of up to +940% of price | Rebased with CRSP split factors |

## The test (written down before the returns were computed)

Hold each flagged firm from day +2 to +253 after its announcement; each day form the equal-weighted portfolio of open positions (delisting returns included);
regress its excess return on Mkt-RF, SMB, HML, UMD; test the intercept against a **placebo distribution** of 2,000 random same-size portfolios of other departures
(matched on announcement-year counts).

| | n | alpha per year | placebo p | Holm-adj. p |
|---|---|---|---|---|
| **Primary: flagged firms** | 42 | **-0.6%** | 0.39 | n/a |
| S1 CEOs only | 42 | -0.6% | 0.39 | 1.0 |
| S2 6-month hold | 42 | +0.4% | 0.49 | 1.0 |
| S3 lenient insider matching | 59 | +1.6% | 0.53 | 1.0 |
| S4 rule-based abrupt | 43 | +1.4% | 0.52 | 1.0 |
| S5 six-factor model | 42 | +0.1% | 0.45 | 1.0 |
| S6 any exit type (no abruptness) | 267 | +3.0% | 0.63 | 1.0 |
| S7 excluding 2008-09, 2020 | 35 | -0.1% | 0.47 | 1.0 |
| S8a announced 2004-2014 | 15 | +3.8% | 0.54 | 1.0 |
| S8b announced 2015-2025 | 27 | -4.0% | 0.31 | 1.0 |

## How far to trust this

* **Strengths:** pre-specified test with a placebo null and multiple-testing correction; measurement validated on hand labels and read twice;
  simulation shows the test recovers a planted effect and holds its false-positive rate; data bug found and fixed.
* **Weaknesses:** only 42 events (all CEOs, half technology); the strict flag cannot exclude modest effects (95% interval about -16% to +15% a year);
  no control group of executives who guided and sold but *stayed*; annotators are language models, not people; insider name matching is tested but not hand-audited;
  S&P 1500 only.
* **Honest disclosures:** the test plan was written after I had seen earlier versions of the project, and a dry-run mistake exposed two secondary results (both null) beforehand
  (`PREREGISTRATION.md`).

## What to take away

1. The "guide, sell, leave" story is **not supported** here; the early positive-looking numbers came from data and labelling errors.
2. The project's value is the **pipeline and the discipline**: it found and fixed its own data problems, validated its measurement, pre-specified its test, and reported a null.
3. A fair description: *"I tested whether executives who issue guidance, sell stock and leave abruptly are followed by underperformance. After repairing data problems I found along the way and
   validating my text labels against 280 hand-labelled cases, a pre-specified calendar-time test on 42 events found no effect (alpha -0.6%/yr, p = 0.39), and the broader group showed none either; the sample could only rule out large effects."*

## Next steps (ranked)

1. Control group: firm-quarter panel of executives who guided and sold, with and without leaving.
2. Larger universe (micro-caps, other officers, non-U.S.) for more flagged events.
3. Human audit of 50-100 labels and 50 insider matches.
4. Direct test of the mechanism: GAAP-matched guidance misses after the executive leaves.
