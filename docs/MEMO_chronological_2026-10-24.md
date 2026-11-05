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

## Stage 5: does the pattern depend on how opaque the firm's reporting is? (run 2026-11-04)

**Bottom line.** None of the four tests supports the idea that the Guidance Reversal pattern is stronger where reporting is opaque (every Holm-adjusted p-value is 1.0). **This is an underpowered null, not evidence that transparency is irrelevant.** With 24 and 18 flagged events in the two groups, Test 1 could only have detected a difference of about 42 percentage points of alpha per year; the interaction tests (2 and 3) were similarly coarse. A real effect of 5 to 10 points a year would not have been visible.

### The question

Stages 1 to 4 asked whether firms whose executive guides, sells stock and leaves later underperform, and found nothing. **Hypothesis H5:** that average hides a split. Where a firm's reporting lets outsiders rebuild its profit as price times quantity (high transparency), an executive who plans to sell and leave has little room to mislead, so the pattern should be harmless. Where reporting is opaque (low transparency), it should hurt. H5 predicts a *more negative* outcome at *low* transparency.

### How transparency is measured: the score $T$

$T \in \{0,1,2,3,4,5\}$ is one integer per 10-K filing. Higher means *more* transparent. It answers one question: *could an outside investor rebuild what the firm earns as price $\times$ quantity, for the parts of the business that make most of the profit?* Price or quantity counts as disclosed if it is stated or can be derived by dividing two disclosed figures for the same line and period.

| $T$ | Meaning |
|---|---|
| 5 | Volume and price (stated or derivable) for every line that makes the profit |
| 4 | Volume and price for about three quarters of the profit; one meaningful line is dollars only |
| 3 | About half the profit covered, or volume *or* price (not both) for the main engine |
| 2 | Dollars only: revenue and profit by line, no volume or price for the main engine |
| 1 | Almost opaque: total or regional revenue, drivers only in words |
| 0 | One revenue number, no breakdown |

A "razor-and-blade" firm (price and quantity only for the low-margin line, dollars only for the high-margin recurring line) is capped at 2. The number of segments does not matter.

- **Who scores it.** Claude Haiku, run through the Claude Code CLI on a frozen configuration (model, prompt, rubric version and CLI flags are hashed and checked before every run). It reads the 10-K (Item 1, segment note, MD&A segment text, operating-statistics text) plus the 10-Qs and earnings-release exhibits (8-K exhibit 99) filed in the 12 months up to that 10-K. Company names are stripped. Website-only investor decks and call transcripts are not included.
- **Phases.** A firm rarely changes how it reports revenue. Filings are grouped into *phases* (a new phase starts when the revenue breakdown changes drastically, found by a fuzzy label comparison plus a model check), and the first filing of each phase carries its score across the phase: 4,341 phases for 24,154 filings (99.2% carry a score).
- **No look-ahead.** For a departure event, $T$ comes from the last 10-K filed strictly before the announcement date the return windows start from; for a CEO-quarter, the last 10-K filed on or before the quarter start; for a guidance release, strictly before the release date. A filing more than 550 days old is treated as missing.
- **Reliability.** Weighted kappa between two reads of the same filing 0.94; against my own hand scores on 100 random filings 0.77 (95% CI 0.67 to 0.85, gate 0.60), 92% within one point. The model scores 0.5 points above my hand scores on average (it credits "price up 37%, units down 12%" and customer counts as price $\times$ quantity). In a recognition test the model named the firm in 55% of the name-stripped bundles it scored.
- **Groups.** Low and high $T$ are split at the 33rd and 67th percentile of $T$ over all Stage 2 departure events. Because $T$ takes only six values, the cut points are 2 and 3: **low is $T \le 2$, high is $T \ge 3$** (the middle group is empty). $Z = (T-\mu)/\sigma$ is $T$ standardised over all scored filings, used where $T$ enters a regression as a number.

### What the tests are (each is a Stage 1 to 3 test, extended by $T$)

Notation: **flagged** event = a CEO or CFO departure with guidance in the 180 days before the announcement, a Form 4 sale between the latest guidance and the announcement, and an abrupt departure (42 events with usable return windows, defined before returns were computed). Other symbols:

| Symbol | Meaning |
|---|---|
| $O$ | Optimistic guidance: 1 if the guided EPS midpoint is above analyst consensus at the guidance date |
| $S$ | Sale: 1 if the executive sold shares after the latest guidance (any open-market sale) |
| $X$ | Exit: 1 if the executive left before the guided period ended |
| $\text{Exit}_{i,q}$ | 1 if CEO $i$ leaves in quarter $q$, times 100 |
| $\text{Miss}_j$ | (actual EPS $-$ guided midpoint) / share price, in percent of price, for guidance release $j$ |
| $\lambda_i,\ \tau_q$ | Firm and year-quarter fixed effects |
| $W$ | Controls: log market cap, 12-month return, log tenure, age (plus a sale-in-last-12-months dummy in Test 2; log days to the period end and a quarterly dummy in Test 3) |
| $\alpha_g$ | Four-factor alpha (annualised) of a portfolio of flagged firms in group $g$ (low or high $T$) |
| $\hat\alpha_e$ | Annualised four-factor alpha of event $e$'s own 252-trading-day window (skipping the first 2 days) |

**Test 1 (T1), returns of flagged firms by $T$ group.** For each group $g$ hold the flagged firms from day +2 to +253 after the announcement and regress the portfolio's excess return on the four Carhart factors:

$$ r_{g,t} - r^{f}_{t} \;=\; \alpha_g + \beta_1 (r^{m}_t - r^{f}_t) + \beta_2\,\mathrm{SMB}_t + \beta_3\,\mathrm{HML}_t + \beta_4\,\mathrm{UMD}_t + \varepsilon_{g,t}. $$

Each group's alpha is compared with random placebo groups of the same size drawn from non-flagged events in the *same* $T$ group, and the test is the difference in excess alpha:

$$ \Delta \;=\; \big(\hat\alpha_{\text{low}} - \bar\alpha^{\text{placebo}}_{\text{low}}\big) \;-\; \big(\hat\alpha_{\text{high}} - \bar\alpha^{\text{placebo}}_{\text{high}}\big), \qquad H_5:\ \Delta < 0. $$

The p-value is the share of centred placebo contrasts at or below the observed $\Delta$.

**Test 2 (T2), exits.** The Stage 3 exit regression (does a CEO leave when guidance was optimistic and they sold?), with $Z$ interacted with every term:

$$ \text{Exit}_{i,q} = \lambda_i + \tau_q + c_1 O + c_2 S + c_3\,OS + \theta_0 Z + \theta_1\,ZO + \theta_2\,ZS + \theta_3\,ZOS + \gamma' W_{i,q} + u_{i,q}. $$

The coefficient of interest is $\theta_3$. $c_3>0$ is the Stage 3 pattern, so H5 (stronger at low $T$) means $\theta_3 < 0$. Standard errors are clustered by firm and by quarter.

**Test 3 (T3), earnings misses.** The Stage 3 miss regression with all seven terms of $O, S, X$ interacted with $Z$:

$$ \text{Miss}_j = \lambda_i + \tau_q + \textstyle\sum_{k} b_k\, m_k + \sum_{k} \theta_k\, Z\, m_k + \theta_0 Z + \gamma' W_j + u_j, \qquad m \in \{O,S,X,OS,OX,SX,OSX\}. $$

The coefficient of interest is $\theta_{OSX}$, on $Z \cdot O \cdot S \cdot X$. Stage 3 predicts a negative $b_{OSX}$ (sell and leave after optimistic guidance, then miss more), so H5 means $\theta_{OSX} > 0$.

**Test 4 (T4), leavers versus stayers.** Among executives who guided and sold, do those who left do worse than those who stayed, and more so at low $T$? On the low-$T$ and high-$T$ events:

$$ \hat\alpha_e = a + b\,\text{Leaver}_e + c\,\text{High}_e + d\,(\text{Leaver}_e \times \text{High}_e) + u_e, \qquad H_5:\ d > 0. $$

A positive $d$ means leaving hurts less where $T$ is high. The p-value comes from a wild cluster bootstrap by announcement quarter that imposes $d=0$ only (so an equal non-zero leaver effect in both groups is the null).

The four tests are corrected together with Holm's method; the project-wide count of formal tests is now 32. One-sided p-values refer to the direction H5 predicts.

### Results

| Test | Estimate (H5 predicts) | One-sided p | Holm-adj. p | Sample |
|---|---|---|---|---|
| T1 returns, $\Delta$ ($<0$) | **+0.283** per year (low $T$: +0.092, 24 flagged; high $T$: $-$0.191, 18 flagged); bootstrap 95% CI +0.048 to +0.520 | 0.961 | 1.0 | 42 flagged events; placebo pools 1,649 and 1,531 |
| T2 exits, $\theta_3$ ($<0$) | $-$0.040 pp per quarter per 1 sd of $T$ (se 0.309) | 0.449 | 1.0 | 73,623 CEO-quarters, 1,943 exits |
| T3 misses, $\theta_{OSX}$ ($>0$) | $-$0.452 pp of price per 1 sd of $T$ (se 0.251; two-sided p 0.075) | 0.962 | 1.0 | 93,013 releases, 1,006 with sale and exit |
| T4 leavers vs stayers, $d$ ($>0$) | +0.004 (se 0.044); leaver minus stayer alpha $-$0.010 at low $T$, $-$0.007 at high $T$ | 0.462 | 1.0 | 81 and 65 leavers; 3,162 and 2,066 stayers |

Secondary (outside the family): $T$ as a continuous number in Test 1, one-sided p 0.81; in Test 4, 0.25.

**How to read this.**
1. *Power.* The smallest effects detectable were about 0.42 per year of alpha difference (T1), 0.77 (T2) and 0.62 (T3). The null rules out only very large opacity effects.
2. *The opposite sign.* Tests 1 and 3 lean the other way: flagged firms did *worse* where reporting is transparent. With 18 and 24 flagged events, an unregistered direction and 32 project-wide tests, this is exploratory and not a finding.
3. *Measurement noise.* $T$ is noisy (the first and last filing of a phase differ by 0.5 points on average, kappa 0.68), which pulls every estimate toward zero. The model's upward bias and its ability to recognise firms are further limits (`docs/STAGE5_SPEC.md`, sections 10 and 11).
4. *Review.* An independent code review found and we fixed four issues (including dating $T$ from the wrong announcement date in Test 1); the corrected results moved no p-value by more than 0.004 (`docs/STAGE5_SPEC.md`, section 11).

Files: `docs/STAGE5_SPEC.md` (design, rulings, results), `outputs/results/stage5_results.json` (numbers), `src/gg_stage5.py` (tests), `docs/superpowers/plans/stage5-progress.md` (decision ledger).

---

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
