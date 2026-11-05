# Stage 7: re-testing H5 at the sale level with between-firm variation and the original treatment

Status: **written 2026-11-05, before any Stage 7 regression was run.** Author: Nayrb. This is an **exploratory re-analysis**: it uses the same CEO sales as Stage 6, after Stage 6's null, so it is not a confirmation. Confirmation needs data not yet used (other officers, 2025 to 2026) and is a separate step.

What I looked at before writing it, and nothing else: counts of sale-days by exit type and by routine versus opportunistic class (below), and variance facts about T. No outcome (avoided loss) was regressed on anything.

## 1. Why these two changes

1. **Between-firm variation (lever 1).** In Stage 6 the firm fixed effects removed 95% of the variation in $Z$ itself: within a firm $Z$ has a standard deviation of 0.13 in the guide-sell-leave sales, against 0.99 across firms, and in 89% of firms $T$ never changes across their sales. That clearly costs power for the terms that use $Z$ on its own (the transparency-only test S1 in Stage 6). **I overstated it for the H5 term:** the coefficient on $Z\times G\times X$ is identified from differences across firms in the effect of a guide-sell-leave sale, which firm effects do not remove, so the gain for $\theta_3$ may be small, and firms with no guide-sell-leave sales add little under firm effects but serve as comparison under industry effects. H5 is also a statement about opaque firms against transparent firms, so industry-by-month effects are the natural specification. Whether power improves is read from the printed standard error of $\theta_3$ against Stage 6's 1.69, **before** the estimate. The price is more room for confounding between firms, handled below.
2. **The original treatment (lever 3).** The pre-registered hypothesis (H1, `docs/PREREGISTRATION.md`) is about executives who sell and leave **abruptly**, and the literature finds that **routine** insider sales carry no information while opportunistic ones do (Cohen, Malloy and Pomorski, 2012). Stage 6 used any exit and any sale, which dilutes the treatment with planned retirements and routine sales.

## 2. Treatment definitions (fixed now)

- **Exit announced within 365 days after the sale, $X^{ann}_s$.** The CEO has a departure event (`events.csv`) with announcement date in $(t, t+365]$ days. **Abrupt** uses the annotators' label where one exists (the flagged candidates) and otherwise the rule-based classifier of `events.csv` (`abrupt_rule`); **planned or other** is the complement among classified exits. Sale-days whose exit is announced but has no classification are **excluded** from the abrupt and the placebo tests (they cannot be assigned). Counts before this spec: 5,256 sale-days with an exit announced within 365 days, of which 929 abrupt, 3,083 planned or other, 1,244 unclassified.
- **Routine sale (after Cohen, Malloy and Pomorski, as I recall their rule).** A sale-day is **routine** if the CEO sold in the same calendar month in each of the three previous calendar years; **opportunistic** if the CEO sold in each of the three previous years but not in that month in all three; **unclassified** otherwise (fewer than three prior years of sales). Only years before the sale year are used. Counts: 3,657 routine, 18,139 opportunistic, 28,327 unclassified. **Non-routine** means not routine (opportunistic or unclassified), because routine requires an established pattern that a first-time or irregular seller cannot have.
- $G_s$, $Z_s$, $Y_s$ (the 63-day avoided loss), the sample, controls and the standard errors are as in Stage 6 (`docs/STAGE6_SPEC.md`), plus **$V_s$, pre-sale return volatility, as an extra control**.

## 3. Primary tests (family of 2, Holm)

Fixed effects for both: **SIC two-digit industry** and **year-month** (in place of firm and year-month). Standard errors clustered by firm and by month. Regression:

$$ Y_s = \lambda_{j(i)} + \tau_m + a\,G_s + b\,X_s + c\,(G_sX_s) + \theta_0 Z_s + \theta_1 Z_sG_s + \theta_2 Z_sX_s + \theta_3\,Z_sG_sX_s + \gamma' W_s + \delta V_s + u_s . $$

| | Exit variable $X_s$ | Sample | Isolates |
|---|---|---|---|
| **P7a** | the Stage 6 definition (CEO leaves within 365 days after the sale, ExecuComp) | the Stage 6 primary sample | the effect of lever 1 alone |
| **P7b** | abrupt exit announced within 365 days | non-routine sale-days with a classified exit (or no exit) | levers 1 and 3 together |

**H5-S: $\theta_3 < 0$** (the guide-sell-leave sale avoids more loss where transparency is low), one-sided, Holm over the two. The project-wide count of formal tests becomes 35.

**Power before estimates.** As in Stage 6: the standard error of $\theta_3$, the minimum detectable effect (about 2.49 times the standard error), and a firm-cluster bootstrap that plants effects $-\delta Z_s$ into the guide-sell-leave sales. Printed before any estimate. If P7b's minimum detectable effect is larger than Stage 6's, I report it as underpowered and do not interpret a null.

## 4. Secondary analyses (outside the family)

| | |
|---|---|
| S1 | P7a and P7b with firm fixed effects (to show how much the fixed-effects choice matters; P7a with firm effects replicates Stage 6 apart from $V$) |
| S2 | **Placebo: guide and a planned or other exit** (the same regression with $X$ = planned exit); H5 should not hold here if it is specific to abrupt exits |
| S3 | P7b on all sale-days including routine ones; and on opportunistic sale-days only |
| S4 | P7b with optimistic guidance $O$ in place of $G$ |
| S5 | Horizons 21 and 126 trading days |
| S6 | First sale-day per CEO-quarter |
| S7 | Industry fixed effects at one-digit SIC division instead of two-digit |

## 5. Interpretation rules

- Support requires $\theta_3<0$ and a Holm-adjusted $p<0.05$ in P7a or P7b. Anything else is "not supported", reported with its minimum detectable effect.
- Because this is a re-analysis, a significant result is **not** a finding: it must be replicated on data not used here before it is reported as support for H5. A null with a small minimum detectable effect is informative on its own.
- The industry-by-month specification does not remove confounding between firms (opaque firms may differ in other ways): the Stage 6 controls, the volatility control and the two-digit industry effects are what is done about it, and the result is read with that limit.

## 6. Known limits fixed now

Same CEO sales as Stage 6 (a re-analysis); the abruptness label is the rule-based classifier of `events.csv` for most exits (its agreement with hand labels is evaluated by `validation/evaluate.py` and was not recomputed for this stage, so its precision for abrupt exits is not known here); 37% of exits have no announcement text and drop out of the abrupt and placebo tests; the routine rule is my recollection of the published rule, not checked against the paper's text; only 146 firms contribute guide-sell-leave sale-days after an abrupt exit.

## 7. Results (run 2026-11-05; the design above is unchanged)

Power, printed before any estimate: P7a standard error 1.78 and minimum detectable effect 4.43 points per standard deviation of $T$ (Stage 6: 1.69 and 4.21), so industry-by-month effects did **not** add power for the H5 term; P7b standard error 3.62 and minimum detectable effect 9.02 (583 guide-sell-leave sale-days from 142 firms), so the abrupt-exit and non-routine restrictions **cost** power.

| Test | $\theta_3$ (standard error) | One-sided p | Holm |
|---|---|---|---|
| P7a | +0.07 (1.78) | 0.52 | 1.00 |
| P7b | +2.53 (3.62) | 0.76 | 1.00 |

Neither primary test is significant. Placebo (planned exits): +0.04 (2.93). Opportunistic sales only: -6.13 (5.43), p 0.13, 279 sale-days from 56 firms. **One nominally significant secondary result: abrupt exits with optimistic guidance $O$: -5.33 (2.57), one-sided p 0.021, 243 sale-days.** It is one of ten secondary tests in a re-analysis of data already used (a 40% chance that at least one reaches 5% by luck), rests on few sale-days, and is the optimism split where Stage 2's only significant result also appeared; it is not a finding and is a candidate for a pre-registered test on new data. The other secondary versions (21 and 126 days, first sale per quarter, SIC division) have the opposite sign. Results file: `outputs/results/stage7_results.json`.
