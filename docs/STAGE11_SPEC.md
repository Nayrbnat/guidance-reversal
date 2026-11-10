# Stage 11: four ways to extract more from the same data (registered 2026-11-07, before any Stage 11 outcome was computed)

**Status:** REGISTERED. What I looked at before writing it: the column layout of the guidance panel, of the I/B/E/S actuals file and of the audit workbook, and the counts already published in Stages 8 and 10. No Stage 11 return, p-value or corrected estimate was computed. Exploratory follow-up to a null average effect; the aim is to say whether the null survives methods designed for low power.

**Why these four.** An arXiv review of low-power methods (2026-11-07; notes in `docs/METHODS_NOTES_POWER.md`) found nothing that creates information, but four things that use the information better: (1) move the test to the date where the effect should appear, (2) use tests built for a few large effects, (3) report how large an effect the data rule out, (4) correct the bias that a noisy language-model score puts into the opacity results.

## P11a. The reaction to the first guidance after the exit

**Question.** When the new management issues its first guidance, do firms whose departed executive guided, sold and left react worse than comparable firms?

- **Sample and flag:** as Stage 15 (`docs/LOSS_RATE_SPEC.md`): eligible departures 2004-2025 (announcement identified, not death); flagged = guided in the prior 365 days, sold after the first of those releases, at least one guided period ends after the exit; comparator = guided with outstanding guidance and **no** sale.
- **Event date:** the earliest I/B/E/S guidance release for the firm dated from announcement + 2 calendar days to announcement + 365 days (any period). The cached actuals file carries no announcement dates, so the date the guided period resolves cannot be used; the first new guidance is dated and is where a reversal would show. Events with no such release are dropped, for both groups.
- **Outcome:** CAR(-1,+1) in percentage points: the sum over the three trading days around the release date of r - rf - beta'F (Mkt-RF, SMB, HML, UMD), with beta estimated by OLS on the 252 trading days ending the day before day -1 (at least 120 observations). Winsorised at 1% and 99%.
- **Statistic:** mean CAR of flagged minus mean CAR of the comparator. **H: negative**, one-sided. Inference: bootstrap over announcement quarters (5,000 draws, seed 0); p = Phi(difference / bootstrap SE).
- **Reported with it:** the bootstrap interval; the smallest detectable difference (2.49 x SE); sample sizes. Secondary: windows (0,+1) and (-1,+5); flagged vs placebo groups of all non-flagged eligible departures with a post-exit release (year-matched, 5,000 draws).

## P11b. Sparse-effect tests on the 12-month return

**Question.** The story predicts a few large losers, not a shift in the average. Is there an excess of unusually bad outcomes among the flagged firms?

- **Event-level p-values:** for each flagged event i, p_i = (1 + number of non-flagged eligible departures in the same announcement year whose 12-month market-adjusted BHAR (days +2 to +253, Stage 15) is at or below event i's) / (1 + the number of such departures). Under the null the p_i are uniform.
- **Statistics:** (a) **Higher criticism** (Donoho-Jin), HC = max over i <= 0.5n of sqrt(n)(i/n - p_(i)) / sqrt(p_(i)(1 - p_(i))); (b) **positive Cauchy combination**, T = mean over p_i < 0.5 of tan((0.5 - p_i) pi); (c) for comparison, the **mean-based test**: mean flagged BHAR minus the mean of the placebo draws.
- **Calibration (no asymptotics):** 5,000 placebo groups of the same size, matched on announcement-year counts, drawn from the non-flagged eligible departures; each draw's p-values are computed against the remaining pool (leave-out); p = (1 + number of draws at least as extreme) / (1 + 5,000).
- **Primary: HC.** Cauchy and the mean test are reported beside it. Secondary: the same tests on the strict flag, and on the P11a CARs.

**Family.** P11a and P11b (HC) are the two Stage 11 tests, Holm-adjusted together. The project-wide count of formal tests becomes **40** (36 after Stage 8, plus the held-out 2025 test, the large-loss test and these two).

## P11c. How large an effect do the data rule out?

Descriptive, no family. For each registered test with an estimate and a standard error, report (i) the 90% interval bound that matters, (ii) a one-sided non-inferiority test against a smallest effect size of interest (SESOI), and (iii) a Bayes factor for no effect.

- **SESOI, fixed now:** Stage 8 (opacity, per SD of the transparency score): **0.5 points** of avoided loss. Stage 15 (large-loss share): **5 percentage points** more than the comparator. Stage 1 (alpha): **-5% a year**. Tests in Stages 5 to 7 are reported with their interval bounds only; I set no SESOI for them because their precision is too low to be informative at any plausible size.
- **Bayes factor:** H0 theta = 0 against H1 theta ~ half-normal in the direction the hypothesis predicts, with scale equal to the SESOI, on the normal likelihood N(estimate; theta, SE^2). Sensitivity at 0.5 and 2 times the SESOI. **Reading:** BF01 above 3 is moderate evidence for no effect, between 1/3 and 3 inconclusive, below 1/3 evidence for an effect.
- **Non-inferiority:** H0: effect is at least the SESOI in the harmful direction; reject at one-sided 5% if the one-sided 95% bound lies inside the SESOI.

## P11d. Correcting the opacity results for the noisy score

**Question.** The model's transparency score has reliability 0.77 against the author's hand scores, which pulls coefficients toward zero. How far?

- **Method (regression calibration):** on the 100 representative audit filings (the author's hand score H and the model score M, matched by accession number), regress H on M by OLS, H = a + beta M. A regression coefficient on the standardised model score Z_m then relates to the effect of a standard deviation of the true score by theta_H(sd) = theta_m x sd(H) / (beta x sd_global(M)), where sd_global(M) is the Stage 5 standard deviation over all scored filings and sd(H) the standard deviation of the hand scores.
- **Assumption:** the outcome depends on the model score only through the true score (surrogacy). **Not implemented:** prediction-powered inference and design-based supervised learning; with 100 labels they add validity, not power.
- **Uncertainty:** 5,000 bootstrap resamples of the 100 audit pairs combined with draws from the original estimate N(theta_m, SE^2); report the median, a 95% interval, and the one-sided p in the original direction.
- **Applied to:** Stage 8 primary (Z), Stage 6 primary (Z x G x X) and Stage 7 P7a (Z x G x X). Reported descriptively; it cannot rescue a failed primary.

## Interpretation rules

- A Stage 11 test is "supported" only if its Holm-adjusted p < 0.05 and, for P11b, the mean-based test does not contradict the direction. Anything else is "not supported", reported with its smallest detectable effect.
- Because Stage 11 reuses the data of Stages 1 to 10, a significant result is exploratory and would need new data before it is reported as a finding.
- Known limits fixed now: the first post-exit guidance is a proxy for the date the guidance comes due; firms that stop guiding after an exit drop out of P11a for both groups; the placebo is matched on year counts only; the BHAR is market-adjusted, not risk-adjusted; P11d assumes surrogacy and rests on 100 labels.

## Ruling (made after the first run of P11d, before any reinterpretation)

P11d as registered took sd(H), the standard deviation of the hand scores, from the 100 audit filings. The audit sample is not distributed like the population of scored filings: its model scores average 3.27 with a standard deviation of 1.25, against 2.60 and 1.04 over all 23,971 scored filings (26 of 100 audit filings score 5, against about 7% of the population). The registered formula therefore overstated the correction (scale factor 1.44). I replaced it, for the headline, with a population-consistent factor that rebuilds sd(H) in the population as sqrt(beta^2 sd_global^2 + residual variance), which assumes the calibration line and its scatter transport (scale factor 1.27). A simulation test (`tests/test_gg11.py`, `test_regression_calibration_unrepresentative_audit`) reproduces the bias with an audit sample that oversamples both tails and shows the new factor recovers the true effect. **Both versions are reported below.** The change makes the correction smaller, not larger, and changes no conclusion. P11a to P11c were not changed.

## Results (run 2026-11-07, `python scripts/run_stage11.py`, 5,000 draws, seed 0; `outputs/results/stage11_results.json`)

**Verdict: none of the four methods finds anything. P11a and P11b are not supported (Holm-adjusted p = 1.0 for both; project-wide count of formal tests 40), and both point the opposite way from the hypothesis.**

**P11a, reaction to the first guidance after the exit.** First post-exit guidance found for 460 of the 480 flagged firms and 1,152 of 1,254 in the two groups. CAR(-1,+1): flagged +0.08 points, comparator (guided, no sale) -0.26, difference **+0.34 (SE 0.45), one-sided p = 0.773** (wrong direction); smallest detectable difference **1.12 points** (n 459 and 682). Windows (0,+1): -0.03 (p 0.476); (-1,+5): +0.64 (p 0.914). Against all non-flagged departures: mean-based p 0.888; higher criticism p 0.874; Cauchy p 0.796.

**P11b, sparse tests on the 12-month return (480 flagged, 2,854 in the pool).** Higher criticism p = 1.000, Cauchy p = 1.000, mean-based p = 0.775. Only **4** flagged firms have an event-level p-value below 0.05, against 24 expected by chance: the flagged firms have fewer, not more, extreme losers. Strict flag (36 events): HC p = 0.607, Cauchy 0.673, mean 0.528. The pool includes firms that never issue guidance, who are less steady; the Stage 15 comparison among guided firms (-3.4 points) has the same sign, so part of this reflects selection, not protection.

**P11c, how large an effect is ruled out.**

| Test | Estimate (SE) | Largest harm not ruled out (95%) | SESOI | Rejects harm at SESOI | BF01 (0.5x, 1x, 2x scale) |
|---|---|---|---|---|---|
| Stage 15 large-loss share, flagged vs no-sale (pts) | -3.40 (2.18) | +0.19 | 5.0 | yes (p 0.0001) | 3.2, **5.9**, 11.5 |
| Stage 8 opacity (pts per SD of T) | -0.13 (0.15) | +0.38 | 0.5 | yes (p 0.007) | 1.0, **1.6**, 2.9 |
| Stage 1 alpha (% a year) | -0.59 (5.48) | +9.61 | 5.0 | no (p 0.21) | 1.1, **1.3**, 1.9 |

Reading (rule fixed in advance): Stage 15 is **moderate evidence for no effect** (BF01 5.9). Stage 8 excludes effects larger than about 0.4 points per standard deviation but is **inconclusive** on the Bayes factor (1.6), because its estimate has the predicted sign and a small predicted-sign effect is not ruled out. Stage 1 cannot rule out an alpha of -5% a year; the 42-event test is uninformative, as the power analysis said. Interval bounds only, harm per standard deviation of T: Stage 5 T2 0.55, T3 0.86, T4 0.07; Stage 6 2.51; Stage 7 P7a 2.85, P7b 3.43.

**P11d, correction for the noisy transparency score** (100 audit pairs: correlation 0.84, R-squared 0.71; the hand scores average 2.77 against 3.27 for the model).

| Result | Model | As registered (x1.44) | Population-consistent (x1.27) |
|---|---|---|---|
| Stage 8 primary (Z) | -0.129 (SE 0.152) | -0.185, 95% [-0.604, +0.254] | **-0.163, 95% [-0.532, +0.226]**, p(effect<0) 0.79 |
| Stage 6 primary (Z x G x X) | +0.266 (1.690) | +0.383 [-4.274, +5.299] | +0.338 [-3.761, +4.683], p 0.43 |
| Stage 7 P7a (Z x G x X) | +0.074 (1.778) | +0.106 [-4.794, +5.286] | +0.094 [-4.227, +4.672], p 0.47 |

Correcting for the noise moves the Stage 8 estimate from -0.13 to -0.16 and widens the interval; no result changes sign or becomes significant. The noise in the score does not explain the null.

**Limits.** The first post-exit guidance is a proxy for the date the guidance comes due; 8% of events have none, and firms that stop guiding drop out of both groups. The P11b pool is not restricted to guided firms. The BHAR is market-adjusted. P11d assumes surrogacy and a linear calibration, rests on 100 labels, and the audit sample is concentrated at high model scores (26 fives). Reuses the data of Stages 1 to 10, so any significant result would have been exploratory.
