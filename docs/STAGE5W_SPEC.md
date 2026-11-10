# Stage 5W: the Stage 5 opacity test T1 re-run on the widened flag (registered 2026-11-09, before any Stage 5W result was computed)

**Status:** REGISTERED, **exploratory by construction.** Stage 5 (`STAGE5_SPEC.md`) tested hypothesis H5 (the pattern hurts more where reporting is opaque, T low) on the 42 firms of the pre-registered strict flag. The supplement now leads with the widened flag (`STAGE1W_SPEC.md`: 492 firms with return data), so the extension's flagged-firm test is repeated on it for consistency. What I had seen: the Stage 5 T1 result on the strict flag (low-T minus high-T excess alpha +0.283, one-sided p 0.96, 24 low-T and 18 high-T firms) and the Stage 1W result (alpha +2.6% a year). **Not seen:** any T-split result for the widened flag.

## 1. Test (identical to Stage 5 T1 except the flag and the returns)

Flagged = the widened flag of `STAGE1W_SPEC.md` (guided in the prior 365 days; sale after the first of those releases; a guided period ends after the exit). T is the Stage 5 transparency score of the last 10-K filed before the announcement (frozen measurement, unchanged). Groups: low T (T below the first cut) and high T (T above the second cut) with the **Stage 5 cut points, 2.0 and 3.0, kept as fixed** (computed there from all events, not re-estimated). Statistic `did`: excess four-factor alpha of flagged firms (own alpha minus the mean alpha of same-year placebo portfolios drawn from non-flagged departures in the same T group) at low T minus the same at high T. H5: `did` < 0. Inference: the lower tail of the centred placebo contrast (5,000 draws per group, seed 0), as in Stage 5; a 95% interval from a bootstrap over announcement quarters (1,000 draws). Returns through 2025-12-31 (`gg_extend`, as in Stage 1W Amendment 1); day +2 to +253; at least 20 observed days.

## 2. Interpretation (fixed now)

Supported only if `did` is negative and the one-sided placebo p is below 0.05. Not corrected for multiple testing (a single exploratory repeat of T1); the Stage 5 family (T1 to T4) and its Holm correction are unchanged. Reported with the group sizes, the smallest detectable effect, and the Stage 5 strict-flag result beside it.

## 2b. Amendment 1 (2026-11-09; after the first attempt, before any waived-run result)

**What happened.** The first run was refused by the Stage 5 viability rule (`gg_transparency.group_viability`: the placebo pool must be at least five times the flagged group): low T has 310 flagged events against a pool of 1,422, high T has 172 against 1,412. Under the registered rule the widened-flag T1 is **not viable** and would be skipped. No estimate was produced.
**Change.** The test is run with that one rule waived (`--waive-viability`). Everything else is as in Section 1; the test already applies a finite-population correction for placebo draws that take a large share of a small pool (`GT.fpc_factor`). Because the rule was written to prevent exactly this situation, the waived result is **exploratory and should be read with caution**: placebo draws of 310 events from a pool of 1,422 overlap heavily, so the placebo distribution is narrower than in the strict-flag test and the p-value is only approximately calibrated. I decided to waive it knowing no waived result; the decision is not conditional on the outcome.

## 3. Results

(To be filled after the run, in `outputs/results/stage5w_results.json`.)
