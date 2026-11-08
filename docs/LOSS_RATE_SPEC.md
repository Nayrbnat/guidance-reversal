# Large-loss-rate test (registered 2026-11-04, before any outcome was computed)

**Numbering.** In the project's stage numbering this is **Stage 10**. It was registered and run on 2026-11-04 and first labelled Stage 9, before the held-out 2025 confirmation (`docs/STAGE9_SPEC.md`) took that number.

**Status:** REGISTERED. Written after Stages 1-5 and after counts of the sample (not outcomes) were seen; no loss rate has been computed. Exploratory follow-up to a null average effect.

## Question

Stages 1-5 test whether the *average* return of firms is lower after a guide-sell-leave pattern. The story behind the project predicts something different: a few large failures (a guidance cut that reverses the old CEO's guidance) with an unremarkable average. So the question here is the **frequency of large losses**: **are firms where the executive guided, sold and left before the guidance came due more likely to lose a large amount relative to the market than other departures?**

## Sample (same claim as the original, widened; no text classification)

An event is a CEO/CFO departure announced 2004-2025 (S&P 1500, as in Stages 1-4), not a death, with the announcement identified and a usable 12-month return window. It is **flagged** when all three hold:

1. **Guided:** at least one EPS guidance release in the 365 days before the announcement.
2. **Sold:** an open-market sale by the executive (strict name match, score 3) between the first of those guidance releases and the announcement (`sell_val_post_guid > 0`).
3. **Left before the guidance came due:** at least one of those guidance releases covers a period that ends after the announcement (`n_unfulfilled >= 1`).

This replaces "left abruptly" (judged from announcement text) with the hypothesis's own wording, "left before the guidance came due". The strict original flag (42 firms) is reported as a sensitivity.

## Outcome

**Large loss** = 12-month market-adjusted buy-and-hold return of the firm below **-30 percentage points**: BHAR = product(1 + firm daily return) - product(1 + CRSP value-weighted market return) over trading days +2 to +253 after the announcement (the same window as Stage 1; delisting returns included). The threshold -30 points is **fixed now** as the primary. Sensitivities fixed now: -20 and -40 points; windows with at least 200 observed days only.

## Test

- **Statistic:** the share of flagged events with a large loss, minus the placebo mean share.
- **Placebo:** 5,000 random groups of non-flagged eligible departures with windows, with the same number of events per announcement year as the flagged set, seed 0. One-sided p-value: the share of placebo draws (with the +1 correction) whose large-loss share is at least the flagged share.
- **Within-departure comparator (secondary):** flagged events vs departures with outstanding guidance but **no** sale, difference in loss shares with a 95% interval from a bootstrap that resamples announcement quarters.
- **Detectable difference:** reported as 2.5 times the standard deviation of the placebo shares.

## Rules

Everything is reported whatever it shows. The threshold, the flag and the placebo rule are not changed after the results are seen. This test adds one formal test to the project. It was registered and run on 2026-11-04, after Stage 5 (32 tests) and before Stages 6 to 8 were written, so it was the 33rd; the Stage 6 to 8 specs were numbered without counting it, so the project-wide count after Stage 8 is 37 (36 as those specs count). A one-sided p between 0.03 and 0.07 is called borderline, as in Stage 3.

## Result (run 2026-11-04, `python src/gg_lossrate.py`, 5,000 placebo draws, seed 0; `outputs/results/lossrate_results.json`)

**Verdict: no excess of large losses. Flagged firms lose a large amount *less* often, not more.** Sample: 3,334 eligible departures with a 12-month window, 480 flagged (guided, sold, left before the guidance came due), 774 with outstanding guidance and no sale.

| Comparison | Large-loss share (BHAR < -30 pts) | Difference | Test |
|---|---|---|---|
| Flagged (480) vs placebo groups of other departures | 14.2% vs 22.0% (placebo sd 1.7 pts) | -7.8 pts | one-sided p = 1.000 (wrong direction) |
| Flagged (480) vs guided-no-sale departures (774) | 14.2% vs 17.6% | -3.4 pts, 95% CI -7.7 to +0.8 | within-departure comparator |

Sensitivities (registered): threshold -20 pts: 22.3% vs 32.4% placebo; flagged vs no-sale -6.0 pts, CI -11.1 to -0.9 (significant in the wrong direction). Threshold -40 pts: 8.3% vs 14.8%; vs no-sale -1.7 pts, CI -5.3 to +2.2. Complete windows only (>= 200 days): 14.6% vs 21.6%. Strict abrupt-exit flag (36 events with a window): 22.2% vs 23.4%, p = 0.63, detectable difference 16.9 pts (underpowered).
Power: on the 480-event sample the test could detect a difference of about 4.3 pts in the large-loss share; the effect observed is the opposite sign and larger than that.

**Reading.** With about 11 times the events of Stage 1 and an outcome that targets the large-loss tail, the pattern still does not predict failure. The placebo comparison is flattered by selection, because firms that issue EPS guidance (all flagged firms do) are larger and steadier than the placebo pool, which includes non-guiders; the within-departure comparison removes that and still points the wrong way, so selling before leaving does not mark firms that later fail in this sample. Both agree with the Stage 3 and 4 reading that executive sales before an exit mark orderly departures.
**Limits.** Market-adjusted, not risk-adjusted, returns. Sale and guidance windows end at the first-identified `dep_ann`, while return windows start from the annotator-corrected announcement date, so a few sales may fall after the corrected date. Windows overlap in calendar time and the placebo matches year counts only, not size or sector. The strict-flag sensitivity has 36, not 42, events because this sample requires an identified announcement and a window under the new rules. Project-wide test count: 37 after Stage 8 (see Rules).
