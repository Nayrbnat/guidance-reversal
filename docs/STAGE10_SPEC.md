# Stage 10: a pre-registered test of the large-firm lead, on sales not yet seen

Status: **written 2026-11-07, before any data that could test it exists on this account.** Author: Nayrb (requested 2026-11-07: "lean on large firms"). **This is a pre-registration, not a run.** Nothing is computed until the trigger in Section 5 is met.

## 1. Why this, and what it is not

Large firms is the one subgroup that points the same way in two samples: Stage 8 (2004 to 2024, 102,606 insider-days) gave -0.265 (standard error 0.164, one-sided p 0.055) and Stage 9 (2025, 3,345 insider-days) gave -2.00 (standard error 1.07, one-sided p 0.031). That is **not** a finding, for three reasons, fixed here so they are not forgotten:

1. **Selection.** Large firms was chosen for Stage 9 *because* it came close in Stage 8. Its Stage 9 result is therefore not an independent test of an a-priori hypothesis, and pooling the two (about -0.30, one-sided p about 0.03) would double-count the selection.
2. **The two groups are not the same.** Stage 8 and Stage 9 each split at their own sample median market cap: about $4.8 billion in Stage 8 but about $18 billion in Stage 9 (79% of Stage 9 insider-days are above the Stage 8 median, because 2025 has more large firms among scored filers). "Large" meant different firms in the two runs.
3. **Power.** Stage 9's estimate is about eight times Stage 8's with a standard error of 1.07; at that precision a significant estimate is likely inflated.

A plausible story exists (the Stage 5 transparency score is built from revenue disclosure that matters more, and is read more reliably by the scorer, for firms with segment reporting, and large firms have more analyst coverage so opacity could matter differently), but I did not predict it before Stage 8 and will not present it as a mechanism until a clean test supports it.

## 2. The hypothesis and test (fixed now)

**H5-L.** Among officers and directors at **large firms**, the avoided loss after an open-market sale is larger where transparency is low: $\theta_L<0$.

- **Large firm, defined once, in dollars:** market cap the day before the sale of **at least $4.83 billion** (the Stage 8 primary-sample median, log market cap 22.298, computed from market cap alone before any outcome was looked at for this stage). It is **not** re-estimated on new data.
- Everything else as in Stage 9: insider-days of at least $100,000 by officers and directors, outcome $Y=-100\times CAR(+1,+63)$ with pre-sale four-factor betas, winsorised at 1% and 99% on the new sample, $Z$ with the Stage 5 mean and standard deviation, controls as in Stage 8, one-sided, $p<0.05$.
- **Design change from Stages 8 and 9, chosen on standard errors alone (Section 2a):** **SIC two-digit industry-by-year-quarter fixed effects** (one effect per industry and quarter) in place of separate industry and quarter effects, with standard errors **clustered by firm only**, whatever the number of quarters.

### 2a. Why the design changed, and what it did and did not rely on

A comparison of design variants on the Stage 8 panel (`scripts/se_bakeoff_stage8.py`, results in `outputs/results/stage8_se_bakeoff.json`) printed only standard errors, sample sizes and minimum detectable effects; **no coefficient was printed or saved**, so the choice could not depend on the answer. On 204,174 insider-days: baseline 0.152, firm-only clustering 0.130, industry-by-quarter effects with firm clustering **0.123**, flexible controls 0.132, industry-by-quarter plus flexible controls 0.124. Two further variants were rejected: dividing the outcome by the stock's own volatility changes the units and gives no gain in standard-deviation units (0.0226 against 0.0240), and shorter or longer horizons change the question. Against the Stage 9 design (firm clustering, separate effects) the gain is about 6% in the standard error (0.130 to 0.123), measured on the large Stage 8 panel; it may differ on a smaller new sample, so the sample-size thresholds in Section 5 are **left at their conservative values**. Stages 8 and 9 are **not** to be re-run under this design and reported; at most a sensitivity table, labelled exploratory.
- One confirmatory test. The project-wide count of formal tests becomes 38.

## 3. What counts as new data

Sale dates **after 2025-09-30**. The 2025-01 to 2025-09 sales are used up: their large-firm result has been seen. Sales from 2025-10-01 need CRSP prices through 63 trading days later; this account's CRSP v2 file ends 2025-12-31, so the first usable sales are those with prices available, which today is none beyond 2025-09-30. **The test cannot run until CRSP 2026 prices are on the account** (WRDS releases annual updates; check `crsp.dsf_v2` for dates after 2025-12-31 and the Fama-French file, which already runs to 2026-09-30).

## 4. Power, fixed now so the stopping rule is not chosen after seeing data

The large-firm coefficient needs a standard error of about 0.40 (minimum detectable effect about 1.0, still 4 times Stage 8's -0.265, because this is what a one-to-two-year sample can reach). Scaling Stage 9's large-firm standard error (1.07 on 3,345 insider-days) says that needs **about 24,000 large-firm insider-days with a score**, roughly the 2025-10 to 2027-09 flow. A smaller sample than that cannot test the lead and the result would be reported as uninformative whatever its p-value.

## 5. Trigger and rules

- **Run once, when** (a) CRSP prices cover the 63-day outcome window of every sale in the sample, and (b) the sample has at least 15,000 large-firm insider-days with a score (so the standard error is about 0.50 and the minimum detectable effect about 1.2; below that the test is not run and the lead stays a lead).
- Print the sample, the standard error and the minimum detectable effect **before** the estimate, as in Stages 8 and 9.
- **Supported:** $\hat\theta_L<0$, one-sided $p<0.05$. **Not supported:** anything else, reported with the minimum detectable effect. A pooled estimate with Stage 8 is descriptive only.
- No other subgroup is run on the new data. The Stage 9 guidance interaction (-2.26) is a second lead; it is **not** part of this test and, if the author wants it tested, needs its own pre-registered line in a separate spec.

## 6. What to do in the meantime

Nothing is leaned on in the write-up: the memo reports large firms as a lead from two samples chosen in sequence. Two actions are free of look-ahead and useful now: (1) score the roughly 29% of Stage 8 insider-days without $T$, which raises power for the large-firm test (and for the Stage 8 test itself) without touching new outcomes, and (2) pull the detectability moderators the cache lacks (analyst coverage, trading volume, institutional ownership), which the literature suggests matter most for large firms. Neither changes the hypothesis in Section 2.
