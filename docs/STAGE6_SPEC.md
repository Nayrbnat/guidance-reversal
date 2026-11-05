# Stage 6: H5 at the level of the individual insider sale

Status: **written 2026-11-04, before any sale-level return was computed.** Author: Nayrb. Everything below is fixed before the outcome is looked at; changes after the run are logged as rulings and reported as such.

## 1. Why this stage exists

Stage 5 tested H5 (the Guidance Reversal pattern is stronger where reporting is opaque) on 42 flagged events and on panel interaction terms, and could detect only very large effects (minimum detectable annual alpha difference of 0.42 in Test 1). The most closely related literature (Duellman et al., 2025, on reporting complexity and insider trade returns; Rogers and Stocken on detectability and insider trading profits) tests opacity against **the returns that follow individual insider trades**, which uses thousands of observations instead of 42 events. Stage 6 asks the same H5 question that way.

**Question.** After a CEO sells stock, is the firm's subsequent abnormal return lower (the sale "avoided a loss") at firms with low transparency than at firms with high transparency, and is that concentrated in guide-sell-leave sales?

**H5-S (directional, one-sided).** The avoided loss on a guide-sell-leave sale is larger where $T$ is low.

## 2. Unit and sample

- **Unit:** a CEO sale-day: one row per CEO episode (`ep_id`) and calendar day with at least one open-market sale, with the sale values summed. Source: `panel_sales.pkl` (Thomson Form 4 sales matched to ExecuComp CEO episodes; 478,792 sales, 105,393 sale-days before filters).
- **Window:** sale dates 2004-01-01 to 2024-12-31 (so that a 252-day window ends within the CRSP sample).
- **Size filter (fixed now):** sale-day value at least **$100,000**. Below that, sales are mostly routine (tax withholding, small plan sales). Robustness: all sales, and at least $1 million.
- **Exclusions:** sales by CEOs who die within 365 days; sale-days with no score $T$ (no 10-K filed strictly before the sale date, a filing older than 550 days, or an extraction failure); sale-days without CRSP returns.
- 10b5-1 plan sales cannot be separated before 2023 (the checkbox appears on Form 4 only from April 2023); this is a stated limitation.

## 3. Variables

| Symbol | Definition |
|---|---|
| $Y_{s}$ | **Avoided loss:** $-100 \times \text{CAR}_s(+1,+h)$ in percentage points; positive when the stock does worse than its factor-predicted return after the sale. Primary $h = 63$ trading days (about one quarter, the next earnings cycle). |
| $\text{CAR}_s(+1,+h)$ | $\sum_{d=t+1}^{t+h}\big(r_{d} - r^f_{d} - \hat\beta_s' F_{d}\big)$ with $F$ = Mkt-RF, SMB, HML, UMD. $\hat\beta_s$ is estimated by OLS (with an intercept) on the 252 trading days ending the day **before** the sale, minimum 120 observations, so no post-sale information enters the benchmark. Delisting returns are included; if the stock delists inside the window the CAR sums the available days (at least $h/2$ required). Winsorised at 1% and 99%. |
| $G_s$ | 1 if the firm issued EPS guidance in the 180 days strictly before the sale date (latest I/B/E/S guidance). |
| $O_s$ | 1 if that latest guidance midpoint is above consensus (secondary). |
| $X_s$ | 1 if the CEO leaves (ExecuComp `leftofc`) within 365 days **after** the sale. Ex-post, as a treatment definition, exactly as in Stage 3 P2. |
| $T_s$, $Z_s$ | Transparency score of the last 10-K filed strictly before the sale date (stale after 550 days), and its standardisation as in Stage 5 (mean and standard deviation over all scored filings). |
| $W_s$ | Log market cap and 12-month return the day before the sale, log sale value, log tenure, age. |

## 4. Specifications

**Primary (the single confirmatory test).**

$$ Y_s = \lambda_i + \tau_m + a\,G_s + b\,X_s + c\,(G_sX_s) + \theta_0 Z_s + \theta_1 Z_sG_s + \theta_2 Z_sX_s + \theta_3\,Z_sG_sX_s + \gamma' W_s + u_s . $$

$\lambda_i$ is a firm effect, $\tau_m$ a year-month effect, standard errors are clustered two ways (firm, year-month). **H5-S: $\theta_3 < 0$**, one-sided, $p<0.05$. $c>0$ would mean guide-sell-leave sales avoid losses on average (the Stage 3 pattern at sale level). The family size is 1; the project-wide count of formal tests becomes 33.

**Secondary (reported, outside the family; none can rescue a failed primary).**

| | Specification |
|---|---|
| S1 | $Y_s$ on $Z_s$ and $W_s$ for all sales: do insiders at opaque firms sell better-timed (the literature's test) |
| S2 | Primary with $O_s$ in place of $G_s$ |
| S3 | Primary with low-$T$ and high-$T$ dummies (cut points 2 and 3) in place of $Z_s$ |
| S4 | Horizons $h = 21, 126, 252$ |
| S5 | Sale value at least $1 million; and all sales |
| S6 | First sale-day per CEO-quarter only (removes overlapping windows) |
| S7 | Sales 2004 to 2014 and 2015 to 2024 |
| S8 | Moderators of Section 6 (detectability and litigation risk), when computed |

## 5. Power, fixed before the real outcome is read

The script plants an effect $\delta \in \{0, 1, 2, 4\}$ percentage points per standard deviation of $Z$ into $Y$ for the guide-sell-leave sales (adding $-\delta Z_s$ when $G_sX_s=1$), re-estimates the primary regression and reports the share of replications with $\theta_3<0$ and $p<0.05$ (200 replications), plus the minimum detectable effect from the standard error (about 2.49 times the standard error, one-sided 5%, 80% power). These are printed **before** the estimates.

## 6. Moderators (S8 and the follow-up)

| Concept | Proxy | Source |
|---|---|---|
| Litigation risk | Francis-Philbrick-Schipper industry indicator (SIC 2833-2836, 3570-3577, 3600-3674, 5200-5961, 7370) | SIC from the filings index |
| Litigation risk | Predicted litigation probability from size, return volatility, skewness and turnover (the CRSP-based part of Kim and Skinner, 2012) | CRSP |
| Detectability | Analyst coverage (number of I/B/E/S forecasters) and forecast dispersion | I/B/E/S |
| Detectability | Earnings quality (abnormal accruals) and accounting complexity | Compustat, if loaded |
| Monitoring | Institutional ownership and short interest | 13F, short-interest files, if loaded |

Fixed in advance: each is entered as a standardised moderator, interacted like $Z$, and reported with and without $Z$; a moderator that is not already in the cache is left for a follow-up and not purchased or approximated.

## 7. Interpretation rules

- Support for H5-S requires $\theta_3<0$ with $p<0.05$ on the primary regression. Anything else is "not supported at this power", reported with its minimum detectable effect.
- If $c$ is not positive, the Guidance Reversal pattern does not show up in sale-level timing at all, and $\theta_3$ is interpreted accordingly.
- The $T$ measurement limits of Stage 5 carry over (about half a point of noise, upward drift, 55% firm recognition).

## 8. Known limits fixed now

CEOs only; ex-post exit as treatment; overlapping 63-day windows from repeated sales (handled by firm clustering and S6); no 10b5-1 identification before 2023; sales sample defined by Form 4 name matching; $T$ attached from the same phase scores as Stage 5; the design was proposed after the Stage 5 results were seen, though no sale-level outcome had been computed.

## 9. Moderator follow-up (S8), fixed 2026-11-04 after the primary result and before any moderator was computed

**What the cache can support.** The cache holds industry codes (SIC for every firm) and CRSP daily returns, price and shares. It does not hold analyst counts or dispersion (the I/B/E/S files carry actuals and guidance only), trading volume (so no turnover), Compustat accruals or institutional ownership. Those proxies are therefore **not run** (not bought or approximated). The predicted litigation probability of Kim and Skinner (2012) cannot be estimated without securities-litigation outcomes and I do not use unverified coefficients, so it is not run either.

**Moderators run (two).**

| Moderator | Definition | Direction fixed now |
|---|---|---|
| $L$ | Litigation-prone industry (Francis, Philbrick and Schipper): SIC 2833-2836, 3570-3577, 3600-3674, 7370-7374, 5200-5961. These ranges are as commonly implemented and recalled by me, not checked against the paper; the Section 6 table above said 7370 only, and this entry widens it to 7370-7374 before anything was computed | Higher litigation exposure makes selling executives disclose more carefully (Cheng and Lo; Rogers), so a **smaller** avoided loss on guide-sell-leave sales: coefficient on $L \times G \times X$ **negative**, one-sided |
| $V$ | Return volatility: standard deviation of daily returns over the 252 trading days ending the day before the sale, standardised over the sample | **No directional prior** (volatility both hides bias and raises litigation exposure): two-sided |

**Specification.** The primary regression plus, for each moderator $M$, the terms $M$, $MG$, $MX$ and $MGX$, entered one at a time and then together. Reported: the coefficient on $M \times G \times X$, and whether the primary $\theta_3$ (on $Z \times G \times X$) changes once the moderator is in. These two tests are outside the family and cannot rescue a failed primary.
