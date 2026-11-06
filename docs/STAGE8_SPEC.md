# Stage 8: does opacity change what insiders avoid? The mechanism, on all officers and directors

Status: **written 2026-11-06, before any Stage 8 outcome was computed.** Author: Nayrb. Chosen by the author (2026-11-06) over a CFO-and-officer departure build, after sizing the options: new CEO sales in 2025 to 2026 are too few (about 1,900 sale-days from 465 CEOs, with many exits still pending), and departures for other officers cannot be built reliably from the Capital IQ text without name matching.

What I looked at before writing it: sale-day counts by role from the insider file (below), the list of role codes, and the structure of the link tables. No avoided loss was computed for any insider except the CEO sales already used in Stages 6 and 7.

## 1. What this tests, and what it does not

H5 says the Guidance Reversal pattern should hurt more where reporting is opaque. Stages 5 to 7 could not detect an effect on the full pattern (guided, sold, left) with 42 events or 583 to 4,176 sale-days. The closest literature (Duellman et al., 2025; Rogers and Stocken) tests a simpler ingredient with far more data: **are insiders' sales better timed where reporting is opaque?** Stage 8 tests that ingredient on all officers and directors.

**It drops the "leave" condition, so it does not test the Guidance Reversal pattern itself.** A null here with a small minimum detectable effect would say the opacity mechanism is absent in its simplest form, which makes a hidden guide-sell-leave effect at opaque firms unlikely. An effect here would not establish the pattern; it would say the mechanism exists and that the Stage 6 and 7 nulls were a power problem.

## 2. Unit and sample (fixed now)

- **Unit:** an insider-day: one row per person (`personid`) and issuer and calendar day with at least one open-market sale (Thomson Reuters Form 4, transaction code S), the value of sales summed (shares times price).
- **Insiders:** officers and directors, by the role code on the filing: CEO, CFO, O (officer), D (director), CB (chairman), GC (general counsel), DO and OD (director and officer), P (president), CO (chief operating officer), CT (chief technology officer). **Beneficial owners (B) and other holders and trusts are excluded**; they are not company insiders in the sense of H5. Sale-day counts before this spec (value of $100,000 or more, 2004 to 2025): officers (O) 111,721, directors 70,152, CEO 59,432, CFO 24,922, general counsel 15,376.
- **Value filter:** at least $100,000 per insider-day (robustness: $1 million and all values).
- **Window:** sale dates 2004-01-01 to 2024-12-31 (the cached price data end 2024-12-31).
- **Linking:** the issuer's eight-digit CUSIP (Thomson `cusip6` plus `cusip2`) to the CRSP permno using the name table **valid on the sale date** (a CUSIP can be reused), and the permno to the Compustat gvkey with the CRSP-Compustat link valid on that date (primary links only). Rows that do not link are dropped.
- Sale-days with no transparency score $T$ (no 10-K filed strictly before the sale, stale after 550 days, extraction failed, or not among the filings scored in Stage 5) or no CRSP returns are dropped; the share lost is reported.

## 3. Variables

- **Outcome $Y$:** the avoided loss, exactly as in Stage 6: $-100\times$ four-factor cumulative abnormal return over the next $h=63$ trading days, betas from the 252 trading days before the sale, winsorised at 1% and 99%; robustness $h=21, 126$.
- **$Z$:** standardised transparency score of the last 10-K filed strictly before the sale (as in Stage 5).
- **Controls $W$:** log market cap and 12-month return the day before the sale, log sale value, pre-sale return volatility $V$ (standardised), and role indicators (CEO, CFO, other officer, director, other).
- **$G$:** guidance in the 180 days strictly before the sale (as in Stage 6), used in a secondary test.

## 4. Primary test (one test, no family correction needed)

$$ Y_s = \lambda_{j(i)} + \tau_q + \theta\,Z_s + \gamma' W_s + u_s , $$

with **SIC two-digit industry effects $\lambda_j$ and year-quarter effects $\tau_q$** (year-quarter rather than year-month because the sample is several times larger and the month dummies would make the regression impractical), standard errors clustered by firm and by quarter. **H5-mechanism: $\theta<0$** (the avoided loss is larger where transparency is low), one-sided, $p<0.05$. The project-wide count of formal tests becomes 36.

**Power before the estimate.** The standard error of $\theta$, the minimum detectable effect (about 2.49 times the standard error) and a firm-cluster bootstrap that plants an effect $-\delta Z_s$ into every sale-day are printed first.

## 5. Secondary analyses (outside the test count; none can rescue a failed primary)

| | |
|---|---|
| S1 | Non-routine sales only (the routine rule of Stage 7, applied per person), and opportunistic sales only |
| S2 | By role: CEO, CFO, other officers, directors, separately |
| S3 | $Z \times G$: does the effect of $Z$ differ after guidance (H5 predicts a more negative coefficient on $Z\times G$) |
| S4 | Horizons 21 and 126 trading days |
| S5 | High-transparency dummy ($T \ge 3$) in place of $Z$ |
| S6 | Firm fixed effects in place of industry effects |
| S7 | Value filters of $1 million and of any value |
| S8 | First insider-day per person and quarter |
| S9 | Small and large firms (below and above the median market cap), separately |

## 6. Interpretation rules

- Support requires $\theta<0$ with one-sided $p<0.05$ in the primary test. Anything else is "not supported", reported with its minimum detectable effect.
- Because the literature suggests that opacity raises insider returns (Duellman et al.), a negative $\theta$ is the expected sign and a positive one would be a surprise worth reporting as such.
- Even a significant $\theta$ is a statement about insider sales in general, not about guide-sell-leave sales.
- The conclusion for H5 is read together with Stages 5 to 7: if the mechanism is absent here with a small detectable effect, there is little room for an opacity effect specific to the pattern.

## 7. Known limits fixed now

Sales include scheduled (10b5-1) and tax-related sales, which cannot be separated before 2023 (the routine rule is the only filter); CUSIP and gvkey links can be wrong (date-valid links reduce but do not remove this); $T$ is missing for firm-quarters outside the scored set (about a quarter of sale-days in Stage 6), which leans the sample toward guiding firms; the effect is identified from differences between firms, which industry effects and the controls limit but do not remove; many insider-days at one firm in one quarter are correlated (firm and quarter clustering); the Stage 5 measurement limits of $T$ carry over.

## 8. Results (added after the run; Sections 1 to 7 above were not changed)

Run 2026-11-06 with `scripts/run_stage8.py` (7 minutes, 12 power replications). Output: `outputs/results/stage8_results.json`.

- **Sample.** 300,936 insider-days of at least $100,000 by officers and directors linked to a permno and gvkey; 297,902 with an avoided loss; **205,958 with a transparency score** (primary sample), from 2,132 firms and 28,801 people. 88% of sale-days follow guidance. Regression n is 204,174 after dropping missing industry or volatility. T is missing for 84,854 insider-days (not among the scored filings), 5,343 (no prior filing), 1,469 (extraction failed) and 278 (stale).
- **Power (printed before the estimates).** Standard error of Z 0.152; minimum detectable effect 0.38 pp per sd of T. Simulated rejection at planted 0, 0.19, 0.38, 0.76: 8%, 50%, 83%, 100%.
- **Primary.** Z = -0.129 (se 0.152), one-sided p 0.199, two-sided 0.399. **Not supported.** The sign is the predicted one.
- **Secondary (outside the test count).** Non-routine -0.128 (p 0.213); opportunistic +0.230 (p 0.897); CEO +0.038; CFO +0.091; other officers -0.258 (p 0.052); directors -0.088; Z x G -0.168; h=21 -0.064 (p 0.11); h=126 -0.020; high-T dummy -0.128; firm FE -0.495 (p 0.153); sales >= $1M -0.268 (p 0.099); first per person-quarter -0.105; small firms -0.071; large firms -0.265 (p 0.055). None is significant at two-sided 5% and none is a finding.
- **Not run.** The all-values robustness named in Section 5 (S7) was not part of the script. The project-wide count of formal tests is 36.
- **Reading under Section 6.** Not supported, with a minimum detectable effect of 0.38. This is a statement about insider sales in general, not about guide-sell-leave.
