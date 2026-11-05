# Guidance Reversal: do executives who guide, sell and leave leave behind underperforming firms?

**Date:** 2026-11-04 (rebuilt from the chronological memo of 2026-10-24, kept as `docs/MEMO_chronological_2026-10-24.md`)
**Data:** WRDS (ExecuComp, CRSP including delisting returns, I/B/E/S guidance, Thomson Reuters Form 4, Capital IQ text, Fama-French factors, SEC filings index); S&P 1500 CEO and CFO departures announced 2004 to 2025.
**This file is generated** by `scripts/build_memo.py` from `docs/MEMO_TEMPLATE.md` and `outputs/results/*.json`; every number in the tables comes straight from the result files. Edit the template, not this file.

## Summary

Across seven stages of tests, no version of the Guidance Reversal pattern predicts underperformance, earnings misses or a higher chance of leaving, and the pattern is not stronger where a firm's reporting is opaque. **These are mostly underpowered nulls, not proofs of absence:** the strict definition has only 42 usable events, and the tests could reliably detect a risk-adjusted return of about -20% a year, not -5% to -10%. The wider tests rule out a large effect and no more. Sections 5.1 to 5.7 give the results and Section 5.8 discusses what they do and do not show. Stage 6, which tests the opacity hypothesis on 4,176 individual guide-sell-leave CEO sales instead of 42 events, is the best-powered test and is also null; a Stage 7 re-analysis (industry fixed effects, abrupt exits, non-routine sales) did not raise power and is null as well.

---

# 1. Research hypothesis

## 1.1 The question

**Do executives who issue earnings guidance, sell stock and then leave leave behind firms that underperform?** The pattern is called **Guidance Reversal**. A chief executive or chief financial officer guides the market to an earnings number, sells shares after the guidance, and leaves before the guided period ends; the firm then does badly.

## 1.2 Motivation

Earnings guidance is the main way a firm sets investors' expectations between reports. If an executive who knows the guidance is too optimistic uses it to support the share price while selling, and leaves before the shortfall arrives, then outside investors who rely on the guidance pay for it. That matters for three audiences. Investors could use the pattern as a red flag, or as a short signal if it predicts underperformance. Boards and regulators care because it would suggest that executive pay and trading rules let people profit from guidance they do not stand behind. And it is a clean test of whether insider trading data and departure announcements contain information the market does not price in advance.

The project does not test intent: it tests whether the observable pattern is followed by bad outcomes.

## 1.3 Hypotheses, by stage

| Stage | Hypothesis | What would support it |
|---|---|---|
| 1 | **H1.** Firms whose departing CEO or CFO (1) issued guidance in the 180 days before the announcement, (2) sold shares between the latest guidance and the announcement and (3) left abruptly earn negative risk-adjusted returns over the next 12 months | A negative four-factor alpha, beyond a placebo distribution |
| 2 | **H2.** The same, with abruptness dropped, and more so when the guidance was optimistic (above consensus), which proxies for intent | Negative alpha overall, and a more negative alpha for optimistic guidance |
| 3 | **H3.** Among executives who guided and sold, leaving matters: (P1) optimistic guidance followed by a sale raises the chance of leaving; (P2) optimistic guidance, a sale and an exit are followed by a worse earnings miss; (P3) firms where the executive left underperform firms where the executive stayed | Positive P1 interaction, negative P2 interaction, negative P3 difference |
| 4 | Exploration: is there anything in the pattern at all, found on 2004 to 2014 and confirmed on 2015 to 2025? | A selected effect that keeps its sign and size in the hold-out |
| 5 | **H5.** The pattern is stronger where the firm's reporting is opaque. Where outsiders can rebuild a firm's profit as price times quantity (high transparency), they can check guidance against the business and an executive has little room to mislead. Where reporting is opaque, the same pattern should hurt more | Each Stage 1 to 3 effect more negative (in the direction of H1 to H3) at low transparency than at high |
| 6 | **H5-S.** The same H5, tested on individual CEO insider sales: after a guide-sell-leave sale, the stock's later abnormal return is lower (the sale avoided a loss) where transparency is low | A negative coefficient on transparency interacted with guided and left, on the avoided loss after the sale |
| 7 | **H5-S again**, re-tested on the same sales with industry-by-month fixed effects (lever 1) and then with abrupt exits and non-routine sales only (lever 3) | The same negative coefficient on transparency interacted with guided and left, in either of two primary tests |

## 1.4 What counts as evidence, and how to read a null

A test supports a hypothesis only if its estimate has the predicted sign and clears a multiple-testing correction. A null is informative only up to the effect size the test could have detected. Every table in Section 5 therefore reports the smallest effect the design could detect (the minimum detectable effect) next to the estimate. Section 5.8 returns to this.

---

# 2. Related literature

No paper I found tests the pattern studied here (guidance, then a sale, then a departure) or scores a firm's transparency by whether its profit can be rebuilt as price times quantity. The work below bears on one piece each. It is drawn from a quick search of abstracts, not a full review; findings marked (abstract) were read only at that level and should be checked before citing.

## 2.1 Do insiders use voluntary disclosure to trade?

- **Noe (1999)**, *Journal of Accounting and Economics* 27(3), and **Cheng and Lo (2006)**, *Journal of Accounting Research* 44(5). Managers time disclosure around their own trades. Cheng and Lo find that managers planning to *buy* issue more bad-news forecasts beforehand, but find no comparable forecast adjustment before *sales*, which they attribute to higher litigation concerns (abstract).
- **Rogers (2008)**, *Journal of Accounting Research* 46(5). Managers appear to provide *higher*-quality disclosure before selling shares, which the author links to reducing litigation risk, and weaker evidence of lower-quality disclosure before purchases (abstract).
- **Rogers and Stocken (2005)**, *Credibility of Management Forecasts* (SSRN). Managers with more litigation exposure issue less optimistic forecasts, but this weakens when bias is harder for outsiders to detect, and then forecasts are more likely to raise insider trading profits (secondary summary).

*Relevance.* These papers say that *sales are the legally risky side*: executives who sell tend to disclose carefully, and bias is more likely where it is hard to detect. That is the mechanism behind H5 (detectability) and the reason litigation risk is a natural moderator (Section 4.7 and the Stage 6 design).

## 2.2 Does opacity let insiders hide bad news?

- **Jin and Myers (2006)**, *Journal of Financial Economics* 79, and **Hutton, Marcus and Tehranian (2009)**, *Journal of Financial Economics* 94(1). Opaque firms, with opacity measured by earnings management, have higher R-squared and are more prone to price crashes, as the Jin and Myers model predicts (HMT abstract). The relation **dissipated after the Sarbanes-Oxley Act** (HMT abstract).
- **Hope and Thomas (2008)**, *Journal of Accounting Research* 46(3). Firms that stopped disclosing geographic earnings after SFAS 131 expanded foreign sales more, earned lower foreign margins and had lower firm value, consistent with an agency-cost view of non-disclosure (abstract).

*Relevance.* This is the theoretical home of H5: opacity helps managers conceal what is going wrong, and segment-level non-disclosure is associated with weaker outcomes. The Sarbanes-Oxley result matters for our prior: **our whole sample (2004 to 2025) is post-Sarbanes-Oxley**, when the opacity effect on crash risk had already faded in the one study that tested it. That lowers the prior that Stage 5 would find a strong effect.

## 2.3 Does reporting opacity help insiders trade?

- **Duellman et al. (2025)**, *Journal of Business Finance and Accounting*. Insider trades in the 60 days after a filing earn higher returns as accounting reporting complexity rises (abstract).
- **Cohen, Malloy and Pomorski (2012)**, *Journal of Finance* 67(3), *Decoding Inside Information*. Insider trades with a predictable timing pattern ("routine") carry no information; the remaining "opportunistic" trades earn abnormal returns of 82 basis points a month and predict firm news and the returns around analyst forecasts, management forecasts and earnings announcements (abstract). Stage 7 uses their routine-versus-opportunistic idea.
- **Journal of Corporate Finance (2026)**, internal information asymmetry and insider trading. After SFAS 131 required more segment detail, divisional managers purchased more often and earned larger profits, so tighter external segment reporting did *not* remove the insider advantage (abstract).
- **Beneish and Vargus (2002)**, *The Accounting Review*. Income-increasing accruals are less persistent in firms with high insider selling; **Agrawal and Cooper (2015)**, *Journal of Corporate Finance*. Across 500-plus earnings-decreasing restatements, evidence that top managers sold more during the misstated period is weak overall but substantial where incentives to sell were stronger (abstracts).

*Relevance.* Duellman et al. is the closest to the Stage 6 design below, because it tests opacity against **insider trade returns** rather than against firm returns after a rare event. Its trade-level design is the main reason Stage 6 exists.

## 2.4 Measuring litigation risk and detectability

- **Francis, Philbrick and Schipper (1994)**, *Journal of Accounting Research* 32(2). The industry proxy for litigation risk: biotechnology, computers, electronics and retail firms.
- **Kim and Skinner (2012)**, *Journal of Accounting and Economics* 53(1). The industry indicator alone predicts securities litigation poorly; adding size, growth and stock volatility improves prediction considerably (abstract). Section 4.7 uses their idea of a predicted litigation probability.

## 2.5 Long-run returns after events

- **Fama (1998)** and **Mitchell and Stafford (2000)**, *Journal of Business* 73(3). Buy-and-hold tests assume independent returns that event firms clustered in time do not have, which inflates test statistics; calendar-time portfolios fix this. **Loughran and Ritter** reply that calendar-time regressions have low power. Our Stage 1 follows the calendar-time convention, and the power criticism applies to us.

## 2.6 Measuring firm characteristics with language models, and look-ahead bias

- **Glasserman and Lin (2023)** and **Sarkar and Vafa (2024)** show that language models used in finance can carry knowledge of what happened after the text was written. Glasserman and Lin find that anonymising company names did not remove the effect they feared (the anonymised in-sample results were better, which they attribute to a "distraction" effect from knowing the firm); Sarkar and Vafa find lookahead bias in two social-science applications and argue that models trained only on pre-period text are the clean remedy (abstracts).
- **Liu, Cheng and Lai (2026)**, arXiv 2605.23924, extract segment disclosures from 10-Ks with a language model; the abstract reports no sample or accuracy figures.

*Relevance.* Our recognition test (the scorer named the firm in 55% of name-stripped bundles) is the same concern. It is a limitation of Stage 5, and a reason to prefer measures of transparency that do not depend on a model's knowledge of the firm.

## 2.7 What is new here

(1) The joint pattern of guidance, an insider sale and a departure, tested with a stayer control group, a panel and a placebo design. (2) A transparency score built from whether the profit engine can be rebuilt as price times quantity, with phases, a frozen scorer and a hand audit. (3) Stage 6: opacity tested at the level of the **individual insider sale** for the pattern, rather than only at the level of rare flagged events. A quick search cannot establish novelty; the claim is that I did not find these elsewhere.

---

# 3. Data

## 3.1 Sources

| Source (WRDS) | What it provides | Used for |
|---|---|---|
| ExecuComp | Executives, roles (CEO or CFO), tenure, age, departure dates, exit reason | Population of departures, CEO-quarter panel, controls |
| CRSP daily returns, delisting returns, names, CRSP-Compustat link | Daily stock returns (delisting returns included), market cap, identifier links | Portfolios and alphas, size controls, size-matched placebos |
| Fama-French daily factors | Mkt-RF, SMB, HML, UMD (and a six-factor set) | Risk adjustment |
| I/B/E/S guidance, actuals and consensus | EPS guidance releases, realised EPS, analyst consensus | Recent guidance, optimism, earnings miss |
| Thomson Reuters Form 4 | Insider open-market sales | The "sale" condition |
| Capital IQ Key Developments | Executive-change announcements and text | Announcement date, whether a departure was abrupt, exit type |
| WRDS SEC filings index and EDGAR documents (Stage 5) | 10-K, 10-Q and 8-K filings and their text | The transparency score $T$ |

Sample: S&P 1500 firms, departures announced 2004 to 2025. Point-in-time index membership could not be verified, so membership is taken from the constituent history as available (a limitation).

## 3.2 Definitions

- **Departure and announcement date.** An ExecuComp CEO or CFO departure; the announcement date is the earliest Capital IQ executive-change item in which that executive is the person leaving. Where blind annotators identified the item, their choice overrides the rule-based one.
- **Recent guidance.** An I/B/E/S EPS guidance release in the 180 days before the announcement (365 days in the Stage 2 variants).
- **Sale.** A Thomson Form 4 open-market sale (transaction code S, disposed) by an owner matching the executive by last name and full first name or nickname, dated from the most recent guidance to the announcement. In the Stage 3 panel, any open-market sale counts, with no size threshold.
- **Abrupt.** Judged from the announcement text by independent annotators who never see returns, using a written rubric (effective immediately or within about 30 days with no orderly transition; terminated or forced out; resignation without announced succession; planned retirements and succession plans are not abrupt). Every flagged candidate was read twice (94% agreement).
- **Flagged event (Guidance Reversal).** A CEO or CFO departure that meets all of: recent guidance, a sale after that guidance, an abrupt exit. 42 events have usable return windows.
- **Optimistic guidance, $O$.** 1 if the guided EPS midpoint is above analyst consensus at the guidance date, else 0 (continuous version: (midpoint $-$ consensus) / price, standardised).
- **Sale, $S$.** 1 if the executive sold after the latest guidance (up to the quarter start in the CEO-quarter panel).
- **Exit, $X$.** 1 if the executive left before the guided period ended. In the CEO-quarter panel, $\text{Exit}_{i,q}$ is 1 if CEO $i$ leaves in quarter $q$, times 100.
- **Earnings miss.** (actual EPS $-$ guided midpoint) / share price, in percent of price, winsorised at 1% and 99%.
- **Stayer.** A CEO who guided and sold since the latest guidance but did not leave; the stayer is dated at a pseudo-event at the quarter start so it can be compared with leavers.

## 3.3 Samples and panels

| Object | Size |
|---|---|
| CEO and CFO departures with data, 2004 to 2025 | 5,163 (3,336 eligible for the Stage 2 universe) |
| Flagged events (strict flag) with usable windows | 42 |
| Guidance and sale and any exit (Stage 2 universe U1, L1 link) | 235; wider links reach 1,114 |
| CEO-quarter panel (Panel A) | 77,414 CEO-quarters, 4,648 CEOs, 2,290 firms, **1,943 exits** |
| Guidance-release panel (Panel B) | 98,169 releases; 1,006 with a sale and an exit |
| Stayer events and leavers for the leaver-versus-stayer test | 5,388 stayer events (1,981 executives); 157 leavers |

## 3.4 Measurement validation, and what was rebuilt

An earlier version of the project produced a striking gap (-21% against -4% abnormal return) that turned out to be an artefact. The measurement was rebuilt and validated:

| Problem in the earlier version | What it did to the results | Fix |
|---|---|---|
| WRDS REST API silently cut off large downloads | About half of all firms had no price history, so every return rested on half the sample | Pull via Postgres (10.2M against 5.1M CRSP rows in the 2004 pull) |
| Old "abrupt" rule | 23% precision against hand labels; called 57% of departures abrupt (true rate about 21%) | Sentence-level classifier tested on 280 hand labels; every flagged event read twice by independent annotators |
| Old "government destination" label | 0 of 40 sampled cases were real | Removed |
| 12% of executive names mis-parsed ("J.D." read as surname "D") | Broke insider and text matching | Fixed and unit-tested |
| Flag required a later bad outcome | The gap was circular | Flag uses only announcement-date information |
| Guidance and actuals on different split bases | "Misses" of up to +940% of price | Rebased with CRSP split factors |

## 3.5 Stage 5 data: the transparency score $T$

**$T$ is one integer from 0 to 5 per 10-K filing; higher means more transparent.** It answers one question: *could an outside investor rebuild what the firm earns as price times quantity, for the parts of the business that make most of the profit?* A price or quantity counts as disclosed if it is stated or can be derived by dividing two disclosed figures for the same line and period.

| $T$ | Meaning |
|---|---|
| 5 | Volume and price (stated or derivable) for every line that makes the profit |
| 4 | Volume and price for about three quarters of the profit; one meaningful line is dollars only |
| 3 | About half the profit covered, or volume or price (not both) for the main engine |
| 2 | Dollars only: revenue and profit by line, no volume or price for the main engine |
| 1 | Almost opaque: total or regional revenue, drivers only in words |
| 0 | One revenue number, no breakdown |

A razor-and-blade firm (price and quantity only for the low-margin line, dollars only for the high-margin recurring line) is capped at 2; so is a firm where the source of profit cannot be told. The number of segments does not matter.

- **Materials read.** The 10-K (Item 1, the segment note and MD&A segment text, operating-statistics text) plus the 10-Qs and the earnings-release exhibits attached to 8-Ks filed in the 12 months up to that 10-K, at most about 72,000 characters. Company names are stripped. Website-only investor decks and call transcripts are not included, because they cannot be dated or fetched reproducibly.
- **Scorer.** Claude Haiku through the Claude Code command line, with the model, prompt, rubric version and command-line flags hashed into a frozen configuration that is checked before every run.
- **Phases.** A firm rarely changes how it reports revenue. Each firm's filings are grouped into **phases**: a new phase starts when the revenue breakdown changes drastically (found by a fuzzy comparison of revenue-table labels, then confirmed by a model check of the two label lists). The first filing of each phase is scored once and its score is carried across the phase. The 24,154 filings fall into 4,341 phases (1,525 candidate changes were triaged); 99.2% of filings carry a score.
- **Timing, no look-ahead.** For a departure event, $T$ comes from the last 10-K filed strictly before the announcement date the return windows start from; for a CEO-quarter, the last 10-K filed on or before the quarter start; for a guidance release, strictly before the release date. A filing more than 550 days old counts as missing.
- **Distribution.** Over phases: $T=0$: 6; 1: 300; 2: 1,843; 3: 1,018; 4: 718; 5: 417 (mean 2.79, standard deviation 1.11).
- **Reliability.** Weighted kappa between two reads of the same filing 0.94 (within one point 100%) on a 199-filing pilot. Against my own hand scores on 100 random filings: kappa 0.77 (95% interval 0.67 to 0.85; the gate was 0.60) and 92% within one point. The model scores 0.5 points above my scores on average (it credits "price up 37%, units down 12%" and customer counts as price times quantity): 26 fives against my 10. The first and last filing of a phase differ by 0.5 points on average (kappa 0.68), which is noise. In a recognition test the model named the firm in 55% of 58 name-stripped bundles.

---

# 4. Methodology

## 4.1 Design at a glance

| Stage | Question | Design | Why this design |
|---|---|---|---|
| 1 | Do flagged firms underperform? | Calendar-time four-factor alpha, placebo-portfolio p-value | Overlapping 12-month windows make event-time returns dependent; a calendar-time portfolio handles this. A placebo test is needed because only about two firms are held at a time |
| 2 | Same, wider links, optimism split | Same, 14 tests, Holm correction | Abruptness cut 286 candidates to 45 without a substantive reason; intent proxied by optimism |
| 3 | Is there a control group effect? | CEO-quarter and guidance-release panels, firm and year-quarter fixed effects, two-way clustering | Adds stayers as controls, and measures the mechanism (exit, miss) directly |
| 4 | Is there anything in the pattern? | Explore on 2004 to 2014, lock a selection rule, confirm on 2015 to 2025 | A hold-out limits fishing |
| 5 | Does opacity change the answer? | Tests 1 to 4: the Stage 1 and 3 designs with the transparency score $T$ | Same hypotheses, split by $T$ |
| 6 | Does opacity change what insiders avoid? | Sale-level avoided loss after CEO sales, with transparency interacted with guided and left | Uses thousands of sales instead of 42 events, as the closest literature does |
| 7 | Do between-firm variation and the original treatment add power? | Stage 6 regression with industry-by-month effects (P7a), then abrupt exits and non-routine sales (P7b) | Tests whether the Stage 6 null came from the design |

## 4.2 Stage 1: calendar-time alpha with a placebo test

Hold each flagged firm from trading day +2 to +253 after the announcement (the first two days are the announcement reaction), form an equal-weighted portfolio each day from the open positions (delisting returns included), and regress its excess return on the four Carhart factors:

$$ r_{p,t} - r^{f}_{t} \;=\; \alpha + \beta_1\,(r^{m}_t - r^{f}_t) + \beta_2\,\mathrm{SMB}_t + \beta_3\,\mathrm{HML}_t + \beta_4\,\mathrm{UMD}_t + \varepsilon_t . $$

$\alpha$ is the test: a negative alpha is underperformance beyond risk. A conventional t-test understates the noise when about two firms are held at a time, so the p-value is the share of 2,000 random same-size groups of other departures (matched on announcement-year counts) whose alpha is at least as low. The test was written down before returns were computed (`docs/PREREGISTRATION.md`). Eight secondary tests (CEOs only, a 6-month hold, looser insider matching, a keyword abruptness rule, a six-factor model, dropping 2008 to 2009 and 2020, two sub-periods) are corrected with Holm's method.

## 4.3 Stage 2: wider links and the optimism split

Stage 2 drops abruptness: any CEO or CFO exit other than death where an insider sale is linked to guidance. Links: **L1** guidance within 180 days and a sale after the latest guidance; **L2** guidance within 365 days and a sale after it; **L3** guidance within 365 days and any sale in the prior 365 days; **L4** any sale in the prior 365 days. Universes: **U1** all such events; **U2** departures whose role is CEO or CFO. Family F1 is the alpha of each of the 8 link-by-universe cells against the placebo. Family F2 splits each of 6 cells by optimistic versus other guidance and tests the alpha difference by a within-year permutation. Holm covers all 14.

## 4.4 Stage 3: panel regressions with stayers as controls

All three tests use firm and year-quarter fixed effects ($\lambda_i$, $\tau_q$) and standard errors clustered two ways, by firm and by quarter. Fixed effects remove persistent firm traits and market-wide shocks; two-way clustering allows for correlation across firms within a quarter and within a firm over time. Covariates are measured at the start of the period (or the day before guidance), so none uses information from after the event.

**P1, exits.** Is a CEO more likely to leave in a quarter when the latest guidance was optimistic and they sold after it? On CEO-quarters with guidance in the previous 365 days:

$$ \text{Exit}_{i,q} = \lambda_i + \tau_q + c_1\,O + c_2\,S + c_3\,OS + \gamma_1\,\text{Sale12} + \gamma'W_{i,q} + u_{i,q}, \qquad H_3:\ c_3 > 0 . $$

**P2, earnings misses.** Is the realised miss worse when optimism, a sale and an exit all occur? On guidance releases:

$$ \text{Miss}_{j} = \lambda_i + \tau_q + \textstyle\sum_{m} b_m\, m_j + \gamma'W_j + u_j, \qquad m \in \{O,S,X,OS,OX,SX,OSX\}, \qquad H_3:\ b_{OSX} < 0 . $$

The triple interaction is used because $O$ alone is negative almost mechanically (guide above the street, land near consensus, record a "miss"), and $X$ and $OX$ absorb a new CEO's write-offs.

**P3, leavers versus stayers.** Is the alpha of firms where the executive guided, sold and **left** lower than where the executive guided, sold and **stayed**? Calendar-time four-factor alphas for the two groups, difference tested by permuting the leaver label within announcement year.

**Controls $W$.** Log market cap and the 12-month return (size and momentum or deterioration), log tenure and age (retirement and new-CEO effects). P1 adds a dummy for any sale in the prior 12 months; P2 adds log days to the period end (a longer horizon allows a bigger miss) and a quarterly-release dummy. Seventeen robustness checks vary the optimism definition, sale size, outcome, sample and timing.

## 4.5 Stage 4: explore, then confirm

Explore on 2004 to 2014: test whether the sign, missing controls (age bands, last miss, guidance style, volatility, ownership changes) or exit type rescue the pattern; fix a selection rule; then test only the locked candidates once on 2015 to 2025.

## 4.6 Stage 5: the same tests, split by transparency $T$

$Z=(T-\mu)/\sigma$ is $T$ standardised over all scored filings. **Low $T$** is $T \le 2$ and **high $T$** is $T \ge 3$: the groups are split at the 33rd and 67th percentile of $T$ over all departure events, and because $T$ takes only six values the cut points are 2 and 3, leaving no middle group. H5 predicts a more negative outcome at low $T$ in every test.

**Test 1 (T1), returns.** Stage 1 by group $g \in \{\text{low},\text{high}\}$: $r_{g,t}-r^f_t=\alpha_g+\beta' F_t+\varepsilon_{g,t}$. Each group's alpha is compared with random placebo groups of the same size drawn from non-flagged events in the same $T$ group (announcement identified), giving an excess alpha, and the test is the difference in excess alpha:

$$ \Delta = \big(\hat\alpha_{\text{low}} - \bar\alpha^{\text{placebo}}_{\text{low}}\big) - \big(\hat\alpha_{\text{high}} - \bar\alpha^{\text{placebo}}_{\text{high}}\big), \qquad H_5:\ \Delta < 0 , $$

with the lower-tail share of centred placebo contrasts as the p-value, a calendar-quarter block bootstrap for the interval, and the calendar-time low-minus-high alpha spread as a secondary statistic.

**Test 2 (T2), exits.** P1 with $Z$ interacted with every term:

$$ \text{Exit}_{i,q} = \lambda_i + \tau_q + c_1 O + c_2 S + c_3\,OS + \theta_0 Z + \theta_1 ZO + \theta_2 ZS + \theta_3\,ZOS + \gamma'W_{i,q} + u_{i,q}, \qquad H_5:\ \theta_3 < 0 . $$

Since $c_3>0$ is the Stage 3 pattern, a pattern that is stronger where $T$ is low means $\theta_3<0$.

**Test 3 (T3), misses.** P2 with $Z$ interacted with all seven terms:

$$ \text{Miss}_{j} = \lambda_i + \tau_q + \textstyle\sum_{m} b_m\, m_j + \theta_0 Z + \sum_{m} \theta_m\, Z\, m_j + \gamma'W_j + u_j, \qquad H_5:\ \theta_{OSX} > 0 , $$

because Stage 3 predicts $b_{OSX}<0$ and "stronger at low $T$" then means a positive $\theta_{OSX}$.

**Test 4 (T4), leavers versus stayers.** On the low-$T$ and high-$T$ events, with $\hat\alpha_e$ the annualised four-factor alpha of event $e$'s own 252-day window:

$$ \hat\alpha_e = a + b\,\text{Leaver}_e + c\,\text{High}_e + d\,(\text{Leaver}_e \times \text{High}_e) + u_e, \qquad H_5:\ d > 0 , $$

so a positive $d$ means leaving hurts less where $T$ is high. The p-value is a wild cluster bootstrap by announcement quarter that imposes $d=0$ only, so an equal non-zero leaver effect in both groups is the null.

## 4.7 Stage 6: H5 at the level of the individual insider sale

**Why.** Stage 5 tested H5 on 42 flagged events and could detect only very large effects. The closest literature (Section 2.3) tests opacity against the returns that follow *individual insider trades*, which uses thousands of observations. Stage 6 asks the same H5 question that way. The specification (`docs/STAGE6_SPEC.md`) was written before any sale-level return was computed.

**Unit and sample.** A CEO sale-day: one row per CEO episode and calendar day with at least one open-market sale, sales summed, value at least $100,000, sale dates 2004 to 2024, excluding CEOs who die within 365 days. Sales that follow a score-less filing, or that have no CRSP returns, drop out.

**Outcome: the avoided loss.** After each sale $s$ at date $t$:

$$ Y_s = -100 \times \text{CAR}_s(+1,+h), \qquad \text{CAR}_s(+1,+h) = \sum_{d=t+1}^{t+h}\Big( r_d - r^f_d - \hat\beta_s' F_d \Big), $$

with $F$ the four Carhart factors and $\hat\beta_s$ estimated by OLS on the 252 trading days ending the day before the sale (at least 120 observations), so no post-sale information enters the benchmark. Delisting returns are included. $Y_s$ is positive when the stock does worse than its risk-predicted return after the sale. Primary $h=63$ trading days (about one quarter); robustness $h=21, 126, 252$.

**Treatment.** $G_s=1$ if the firm issued guidance in the 180 days before the sale; $O_s=1$ if that guidance was above consensus; $X_s=1$ if the CEO leaves within 365 days after the sale (ex-post, as in Stage 3 P2); $Z_s$ is the standardised transparency score of the last 10-K filed strictly before the sale.

**Primary regression.**

$$ Y_s = \lambda_i + \tau_m + a\,G_s + b\,X_s + c\,(G_sX_s) + \theta_0 Z_s + \theta_1 Z_sG_s + \theta_2 Z_sX_s + \theta_3\,Z_sG_sX_s + \gamma' W_s + u_s , $$

with firm effects $\lambda_i$, year-month effects $\tau_m$, controls $W_s$ (log market cap, 12-month return, log sale value, log tenure, age) and standard errors clustered by firm and by month. **H5-S: $\theta_3<0$**: the guide-sell-leave sale avoids more loss where transparency is low. A positive $c$ would mean such sales avoid losses on average (the Stage 3 pattern at sale level). The family size is one.

**Secondary specifications** (outside the family): the avoided loss on $Z$ alone for all sales (the literature's own test); optimistic guidance $O$ in place of $G$; a high-$T$ dummy in place of $Z$; the other horizons; sales of at least $1 million and all sales; the first sale-day per CEO-quarter (removes overlapping windows); the two sub-periods. The follow-up with litigation-prone industry and pre-sale return volatility as moderators (spec section 9) was run on the saved panel; the other detectability proxies need data that are not in the cache.

**Power, printed before the estimates.** The minimum detectable $\theta_3$ is about $2.49$ times its standard error (one-sided 5%, 80% power). A firm-cluster bootstrap plants effects of $-\delta Z_s$ into guide-sell-leave sales and reports how often the primary test rejects at several $\delta$.

## 4.8 Stage 7: between-firm variation, abrupt exits and routine sales

**Why.** After the Stage 6 null I asked what could add power without changing the question. Two changes were tested in a specification fixed before any Stage 7 regression was run (`docs/STAGE7_SPEC.md`). It is an **exploratory re-analysis**: it uses the same CEO sales as Stage 6, so a significant result would need replication on data not used here.

1. **Industry-by-month fixed effects instead of firm effects (lever 1).** Transparency barely changes within a firm (firm effects remove 95% of the variation in $Z$ itself), and H5 is a statement about opaque firms against transparent ones. The spec records my own correction: this clearly matters for $Z$ on its own, but the H5 term is identified from differences across firms in the effect of a guide-sell-leave sale, so the gain for it was uncertain and was to be read from the printed standard error.
2. **The original treatment (lever 3).** H1 is about executives who leave **abruptly**, and the literature finds routine insider sales uninformative and opportunistic ones informative (Section 2.3).

**Definitions.** An **abrupt exit** is a departure event announced strictly after the sale and within 365 days that the annotators' label (where one exists) or otherwise the rule-based classifier of the event file calls abrupt; **planned or other** is the complement among classified exits, and sale-days whose exit is announced but unclassified are excluded from the abrupt and placebo tests. A **routine sale** follows the rule of Cohen, Malloy and Pomorski (2012) as specified in the spec: the CEO sold in the same calendar month in each of the three previous calendar years; **opportunistic** means the CEO sold in each of the three previous years but not in that month in all three; **unclassified** means fewer than three prior years of sales. **Non-routine** means not routine, because a first-time or irregular seller cannot have an established pattern. Only years before the sale year are used.

**Primary tests (two, Holm).** Both use SIC two-digit industry and year-month fixed effects, standard errors clustered by firm and month, and the Stage 6 controls plus pre-sale volatility $V_s$:

$$ Y_s = \lambda_{j(i)} + \tau_m + a\,G_s + b\,X_s + c\,(G_sX_s) + \theta_0 Z_s + \theta_1 Z_sG_s + \theta_2 Z_sX_s + \theta_3\,Z_sG_sX_s + \gamma' W_s + \delta V_s + u_s . $$

**P7a** uses the Stage 6 exit variable and sample, so it isolates lever 1. **P7b** uses an abrupt exit and non-routine sale-days, so it adds lever 3. **H5-S: $\theta_3<0$**, one-sided. The project-wide count of formal tests becomes 35.

**Secondary (outside the family).** The same regressions with firm effects; a **placebo** in which $X$ is a planned or other exit; P7b on all sales and on opportunistic sales only; optimistic guidance $O$ in place of $G$; horizons of 21 and 126 trading days; the first sale-day per CEO-quarter; and SIC divisions in place of two-digit industries. The standard error, minimum detectable effect and a firm-cluster bootstrap power simulation are printed for both primary tests before any estimate.

## 4.9 Inference and safeguards

- **Multiple testing.** Holm's method within each stage's family (Stage 1: 8 secondary tests; Stage 2: 14; Stage 3: 3; Stage 5: 4). The project-wide count of formal tests is 32.
- **Null calibration.** Stage 3 permuted the outcome within year-quarter on the real panels and re-ran P1 and P2 (60 draws): the tests were mildly liberal, so a null is if anything firmer than its p-value. Stage 5 ran the whole pipeline with $T$ shuffled (a dry run) before the real run.
- **Look-ahead.** Each regression's covariates and each $T$ use only information dated before the event; the Stage 3 run asserts these timing rules in code, and the Stage 5 run refuses to start unless the scoring configuration is frozen, the reliability gate passed, and every score matches the text it was made on.
- **Power.** Each result is reported with its minimum detectable effect.
- **Independent review.** A read-only code review of Stage 5 found four issues (including dating $T$ in Test 1 by a different announcement date from the one the return windows use); all were fixed test-first and the results moved by no more than 0.004 in any p-value.

## 4.10 How the code base fits together

Everything runs from the repository root. The data flow is: **WRDS pulls and features** $\to$ **validated labels** $\to$ **Stage 1 to 4 tests** $\to$ **Stage 5 transparency pipeline** $\to$ **results JSON** $\to$ **this memo**.

| Piece | Scripts | What they do |
|---|---|---|
| Data layer | `src/guide_and_go.py`, `src/gg_panel_data.py` | WRDS pulls (ExecuComp, CRSP, I/B/E/S, Form 4, Capital IQ), departures and features; Stage 3 data: CEO episodes, matched sales, guidance panel, CEO-quarter panel |
| Text labels | `src/gg_labels.py`, `validation/` | Name parsing, insider-name matching, abruptness classifier; blind annotation tasks, 280 hand-labelled departures, two-pass reads of every flagged event, `candidate_labels.csv` |
| Statistics core | `src/gg_tests.py`, `src/gg_econ.py` | Calendar-time portfolio, factor alpha, placebo and permutation tests, Holm; fixed-effects OLS with two-way clustered errors, cluster logit |
| Stage 1 and 2 | `src/gg_study.py`, `src/gg_variants.py` | The pre-specified primary and secondary tests; wider links and the optimism split |
| Stage 3 and 4 | `src/gg_stage3.py`, `src/gg_stage3_calibrate.py`, `src/gg_stage4.py` | P1 to P3 and robustness (`--dry-run` randomises outcomes), null calibration, explore and confirm |
| Stage 5 filings | `src/gg_filings.py` | Finds each firm's 10-K, 10-Qs and 8-K exhibits (WRDS index, EDGAR), bundles only documents filed on or before the 10-K, caches extracted text |
| Stage 5 phases | `src/gg_phases.py`, `src/gg_phase_run.py` | Reads revenue-table labels, compares consecutive filings (NONE, MINOR, MAJOR), triages MAJOR changes with a model check, builds phases and the filing-level score table; resumable batch runner |
| Stage 5 scorer | `src/gg_transparency.py`, `scripts/score_phases.py`, `scripts/assemble_scores.py` | Rubric, extraction, frozen configuration, the Claude Code scorer, reliability gate; scores each phase's first filing in parallel shards; builds `stage5_scores.pkl` using only reads made on the current text |
| Stage 5 tests | `src/gg_stage5.py` | Attaches $T$ with no look-ahead, Tests 1 to 4, secondary analyses, preconditions, dry run (`--dry-run`) and the registered run |
| Stage 6 | `src/gg_stage6.py`, `scripts/run_stage6.py`, `scripts/run_stage6_moderators.py` | CEO sale-days (value filter, exit flag), guidance flags with no look-ahead, cumulative abnormal returns after each sale with pre-sale betas, the primary regression, power simulation; the run script prints the sample and power before the estimates |
| Stage 7 | `src/gg_stage7.py`, `scripts/run_stage7.py` | Exit announcements with abruptness class, the routine versus opportunistic sale rule with no look-ahead, two-digit industry codes; the run script prints samples and power before the estimates, then runs the two primary tests (Holm) and the secondary checks |
| Hand audit | `src/gg_audit_book.py` | Builds the blind Excel workbook for the 100-filing hand audit |
| Reporting | `scripts/memo_tables.py`, `scripts/build_memo.py`, `notebooks/make_nb.py` | Regression tables from the result files; this memo; the results notebook |
| Tests | `tests/test_gg.py`, `tests/test_gg5.py`, `tests/test_gg6.py`, `tests/test_gg7.py` | Unit tests and simulations (planted-effect recovery, false-positive rate, power) for Stages 1 to 4 and Stages 5 to 7 |

To rebuild: `python src/guide_and_go.py build`, then `src/gg_study.py`, `src/gg_variants.py`, `src/gg_stage3.py`, `src/gg_stage4.py`; for Stage 5 `scripts/score_phases.py`, `scripts/assemble_scores.py` and `src/gg_stage5.py`; finally `python scripts/build_memo.py`. The README lists every file and the commands.

---

# 5. Results and discussion

**How to read the tables.** Each column is one specification; each variable has a coefficient row (stars: * p<0.10, ** p<0.05, *** p<0.01 on the two-sided p-value) and a standard error in parentheses beneath it. Standard errors are clustered by firm and quarter unless a table says otherwise. A blank cell means the variable is not in that specification.

## 5.1 Stage 1: flagged firms do not underperform

|  | Primary | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8a | S8b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Four-factor alpha, % per year | -0.6 | -0.6 | 0.4 | 1.6 | 1.4 | 0.1 | 3.0 | -0.1 | 3.8 | -4.0 |
| Naive t-statistic | -0.11 | -0.11 | 0.06 | 0.38 | 0.31 | 0.02 | 1.51 | -0.01 | 0.38 | -0.61 |
| Placebo p-value (one-sided, lower tail) | 0.39 | 0.39 | 0.49 | 0.53 | 0.52 | 0.45 | 0.63 | 0.47 | 0.54 | 0.31 |
| Holm-adjusted p | n/a | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Flagged events | 42 | 42 | 42 | 59 | 43 | 42 | 267 | 35 | 15 | 27 |
| Mean positions held | 2.3 | 2.3 | 1.6 | 3.0 | 2.7 | 2.3 | 12.2 | 2.3 | 1.7 | 3.0 |
| Factors: Mkt-RF, SMB, HML, UMD | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

*Columns: Primary is the pre-specified test. **S1** CEO only; **S2** 6-month hold; **S3** lenient insider match (adds initial-only matches); **S4** rule-based abrupt instead of annotators; **S5** six-factor model; **S6** guidance + sale, any exit type (no abruptness); **S7** excluding 2008-09 and 2020; **S8a** announcements 2004-2014; **S8b** announcements 2015-2025.*

The pre-specified primary test finds a four-factor alpha of **-0.6% a year** (placebo p 0.39) for the 42 flagged firms, and every secondary test is null (Holm-adjusted p 1.0). With 42 events the test could reliably detect an alpha of about -20% a year, not -5% to -10%; the 95% interval is roughly -16% to +15%. The broader group of any guide-sell-leave firm (S6, 267 events) shows +3.0% a year and excludes underperformance larger than about 4% a year. The two sub-periods go in opposite directions (+3.8% on 15 events, -4.0% on 27), which is what noise looks like.

## 5.2 Stage 2: the wider group is not underperforming

**Panel A. Alpha of the whole group (no optimism split)**

|  | L1 x U1 | L1 x U2 | L2 x U1 | L2 x U2 | L3 x U1 | L3 x U2 | L4 x U1 | L4 x U2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Events | 235 | 157 | 277 | 187 | 670 | 483 | 1114 | 821 |
| Four-factor alpha, % per year | 2.0 | 0.2 | 0.9 | 0.5 | 2.2 | 2.8 | 1.3 | 1.7 |
| Placebo p-value | 0.52 | 0.38 | 0.33 | 0.35 | 0.73 | 0.80 | 0.41 | 0.62 |

**Panel B. Optimistic versus other guidance**

|  | L1 x U1 | L1 x U2 | L2 x U1 | L2 x U2 | L3 x U1 | L3 x U2 |
|---|---:|---:|---:|---:|---:|---:|
| Optimistic-guidance events | 87 | 62 | 99 | 71 | 221 | 161 |
| Other events | 134 | 87 | 160 | 106 | 400 | 289 |
| Alpha, optimistic, % per year | -0.7 | -1.0 | 0.5 | 0.6 | -1.9 | -0.8 |
| Alpha, other, % per year | 4.2 | 2.1 | 1.6 | 1.3 | 4.2 | 4.6 |
| Difference (optimistic minus other) | -4.9 | -3.2 | -1.1 | -0.7 | -6.1 | -5.4 |
| Permutation p (one-sided) | 0.066 | 0.200 | 0.276 | 0.425 | 0.003 | 0.046 |
| Holm-adjusted p (family of 14) | 0.789 | 1.000 | 1.000 | 1.000 | 0.048 | 0.603 |

All eight cells have alpha between +0.2% and +2.8% a year (placebo p 0.33 to 0.80), with intervals of roughly plus or minus 2% (1,114 events) to 8% (157 events). One optimism-split test clears Holm: **L3 x U1, -6.1% a year, permutation p 0.003, Holm-adjusted 0.048.** I do not treat it as a finding: (1) the optimistic arm is only -1.9% and the gap comes mostly from the other arm at +4.2%; (2) it vanishes in the strict links (L1 p 0.07, L2 p 0.28); (3) an alternative intent split (guidance raised against the firm's own previous guidance) goes the opposite way; (4) misses after the exit are not worse; (5) continuous optimism against 12-month returns has a Spearman correlation of -0.03 (p 0.47); (6) Stage 2 followed a Stage 1 null, a forking path.

## 5.3 Stage 3: no effect on exits, misses or relative performance

**Table 5.3a. P1, probability of leaving (outcome: exit indicator times 100, percentage points per quarter; baseline exit rate 2.5%)**

|  | (1) Baseline | (2) O continuous | (3) Sale >= $1M | (4) Excl. retirements | (5) 2004-14 | (6) 2015-25 | (7) Skip a quarter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Optimistic guidance (O) | 0.009 | -0.050 | 0.018 | 0.071 | -0.070 | 0.260 | -0.019 |
|  | (0.129) | (0.087) | (0.124) | (0.128) | (0.173) | (0.172) | (0.114) |
| Sale since guidance (S) | 0.439* | 0.313 | 0.601* | 0.430* | 0.395 | 0.673* | 0.915* |
|  | (0.247) | (0.198) | (0.328) | (0.237) | (0.316) | (0.386) | (0.493) |
| O x S | -0.321 | -0.191 | -0.670* | -0.354 | -0.309 | -0.463 | -0.979 |
|  | (0.303) | (0.221) | (0.379) | (0.286) | (0.409) | (0.457) | (0.679) |
| Sale in prior 12 months | -0.046 | -0.048 | 0.010 | -0.110 | 0.252 | -0.432 | 0.035 |
|  | (0.171) | (0.170) | (0.169) | (0.157) | (0.207) | (0.278) | (0.168) |
| Log market cap | -1.081*** | -1.071*** | -1.087*** | -1.016*** | -1.750*** | -1.264*** | -1.081*** |
|  | (0.170) | (0.170) | (0.171) | (0.167) | (0.269) | (0.328) | (0.171) |
| 12-month return | -1.000*** | -0.981*** | -0.992*** | -0.986*** | -0.267 | -1.349*** | -0.979*** |
|  | (0.206) | (0.207) | (0.203) | (0.203) | (0.251) | (0.388) | (0.206) |
| Log tenure | 2.026*** | 2.027*** | 2.022*** | 2.001*** | 2.834*** | 3.289*** | 2.029*** |
|  | (0.162) | (0.162) | (0.162) | (0.156) | (0.294) | (0.264) | (0.161) |
| Age | 0.245*** | 0.245*** | 0.245*** | 0.181*** | 0.297*** | 0.207*** | 0.245*** |
|  | (0.030) | (0.029) | (0.030) | (0.027) | (0.043) | (0.053) | (0.029) |
| Firm fixed effects | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Year-quarter fixed effects | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Observations (CEO-quarters) | 75,016 | 75,016 | 75,016 | 74,839 | 43,921 | 31,095 | 75,016 |
| Firm clusters | 85 | 85 | 85 | 85 | 44 | 41 | 85 |
| One-sided p, O x S > 0 | 0.85 | 0.80 | 0.96 | 0.89 | 0.77 | 0.84 | 0.92 |

*Column (2) uses standardised continuous optimism, so its O coefficients are per standard deviation. A logit version without firm effects (not shown, different scale) agrees. The minimum detectable O x S in column (1) is 0.75 pp, 30% of the baseline exit rate.*

**Table 5.3b. P2, earnings miss (outcome: (actual $-$ guided midpoint) / price, percent of price)**

|  | (1) Baseline | (2) O continuous | (3) Sale >= $1M | (4) vs consensus | (5) First release | (6) Excl. retired | (7) Pre-guidance sale | (8) 2004-14 | (9) 2015-25 | (10) Sale ends 90d pre-exit | (11) Drop near exit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Optimistic guidance (O) | -0.067*** | -0.039 | -0.055*** | 0.487*** | -0.015 | -0.066*** | -0.049** | -0.070*** | -0.076** | -0.066*** | -0.067*** |
|  | (0.019) | (0.024) | (0.018) | (0.026) | (0.021) | (0.019) | (0.019) | (0.021) | (0.030) | (0.019) | (0.019) |
| Sale (S) | 0.137*** | 0.163*** | 0.135*** | 0.222*** | 0.116*** | 0.138*** | -0.001 | 0.136*** | 0.064** | 0.135*** | 0.138*** |
|  | (0.032) | (0.031) | (0.035) | (0.036) | (0.025) | (0.032) | (0.027) | (0.038) | (0.027) | (0.032) | (0.032) |
| Exit (X) | -0.418*** | -0.430*** | -0.389*** | -0.574*** | -0.524*** | -0.416*** | -0.418*** | -0.383*** | -0.372*** | -0.322*** | -0.562*** |
|  | (0.089) | (0.085) | (0.084) | (0.098) | (0.086) | (0.094) | (0.097) | (0.118) | (0.101) | (0.086) | (0.112) |
| O x S | 0.056** | -0.004 | 0.033 | -0.110*** | 0.037 | 0.055** | 0.013 | 0.091*** | 0.051* | 0.053** | 0.057** |
|  | (0.025) | (0.035) | (0.031) | (0.029) | (0.027) | (0.024) | (0.024) | (0.033) | (0.028) | (0.025) | (0.025) |
| O x X | -0.053 | 0.068 | -0.034 | 0.165 | 0.085 | -0.042 | -0.100 | 0.020 | -0.092 | -0.043 | -0.122 |
|  | (0.095) | (0.064) | (0.088) | (0.101) | (0.139) | (0.103) | (0.110) | (0.120) | (0.127) | (0.090) | (0.119) |
| S x X | 0.475*** | 0.453*** | 0.499*** | 0.533*** | 0.377*** | 0.495*** | 0.349*** | 0.350** | 0.323*** | 0.285** | 0.627*** |
|  | (0.140) | (0.126) | (0.133) | (0.149) | (0.119) | (0.157) | (0.111) | (0.150) | (0.117) | (0.113) | (0.158) |
| O x S x X | -0.032 | -0.108 | -0.105 | -0.062 | 0.032 | -0.153 | 0.174 | 0.072 | -0.030 | 0.049 | 0.037 |
|  | (0.173) | (0.125) | (0.175) | (0.181) | (0.191) | (0.208) | (0.139) | (0.158) | (0.209) | (0.213) | (0.205) |
| Log market cap | -0.303*** | -0.297*** | -0.303*** | -0.210*** | -0.236*** | -0.296*** | -0.297*** | -0.386*** | -0.192*** | -0.302*** | -0.314*** |
|  | (0.073) | (0.073) | (0.073) | (0.078) | (0.047) | (0.072) | (0.073) | (0.114) | (0.066) | (0.073) | (0.072) |
| 12-month return | 0.634*** | 0.641*** | 0.641*** | 0.786*** | 0.416*** | 0.624*** | 0.654*** | 0.693*** | 0.407*** | 0.636*** | 0.631*** |
|  | (0.056) | (0.056) | (0.057) | (0.061) | (0.040) | (0.056) | (0.058) | (0.079) | (0.058) | (0.056) | (0.056) |
| Log days to period end | -0.115*** | -0.116*** | -0.108*** | -0.120*** | -0.150*** | -0.114*** | -0.094*** | -0.142*** | -0.061** | -0.115*** | -0.112*** |
|  | (0.022) | (0.022) | (0.022) | (0.023) | (0.024) | (0.022) | (0.021) | (0.029) | (0.025) | (0.022) | (0.023) |
| Quarterly release | 0.317*** | 0.320*** | 0.317*** | 0.315*** | 0.191*** | 0.316*** | 0.315*** | 0.410*** | 0.167*** | 0.316*** | 0.315*** |
|  | (0.054) | (0.053) | (0.054) | (0.056) | (0.050) | (0.053) | (0.054) | (0.076) | (0.036) | (0.054) | (0.054) |
| Log tenure | -0.084** | -0.083** | -0.081** | -0.086** | -0.055** | -0.083** | -0.069* | -0.154** | -0.022 | -0.081** | -0.085** |
|  | (0.037) | (0.037) | (0.037) | (0.040) | (0.024) | (0.037) | (0.037) | (0.071) | (0.037) | (0.036) | (0.037) |
| Age | 0.002 | 0.002 | 0.002 | 0.002 | 0.003 | 0.002 | 0.002 | 0.008 | 0.004 | 0.002 | 0.002 |
|  | (0.005) | (0.004) | (0.005) | (0.005) | (0.003) | (0.005) | (0.005) | (0.008) | (0.006) | (0.004) | (0.004) |
| Firm fixed effects | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Year-quarter fixed effects | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Observations (releases) | 94,742 | 94,742 | 94,742 | 94,736 | 43,673 | 94,328 | 94,742 | 58,853 | 35,889 | 94,742 | 93,219 |
| Firm clusters | 85 | 85 | 85 | 85 | 85 | 85 | 85 | 44 | 41 | 85 | 85 |
| One-sided p, O x S x X < 0 | 0.43 | 0.19 | 0.27 | 0.37 | 0.57 | 0.23 | 0.89 | 0.67 | 0.44 | 0.59 | 0.57 |

*The minimum detectable O x S x X in column (1) is 0.43 pp of price, about the size of the whole exit main effect.*

**Table 5.3c. P3, leavers versus stayers (calendar-time four-factor alpha)**

|  | Events | Four-factor alpha, % per year |
|---|---:|---:|
| Leavers (guided, sold, left) | 157 | 0.2 |
| Stayers (guided, sold, stayed) | 5,388 events from 1,981 executives | 2.3 |
| Difference, leavers minus stayers |  | -2.1 |
| Permutation p (one-sided, lower tail) |  | 0.26 |

None of the three is significant (Holm-adjusted 0.77 to 0.85) and all seventeen robustness checks agree, including a falsification test with sales *before* the guidance. The control group is now in, and the pattern does not predict leaving, misses, or underperformance relative to executives who stayed. Descriptively (not tested hypotheses, not corrected for multiplicity): exits are followed by worse misses ($X$ about $-0.4$), but CEOs who sold before leaving show **no worse** misses than those who left without selling ($S \times X$ is positive), the opposite of the intent story; a sale since the latest guidance goes with a slightly higher exit probability ($S$ about +0.4 pp, two-sided p about 0.08) but not more so when guidance was optimistic.

## 5.4 Stage 4: exploration with a hold-out

- **Sign.** No sign convention rescues the pattern: the O x S x X term is +0.07 (2004 to 2014) and -0.03 (full sample), positive or negative depending on the half and the exit type.
- **Controls.** Adding age bands, last realised miss, guidance style, volatility and CEO ownership changes nothing (P1 -0.31 to -0.31; P2 +0.07 to +0.08).
- **A data problem found.** ExecuComp's exit-reason field is blank for 87% of exits and for nearly all exits after 2009. The one exploratory "hit" (a positive sale-optimism-retirement effect, t 3.5, 56 releases from 25 firms) came from that field and cannot be tested out of sample; exit type now comes from the Capital IQ text classifier. No candidate passed the exploration rule.
- **What replicates in both halves.** CEO exits not preceded by a sale follow quarters of earnings misses (about -0.2 to -0.6 pp of price in the four quarters before and after); exits preceded by a sale in the prior year do not. A sale since the latest guidance raises the quarterly exit probability a little (about +0.4 pp on a 2.5% base, two-sided p about 0.08).
- **Reading.** CEO sales before an exit look like markers of orderly, planned departures, not exits ahead of bad news. Exits that follow deterioration are not preceded by sales (plausibly because executives with adverse information are restricted from selling; not tested). The planned-versus-forced split and every sale-by-exit-type term flip sign between the two halves.

## 5.5 Stage 5: is the pattern stronger where reporting is opaque?

| Test | H5 predicts | Estimate | One-sided p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Test 1: returns, excess alpha low minus high T | < 0 | +0.283 | 0.961 | 1.00 |
| Test 2: exits, Z x O x S | < 0 | -0.040 | 0.449 | 1.00 |
| Test 3: misses, Z x O x S x X | > 0 | -0.452 | 0.962 | 1.00 |
| Test 4: leavers vs stayers, d | > 0 | +0.004 | 0.462 | 1.00 |

**Table 5.5a. Test 1, returns of flagged firms by transparency group**

|  | (1) Low T (T <= 2) | (2) High T (T >= 3) | (3) Low minus high |
|---|---:|---:|---:|
| Flagged events | 24 | 18 |  |
| Four-factor alpha of flagged firms, % per year | 10.6 | -15.6 | 26.2 |
| Mean alpha of same-size placebo groups, % per year | 1.4 | 3.5 |  |
| **Excess alpha (flagged minus placebo), % per year** | 9.2 | -19.1 | **28.3** |
| Placebo pool (non-flagged events, announcement identified) | 1,649 | 1,531 |  |
| 95% bootstrap interval for the difference (calendar-quarter blocks) |  |  | 4.8 to 52.0 |
| One-sided placebo p, H5: difference < 0 |  |  | 0.961 |
| Calendar-time spread, low-T minus high-T portfolio alpha (t) |  |  | 33.6 (t = 2.25) |
| Minimum detectable difference, % per year |  |  | 42 |

**Table 5.5b. Test 2, exits (outcome: exit indicator times 100; $Z$ is standardised $T$)**

|  | (1) Stage 3 baseline (no T) | (2) Test 2: with T |
|---|---:|---:|
| Optimistic guidance (O) | 0.009 | 0.006 |
|  | (0.129) | (0.134) |
| Sale since guidance (S) | 0.439* | 0.415* |
|  | (0.247) | (0.244) |
| O x S | -0.321 | -0.304 |
|  | (0.303) | (0.293) |
| T (standardised, Z) |  | -0.119 |
|  |  | (0.280) |
| Z x O |  | 0.070 |
|  |  | (0.146) |
| Z x S |  | 0.044 |
|  |  | (0.253) |
| **Z x O x S** |  | -0.040 |
|  |  | (0.309) |
| Sale in prior 12 months | -0.046 | -0.047 |
|  | (0.171) | (0.170) |
| Log market cap | -1.081*** | -1.054*** |
|  | (0.170) | (0.174) |
| 12-month return | -1.000*** | -0.985*** |
|  | (0.206) | (0.206) |
| Log tenure | 2.026*** | 2.030*** |
|  | (0.162) | (0.165) |
| Age | 0.245*** | 0.243*** |
|  | (0.030) | (0.030) |
| Firm fixed effects | Yes | Yes |
| Year-quarter fixed effects | Yes | Yes |
| Observations (CEO-quarters) | 75,016 | 73,623 |
| Firm clusters | 85 | 85 |
| Exits | 1,943 | 1,943 |
| H5 test, one-sided p |  | 0.45 (H5: Z x O x S < 0) |
| Minimum detectable Z x O x S |  | 0.77 |

*Column (2) uses CEO-quarters with a transparency score; the sample therefore differs from column (1). The H5 coefficient is Z x O x S (bold).*

**Table 5.5c. Test 3, earnings misses (outcome: percent of price)**

|  | (1) Stage 3 baseline (no T) | (2) Test 3: with T |
|---|---:|---:|
| Optimistic guidance (O) | -0.067*** | -0.076*** |
|  | (0.019) | (0.019) |
| Sale (S) | 0.137*** | 0.125*** |
|  | (0.032) | (0.032) |
| Exit (X) | -0.418*** | -0.378*** |
|  | (0.089) | (0.088) |
| O x S | 0.056** | 0.072*** |
|  | (0.025) | (0.025) |
| O x X | -0.053 | -0.071 |
|  | (0.095) | (0.092) |
| S x X | 0.475*** | 0.451*** |
|  | (0.140) | (0.146) |
| O x S x X | -0.032 | -0.033 |
|  | (0.173) | (0.181) |
| T (standardised, Z) |  | 0.140 |
|  |  | (0.098) |
| Z x O |  | -0.033* |
|  |  | (0.017) |
| Z x S |  | -0.077** |
|  |  | (0.038) |
| Z x X |  | 0.055 |
|  |  | (0.068) |
| Z x O x S |  | 0.090*** |
|  |  | (0.027) |
| Z x O x X |  | 0.108 |
|  |  | (0.103) |
| Z x S x X |  | 0.136 |
|  |  | (0.191) |
| **Z x O x S x X** |  | -0.452* |
|  |  | (0.251) |
| Log market cap | -0.303*** | -0.314*** |
|  | (0.073) | (0.072) |
| 12-month return | 0.634*** | 0.624*** |
|  | (0.056) | (0.056) |
| Log days to period end | -0.115*** | -0.114*** |
|  | (0.022) | (0.023) |
| Quarterly release | 0.317*** | 0.305*** |
|  | (0.054) | (0.052) |
| Log tenure | -0.084** | -0.083** |
|  | (0.037) | (0.037) |
| Age | 0.002 | 0.003 |
|  | (0.005) | (0.005) |
| Firm fixed effects | Yes | Yes |
| Year-quarter fixed effects | Yes | Yes |
| Observations (releases) | 94,742 | 93,013 |
| Firm clusters | 85 | 85 |
| Releases with sale and exit | 1,006 | 1,006 |
| H5 test, one-sided p |  | 0.96 (H5: Z x O x S x X > 0) |
| Minimum detectable Z x O x S x X |  | 0.62 |

*Column (2) uses releases with a transparency score. The H5 coefficient is Z x O x S x X (bold).*

**Table 5.5d. Test 4, leavers versus stayers by transparency group**

|  | (1) Low T | (2) High T | (3) Difference |
|---|---:|---:|---:|
| Leavers (guide, sell, leave) | 81 | 65 |  |
| Stayers (guide, sell, stay) | 3,162 | 2,066 |  |
| Leaver minus stayer alpha, % per year | -1.0 | -0.7 |  |
| **Interaction d = (high-T effect) - (low-T effect), % per year** |  |  | **0.4** |
| Standard error (cluster by announcement quarter) |  |  | (4.4) |
| One-sided p, wild cluster bootstrap, H5: d > 0 |  |  | 0.462 |
| Continuous version: Leaver x Z coefficient |  |  | 0.019 (0.028), one-sided p 0.25 |
| Observations; clusters |  |  | 5,374; 83 |

None of the four tests supports H5, and after Holm correction every adjusted p-value is 1.0. Tests 2 and 4 are essentially zero ($-0.040$ with a standard error of 0.31; $+0.004$ with a standard error of 0.044). Tests 1 and 3 lean the **opposite** way from H5: flagged firms did worse where reporting is transparent (excess alpha $-19.1\%$ a year against $+9.2\%$ at low $T$, with a bootstrap interval for the difference of $+4.8$ to $+52.0$ points), and the Test 3 interaction is $-0.45$ against a predicted positive sign (two-sided p 0.075). That is not a finding: it rests on 24 and 18 flagged events, was not a registered direction, and is one of 32 project-wide tests. A first run before the code review gave the same verdict (family p-values 0.965, 0.448, 0.963, 0.462).

## 5.6 Stage 6: H5 at the level of the insider sale

**Table 5.6a. Sample and power**

|  | Count or value |
|---|---:|
| CEO sale-days, any value | 77,396 |
| Sale-days of at least $100,000 | 68,247 |
| ... with a transparency score T (strictly prior 10-K, not stale) | 50,545 |
| Primary sample (T and the outcome available) | 50,123 sale-days from 1,725 firms |
| Sale-days after guidance in the prior 180 days (G) | 44,586 |
| Sale-days followed by an exit within 365 days (X) | 4,928 |
| **Guide-sell-leave sale-days (G and X)** | **4,176** |
| ... of which guidance was optimistic (O and X) | 1,804 |
| Standard error of the H5 coefficient (pp per sd of T) | 1.69 |
| Minimum detectable effect (one-sided 5%, 80% power), pp per sd of T | 4.21 |
| Simulated rejection rate at planted effects (pp per sd): 0.0, 2.1, 4.2, 8.4 | 0%, 20%, 82%, 100% |
| Sale-days of at least $100,000 with no T, by reason | not_scored: 16,435, no_filing: 796, extraction_failed: 315, stale_filing: 156 |

**Table 5.6b. Avoided loss after CEO sales (outcome: $-100 \times$ four-factor abnormal return over the next $h$ trading days, percentage points; positive means the stock did worse after the sale)**

|  | (1) Primary, h=63 | (2) O in place of G | (3) High-T dummy for Z | (4) h=21 | (5) h=126 | (6) h=252 | (7) Sales >= $1M | (8) All sale values | (9) First sale-day per quarter | (10) 2004-14 | (11) 2015-24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Guided in prior 180 days (G) | -1.03 | -0.50 | -1.16 | -0.53 | -1.68 | -1.18 | -2.12** | -0.75 | -1.32** | -0.34 | -3.11*** |
|  | (1.00) | (0.42) | (1.43) | (0.43) | (1.33) | (2.16) | (0.91) | (1.01) | (0.57) | (1.48) | (0.96) |
| Leaves within 365 days (X) | 1.46 | 1.70** | 1.86 | 0.72 | 1.46 | 3.65 | -0.63 | 2.11 | 1.89 | -0.51 | 1.61 |
|  | (1.42) | (0.76) | (2.05) | (0.67) | (2.16) | (3.12) | (1.25) | (1.48) | (1.15) | (2.20) | (2.18) |
| G x X | 0.44 | 0.23 | 0.25 | -0.07 | 1.63 | -1.85 | 2.69** | -0.31 | -0.45 | 1.10 | 1.57 |
|  | (1.50) | (1.09) | (2.15) | (0.71) | (2.39) | (3.44) | (1.34) | (1.55) | (1.25) | (2.25) | (2.33) |
| Transparency (Z) | -0.55 | -0.63 | -0.81 | -0.06 | -1.86 | 0.38 | -1.63 | -0.33 | -0.95 | 0.87 | -1.84 |
|  | (1.39) | (0.97) | (2.41) | (0.53) | (2.21) | (3.42) | (1.39) | (1.32) | (0.75) | (2.32) | (1.18) |
| Z x G | -0.13 | -0.16 | 0.23 | -0.14 | -0.81 | -3.88 | 0.14 | -0.19 | 0.91 | -1.05 | 0.99 |
|  | (1.15) | (0.48) | (1.99) | (0.42) | (1.79) | (2.99) | (1.12) | (1.07) | (0.59) | (1.95) | (0.76) |
| Z x X | -0.61 | 0.10 | -1.08 | 0.16 | -2.36 | -3.98 | -0.75 | -0.81 | -0.43 | -0.25 | 0.62 |
|  | (1.37) | (0.87) | (2.85) | (0.72) | (2.17) | (2.77) | (1.52) | (1.35) | (1.04) | (1.99) | (1.97) |
| **Z x G x X** | 0.27 | -1.38 | 0.58 | -0.33 | 2.91 | 4.01 | 0.96 | 0.50 | -0.29 | -0.83 | 0.49 |
|  | (1.69) | (1.16) | (3.32) | (0.79) | (2.98) | (3.53) | (1.71) | (1.62) | (1.13) | (2.37) | (2.37) |
| Log market cap | 6.83*** | 6.79*** | 6.85*** | 1.91*** | 12.97*** | 23.94*** | 6.15*** | 6.96*** | 5.69*** | 11.53*** | 9.08*** |
|  | (0.64) | (0.63) | (0.64) | (0.24) | (1.05) | (1.66) | (0.65) | (0.62) | (0.42) | (0.98) | (1.36) |
| 12-month return | -0.01 | 0.05 | -0.02 | 0.18 | 0.03 | 0.27 | -0.32 | -0.02 | -0.28 | -0.16 | -1.07 |
|  | (0.60) | (0.60) | (0.60) | (0.25) | (1.07) | (1.69) | (0.77) | (0.58) | (0.51) | (0.79) | (0.98) |
| Log sale value | 0.08 | 0.09 | 0.08 | 0.05 | 0.27 | 0.70** | -0.06 | 0.15* | 0.02 | 0.10 | 0.12 |
|  | (0.13) | (0.13) | (0.14) | (0.07) | (0.21) | (0.35) | (0.22) | (0.09) | (0.12) | (0.16) | (0.19) |
| Log tenure | -0.07 | -0.07 | -0.06 | -0.05 | -0.53 | -0.48 | 0.10 | -0.32 | 0.14 | 0.13 | 0.13 |
|  | (0.29) | (0.30) | (0.29) | (0.14) | (0.52) | (0.81) | (0.36) | (0.36) | (0.24) | (0.38) | (0.39) |
| Age | -0.04 | -0.04 | -0.04 | -0.02 | -0.08 | -0.12 | -0.05 | -0.01 | -0.04 | -0.04 | -0.08 |
|  | (0.04) | (0.04) | (0.04) | (0.02) | (0.06) | (0.11) | (0.05) | (0.05) | (0.03) | (0.05) | (0.05) |
| Firm fixed effects | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Year-month fixed effects | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Observations (sale-days) | 49,767 | 49,767 | 49,767 | 49,947 | 49,524 | 48,993 | 23,740 | 55,594 | 21,045 | 29,862 | 19,905 |
| Guide-sell-leave sale-days (G x X) | 4,176 | 1,804 | 4,176 | 4,207 | 4,130 | 4,043 | 2,119 | 4,485 | 1,470 | 2,296 | 1,880 |
| One-sided p, Z x G x X < 0 | 0.56 | 0.12 | 0.57 | 0.34 | 0.84 | 0.87 | 0.71 | 0.62 | 0.40 | 0.36 | 0.58 |
| Minimum detectable Z x G x X | 4.2 | 2.9 | 8.3 | 2.0 | 7.4 | 8.8 | 4.2 | 4.0 | 2.8 | 5.9 | 5.9 |

*Column (2) uses optimistic guidance $O$ in place of $G$, so the rows labelled G, G x X, Z x G and Z x G x X are $O$-based there. Column (3) replaces $Z$ by a high-transparency dummy ($T \ge 3$). Standard errors are clustered by firm and by month; the minimum detectable effect is per standard deviation of $T$ on the abnormal return. The H5 coefficient is Z x G x X (bold): a negative value supports H5.*

**Table 5.6c. Secondary test S1: does the avoided loss depend on transparency for all sales?**

|  | Avoided loss on transparency, all sales (S1) |
|---|---:|
| Transparency (Z) | -0.72 |
|  | (0.95) |
| Log market cap | 6.82*** |
|  | (0.63) |
| 12-month return | -0.02 |
|  | (0.60) |
| Log sale value | 0.08 |
|  | (0.13) |
| Log tenure | 0.05 |
|  | (0.29) |
| Age | -0.03 |
|  | (0.04) |
| Firm and year-month fixed effects | Yes |
| Observations (sale-days) | 49,767 |
| One-sided p, Z < 0 | 0.23 |

**Table 5.6d. Moderator follow-up (S8): litigation-prone industry and volatility**

|  | (1) Primary, same sample | (2) + litigation-prone industry (L) | (3) + volatility (V) | (4) + both |
|---|---:|---:|---:|---:|
| Guided (G) | -1.03 | -2.04** | -1.20 | -1.95** |
|  | (1.00) | (0.94) | (0.90) | (0.91) |
| Leaves within 365 days (X) | 1.46 | 1.78 | 1.07 | 1.66 |
|  | (1.42) | (1.39) | (1.32) | (1.38) |
| G x X | 0.44 | -0.57 | 0.74 | -0.37 |
|  | (1.50) | (1.48) | (1.41) | (1.47) |
| Transparency (Z) | -0.55 | -0.76 | -0.66 | -0.82 |
|  | (1.39) | (1.33) | (1.39) | (1.33) |
| Z x G | -0.13 | 0.09 | 0.00 | 0.17 |
|  | (1.15) | (1.07) | (1.14) | (1.06) |
| Z x X | -0.61 | -0.89 | -0.53 | -0.87 |
|  | (1.37) | (1.35) | (1.35) | (1.34) |
| **Z x G x X** | 0.27 | 0.65 | 0.13 | 0.56 |
|  | (1.69) | (1.66) | (1.65) | (1.64) |
| L x G |  | 2.57 |  | 2.07 |
|  |  | (1.99) |  | (1.98) |
| L x X |  | -1.87 |  | -2.66 |
|  |  | (3.54) |  | (3.56) |
| **L x G x X** |  | 3.35 |  | 3.75 |
|  |  | (3.84) |  | (3.82) |
| V |  |  | 0.93 | 1.05 |
|  |  |  | (0.73) | (0.73) |
| V x G |  |  | 0.85 | 0.70 |
|  |  |  | (0.65) | (0.64) |
| V x X |  |  | 1.05 | 1.23 |
|  |  |  | (1.15) | (1.10) |
| **V x G x X** |  |  | -0.35 | -0.71 |
|  |  |  | (1.28) | (1.21) |
| Controls (market cap, 12-month return, sale value, tenure, age) | Yes | Yes | Yes | Yes |
| Firm and year-month fixed effects | Yes | Yes | Yes | Yes |
| Observations (sale-days) | 49,767 | 49,767 | 49,767 | 49,767 |
| One-sided p, Z x G x X < 0 | 0.56 | 0.65 | 0.53 | 0.63 |

*L = 1 for the litigation-prone industries of Francis, Philbrick and Schipper; V = standard deviation of daily returns over the 252 days before the sale, standardised. Each moderator enters as M, M x G, M x X and M x G x X. The direction for L (negative) was fixed before it was computed; V has no directional prior.*

**Result.** The primary coefficient on Z x G x X is **+0.27** percentage points per standard deviation of $T$ (standard error 1.69, one-sided p 0.56). It has the sign opposite to H5 and is nowhere near significant; the 95% interval is about $-3.0$ to $+3.6$, so avoided-loss effects stronger than about 3 points per standard deviation of $T$ over a quarter are excluded. The test was reasonably powered for its question: the minimum detectable effect is 4.2 points per standard deviation and the simulation rejected 82% of the time when exactly that effect was planted, with 4,176 guide-sell-leave sale-days. The main effect of a guide-sell-leave sale (G x X) is +0.44 (standard error 1.50): such sales do not avoid losses on average, which is the Stage 3 conclusion at sale level.

**Secondary specifications.** The optimistic-guidance version has the H5 sign but is not significant ($-1.38$, standard error 1.16, one-sided p 0.12); the avoided loss on transparency alone for all sales also has the H5 sign ($-0.72$, standard error 0.95, p 0.23). At longer horizons the sign reverses (h = 126: +2.91; h = 252: +4.01), the two sub-periods disagree ($-0.83$ and $+0.49$), and sales of at least $1 million and the first sale per quarter change nothing. No specification supports H5, and none is outside the range expected from noise at these standard errors.

**Moderators (S8).** Neither moderator changes the primary result: the Z x G x X coefficient is +0.65 with litigation-prone industry in the regression, +0.13 with volatility and +0.56 with both (standard errors about 1.65; the primary alone is +0.27). The moderators' own triple interactions are also null. Litigation-prone industry has the **opposite** sign from the one fixed in advance (L x G x X = +3.35, standard error 3.84, one-sided p 0.81): guide-sell-leave sales in litigation-prone industries avoided slightly *more* loss, not less, though the interval is wide (42% of sale-days are in these industries). Volatility is also null ($-0.35$, standard error 1.29, two-sided p 0.79). Analyst coverage, trading volume, accruals and institutional ownership are not in the data cache and were not run, and the Kim and Skinner predicted litigation probability cannot be estimated without litigation outcomes.

**A descriptive finding, not a hypothesis.** The coefficient on G is negative in most columns (significantly so for sales of at least $1 million, the first sale per quarter and 2015 to 2024): sales after guidance are followed by *better* returns than other sales, not worse. That is consistent with the upward drift after optimistic guidance reported by Das et al. (Section 2.1) and says that guidance, by itself, is not a bad-news marker for the seller.

**Caveats specific to this stage.** About a quarter of sale-days of at least $100,000 (16,435 of 68,247) have no transparency score because the last 10-K before the sale was not among the 24,154 filings scored in Stage 5; those are mostly firm-quarters outside the guidance panel, so the sample leans toward guiding firms (89% of the primary sample follow guidance). The exit flag is ex-post, 10b5-1 plan sales cannot be separated before 2023, repeated sales have overlapping windows (handled by firm clustering and by the first-sale-per-quarter column), and $T$ carries the Stage 5 measurement limits. One secondary test (S1) was first run with the wrong one-sided tail and re-run on the saved panel (p 0.775 became 0.225); the primary regression was not affected.

## 5.7 Stage 7: between-firm variation, abrupt exits and routine sales

**Table 5.7a. Samples and power (power was printed before any estimate)**

|  | P7a: Stage 6 sample | P7b: abrupt exits, non-routine | Placebo: planned exits | P7b on all sales | P7b, opportunistic only |
|---|---:|---:|---:|---:|---:|
| Sale-days | 50,123 | 45,546 | 44,835 | 49,157 | 17,743 |
| Firms | 1,725 | 1,713 | 1,702 | 1,718 | 890 |
| Sale-days after guidance (G) | 44,586 | 40,619 | 40,036 | 43,879 | 16,250 |
| Sale-days followed by the exit studied (X) | 4,928 | 711 | 2,311 | 751 | 328 |
| **Guide-sell-leave sale-days (G and X)** | **4,176** | **583** | **2,043** | **623** | **279** |
| Firms with such a sale-day | 660 | 142 | 379 | 146 | 56 |

*Exits announced within 365 days after a sale: 4,356 sale-days, of which 751 abrupt, 2,639 planned or other and 966 unclassified. Sale classes (routine rule): 3,657 routine, 18,139 opportunistic, 28,327 unclassified.*

| Power, printed before the estimates | SE of Z x G x X | Minimum detectable effect (pp per sd) | Planted effects (pp per sd) | Simulated rejection rate |
|---|---:|---:|---:|---:|
| P7a | 1.78 | 4.43 | 0.0, 2.2, 4.4, 8.9 | 0%, 17%, 67%, 100% |
| P7b | 3.62 | 9.02 | 0.0, 4.5, 9.0, 18.0 | 3%, 3%, 80%, 100% |
| Stage 6 primary (for comparison) | 1.69 | 4.21 | 0.0, 2.1, 4.2, 8.4 | 0%, 20%, 82%, 100% |

**Table 5.7b. Avoided loss after CEO sales with industry fixed effects (outcome: $-100 \times$ four-factor abnormal return over the next 63 trading days unless stated, percentage points)**

|  | (1) P7a | (2) P7b | (3) P7a, firm FE | (4) P7b, firm FE | (5) Placebo: planned exit | (6) P7b, all sales | (7) P7b, opportunistic only | (8) P7b with O | (9) P7b, h=21 | (10) P7b, h=126 | (11) P7b, first sale per quarter | (12) P7b, SIC division FE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Guided in prior 180 days (G) | -0.22 | -0.43 | -0.84 | -0.49 | -0.58 | -0.33 | -1.46 | -0.16 | -0.17 | -1.20 | -0.79 | -0.43 |
|  | (1.07) | (0.99) | (1.00) | (0.95) | (1.05) | (0.96) | (1.31) | (0.41) | (0.40) | (1.15) | (0.57) | (0.53) |
| Exit studied (X) | 1.92 | 7.73*** | 1.41 | 3.96 | -1.92 | 8.00*** | 6.65 | 4.20* | 2.01* | 6.33* | 5.28** | 7.78** |
|  | (1.60) | (2.54) | (1.43) | (3.26) | (2.70) | (2.57) | (4.08) | (2.20) | (1.19) | (3.34) | (2.10) | (3.09) |
| G x X | -0.56 | -3.33 | 0.37 | 2.62 | 2.79 | -3.35 | -2.22 | 2.20 | -1.75 | 2.44 | -0.01 | -3.24 |
|  | (2.02) | (2.82) | (1.50) | (3.55) | (2.88) | (2.84) | (3.54) | (2.03) | (1.43) | (3.51) | (2.34) | (2.52) |
| Transparency (Z) | 0.42 | -0.19 | -0.65 | -1.63 | -0.16 | 0.32 | 0.80 | -0.18 | -0.13 | -0.63 | -0.45 | -0.10 |
|  | (0.89) | (0.71) | (1.39) | (1.20) | (0.65) | (0.86) | (1.10) | (0.26) | (0.35) | (1.12) | (0.66) | (0.42) |
| Z x G | -0.71 | 0.00 | -0.02 | 0.91 | 0.00 | -0.66 | -0.36 | -0.00 | -0.08 | 0.57 | 0.34 | 0.02 |
|  | (0.95) | (0.71) | (1.13) | (0.83) | (0.68) | (0.89) | (1.12) | (0.28) | (0.35) | (1.13) | (0.64) | (0.49) |
| Z x X | -0.41 | -1.58 | -0.53 | -0.31 | -0.15 | -2.16 | 6.76 | 2.80 | -1.69 | -4.18 | -1.24 | -1.42 |
|  | (1.47) | (3.62) | (1.37) | (2.79) | (2.75) | (3.59) | (5.53) | (2.26) | (1.11) | (3.90) | (1.97) | (2.55) |
| **Z x G x X** | 0.07 | 2.53 | 0.17 | 1.76 | 0.04 | 3.27 | -6.13 | -5.33** | 1.81 | 4.94 | 0.85 | 2.43 |
|  | (1.78) | (3.62) | (1.68) | (2.75) | (2.92) | (3.57) | (5.43) | (2.57) | (1.32) | (4.34) | (2.12) | (2.76) |
| Log market cap | 0.56*** | 0.45** | 7.22*** | 7.26*** | 0.44** | 0.56*** | 0.48*** | 0.45** | 0.06 | 0.97*** | 0.32** | 0.39** |
|  | (0.17) | (0.18) | (0.64) | (0.69) | (0.17) | (0.17) | (0.18) | (0.19) | (0.08) | (0.31) | (0.13) | (0.12) |
| 12-month return | -0.98 | -1.10 | -0.32 | -0.16 | -0.89 | -1.02 | -0.58 | -1.05 | -0.26 | -1.17 | -0.66 | -1.13** |
|  | (0.70) | (0.77) | (0.61) | (0.61) | (0.78) | (0.71) | (0.85) | (0.79) | (0.38) | (1.44) | (0.40) | (0.47) |
| Log sale value | 0.11 | 0.10 | 0.07 | 0.05 | 0.10 | 0.11 | 0.29 | 0.10 | 0.06 | 0.15 | 0.11 | 0.08 |
|  | (0.13) | (0.11) | (0.13) | (0.14) | (0.12) | (0.13) | (0.18) | (0.11) | (0.06) | (0.25) | (0.10) | (0.06) |
| Log tenure | 0.21 | 0.25 | -0.03 | 0.11 | 0.23 | 0.23 | 0.48 | 0.25 | 0.10 | 0.28 | -0.04 | 0.25 |
|  | (0.30) | (0.30) | (0.29) | (0.30) | (0.28) | (0.31) | (0.31) | (0.30) | (0.12) | (0.62) | (0.21) | (0.21) |
| Age | -0.03 | -0.03 | -0.04 | -0.04 | -0.03 | -0.03 | -0.04 | -0.03 | -0.02 | -0.07 | 0.01 | -0.03 |
|  | (0.04) | (0.04) | (0.04) | (0.04) | (0.04) | (0.05) | (0.03) | (0.04) | (0.01) | (0.09) | (0.02) | (0.04) |
| Pre-sale volatility (V) | 1.12*** | 1.12*** | 1.69*** | 1.75*** | 1.02*** | 1.11*** | 1.09** | 1.12*** | 0.48*** | 2.04*** | 0.83*** | 1.03*** |
|  | (0.32) | (0.33) | (0.45) | (0.46) | (0.34) | (0.32) | (0.47) | (0.32) | (0.18) | (0.65) | (0.25) | (0.22) |
| Fixed effects | SIC2 + month | SIC2 + month | Firm + month | Firm + month | SIC2 + month | SIC2 + month | SIC2 + month | SIC2 + month | SIC2 + month | SIC2 + month | SIC2 + month | SIC division + month |
| Observations (sale-days) | 49,767 | 45,202 | 49,767 | 45,202 | 44,492 | 48,811 | 17,715 | 45,202 | 45,202 | 44,999 | 19,817 | 45,202 |
| Guide-sell-leave sale-days (G x X) | 4,176 | 583 | 4,176 | 583 | 2,043 | 623 | 279 | 243 | 583 | 577 | 240 | 583 |
| One-sided p, Z x G x X < 0 | 0.52 | 0.76 | 0.54 | 0.74 | 0.51 | 0.82 | 0.13 | 0.02 | 0.91 | 0.87 | 0.65 | 0.80 |
| Minimum detectable Z x G x X | 4.4 | 9.0 | 4.2 | 6.8 | 7.3 | 8.9 | 13.5 | 6.4 | 3.3 | 10.8 | 5.3 | 6.9 |
| Holm-adjusted p (primary tests only) | 1.00 | 1.00 |  |  |  |  |  |  |  |  |  |  |

*Columns (1) and (2) are the two primary tests; the rest are secondary. In (8) the rows labelled G, G x X, Z x G and Z x G x X are based on optimistic guidance $O$. The H5 coefficient is Z x G x X (bold): a negative value supports H5.*

**Neither lever raised power.** Lever 1 alone (P7a) gives a coefficient of +0.07 with a standard error of **1.78**, against Stage 6's 1.69, and the firm-effects version on the same sample (column 3) gives +0.17 (1.68). So industry-by-month effects bought nothing for the H5 term, as the spec had warned they might not. Adding the abrupt-exit and non-routine restrictions (P7b) **lowered** power: only 583 guide-sell-leave sale-days from 142 firms remain, the standard error is 3.62 and the smallest detectable effect is **9.0** points per standard deviation of $T$, more than twice Stage 6's 4.2. P7b's estimate is +2.53, with a one-sided p of 0.76.

**Result.** Both primary tests are null: one-sided p-values of 0.52 and 0.76, Holm-adjusted 1.0 for both. The placebo with planned exits is also null (+0.04, standard error 2.93). Dropping routine sales, using only opportunistic ones, changes the sign to the H5 direction but not the conclusion ($-6.1$, standard error 5.4, one-sided p 0.13, from 279 sale-days at 56 firms). The 21-day, 126-day, first-sale-per-quarter and SIC-division versions all have the opposite sign.

**One nominally significant secondary result, which is not a finding.** With optimistic guidance in place of any guidance, the abrupt-exit test gives $-5.33$ (standard error 2.57, one-sided p 0.021, two-sided 0.042) from 243 guide-sell-leave sale-days. I report it because it points in the H5 direction, but: (1) it is one of ten secondary tests in a re-analysis of data already used, so the chance that at least one reaches 5% by luck alone is about 40%; (2) it rests on 243 sale-days, which the power table shows is too few to detect effects under 9 points per standard deviation reliably, so a significant estimate at that size is as likely to be an overestimate as a discovery; (3) the optimism split is exactly where Stage 2's one significant result (L3 x U1) appeared and then fell apart (Section 5.2). It is a lead for a pre-registered test on data not yet used, and nothing more.

## 5.8 Discussion

**What do the results answer?** Stages 1 to 3 answer: *is there a large, average underperformance, extra exit risk or extra earnings miss attached to guide-sell-leave?* The answer is no. Stage 5 answers a narrower question: *is the difference between low-transparency and high-transparency firms large enough to be seen with 24 and 18 flagged events and these noisy interaction tests?* The answer is again no. Stage 6 asks it on 50,123 CEO sale-days (4,176 of them guide-sell-leave sales) and finds a coefficient of +0.27 (standard error 1.69) against a minimum detectable effect of 4.2 points per standard deviation of $T$. It does not answer whether transparency matters, and my earlier wording that the null "does not hide a pattern" was too strong.

**Power.** The null rules out only large effects:

| Test | Smallest effect the design could reliably detect | Sample behind it |
|---|---|---|
| Stage 1 primary | an alpha of about -20% a year | 42 events |
| Stage 3 P1, O x S | 0.75 pp per quarter (30% of the baseline exit rate) | 1,943 exits |
| Stage 3 P2, O x S x X | 0.43 pp of price | 1,006 sale-and-exit releases |
| Test 1 | a difference of about 42 points of annual alpha | 24 and 18 flagged events |
| Test 2, Z x O x S | 0.77 | 1,943 exits |
| Test 3, Z x O x S x X | 0.62 | 1,006 cells |
| Test 4 | standard error 0.044 | 81 and 65 leavers |
| Stage 6, Z x G x X | 4.2 pp of the 63-day abnormal return per standard deviation of $T$ | 4,176 guide-sell-leave sale-days |
| Stage 7 P7a (industry effects) | 4.4 pp per standard deviation | 4,176 guide-sell-leave sale-days |
| Stage 7 P7b (abrupt exits, non-routine) | 9.0 pp per standard deviation | 583 sale-days from 142 firms |

A real opacity effect of 5 to 10 points a year would not have been visible. The bottleneck is the number of flagged events; the interaction tests (a difference of a difference of a difference) are the weakest of all.

**Why power is low.** Only 42 events meet the strict flag. The outcome (a 12-month alpha) is noisy for a handful of firms, and any signal may arrive on a few dates. The interaction tests ask for a difference of differences. $T$ is split into two coarse groups and measured with about half a point of noise, which pulls every estimate toward zero. Four tests are corrected together.

**Alternative explanations of the null.** (1) There is no effect. (2) The effect exists but is diluted: "optimism" is a proxy for intent, the sale condition includes routine and planned sales, and the flag mixes planned and forced exits; Stage 4 suggests sales before an exit mark orderly departures, which would make the flag select the wrong executives. (3) The effect is concentrated in a small subset (for example technology firms, half the strict sample). (4) The measurement of $T$ is too coarse or too inflated to separate firms.

**Limitations.** Only the S&P 1500 and CEOs in the panels (CFO exits are not recorded in ExecuComp); the strict flag has 42 events, half in technology; annotators are language models, not people, and insider name matching is tested but not hand-audited; the model scores $T$ about half a point above my own scores, and recognises the firm 55% of the time even with names stripped (recognised firms scored 0.36 points higher, not significantly); phases carry one score across filings (first and last filing differ by 0.5 points); materials are limited to EDGAR filings; point-in-time index membership could not be verified; the complexity controls for Tests 2 and 3 (log assets, segment count) are not in the panels, so those sensitivities were skipped.

**What Stages 6 and 7 changed, and what could still raise power.** Moving from 42 events to individual insider sales (Stage 6) cut the minimum detectable effect, but the result is still null. Stage 7 then tested the two changes I had thought would add power: industry-by-month effects did **not** (standard error 1.78 against 1.69; the H5 term is identified from differences across firms in the effect, which firm effects do not remove, so the firm-effects design was not the bottleneck I first thought), and restricting to abrupt exits and non-routine sales **lowered** power, because only 583 guide-sell-leave sale-days from 142 firms remain (minimum detectable effect 9.0 against 4.2). The binding constraint is the number of guide-sell-leave sales and the number of firms behind them. The levers that act on that are a larger population (other officers, directors, 2025 to 2026), outcomes measured where the news arrives (abnormal returns in the three days around each of the next earnings releases, plus the earnings surprises), two filings per phase averaged to cut the noise in $T$ (reliability from about 0.68 to about 0.81), scoring the roughly 24% of sale-days that have no $T$, and the detectability moderators that need data not in the cache (analyst coverage, trading volume, accruals, institutional ownership; Section 4.7). Any further redesign must be written down before it touches data and run on events not yet used, because these results have been seen.

**What to take away.**
1. The "guide, sell, leave" story is **not supported** here; the early positive-looking numbers came from data and labelling errors.
2. Opacity does not rescue it. Stage 5 could only exclude very large effects, and Stage 6, with 4,176 guide-sell-leave sales, excludes avoided-loss effects stronger than about 3 points per standard deviation of $T$ over a quarter. The Stage 7 re-analysis did not add power. None of them can say whether a small effect exists, and one nominally significant secondary Stage 7 result (abrupt exits after optimistic guidance, p 0.02) is a lead to test on new data, not a finding.
3. The project's value is the pipeline and the discipline: it found and fixed its own data problems, validated its measurement, pre-specified its tests, and reported nulls with their power.

**Honest disclosures.** The Stage 1 test plan was written after I had seen earlier versions of the project, and a dry-run mistake exposed two secondary results (both null) beforehand (`docs/PREREGISTRATION.md`). Stage 2 followed a Stage 1 null. Stage 5 was designed after the Stage 1 to 4 nulls and its first run was repeated after a code review. The tests are not independent of one another (same firms and sales data). Stage 6 was proposed after the Stage 5 results were seen, though its specification (`docs/STAGE6_SPEC.md`) was fixed before any sale-level outcome was computed; one secondary test of it (S1) was first run with the wrong one-sided tail and re-run. Stage 7 is an exploratory re-analysis of the Stage 6 sales, specified before it was run (it looked only at counts and variance facts beforehand), and one of its design arguments (that firm effects were costing power for the H5 term) was overstated by me and corrected in its spec before any regression. Open items and what was fixed are in `docs/limitations.md`, `docs/STAGE5_SPEC.md`, `docs/STAGE6_SPEC.md` and `docs/STAGE7_SPEC.md`.

## 5.9 Next steps

1. **A confirmatory test on data not yet used**, pre-registered before it is run: the sale-level H5 test on all officers (CFOs and other executives, which needs their departures from the Capital IQ text, since the event file holds almost none) and on 2025 to 2026 sales, with the optimistic-guidance and abrupt-exit variant that Stage 7 left as a lead written in as one named test.
2. Add the detectability moderators the cache lacks (analyst coverage and dispersion, trading volume, accruals, institutional ownership).
3. Score the filings that the 16,435 unscored sale-days point to, so Stage 6 and 7 cover firms outside the guidance panel.
4. A human audit of 50 to 100 labels and 50 insider matches, and a check of the abruptness classifier's precision, which Stage 7 relies on.
5. A direct test of the mechanism: guidance misses that are matched on accounting basis, after the executive leaves.
