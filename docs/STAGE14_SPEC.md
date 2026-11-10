# Stage 14: do valuation and turnaround explain the flagged firms' returns? Controls, matching and the announcement reaction

Status: **registered 2026-11-09, before any Stage 14 return, regression or p-value was computed.** Author: Nayrb (requested by the author on 2026-11-09). **Exploratory:** it reuses the departure events and returns of Stages 1 to 5 and 13; nothing here is confirmatory, and it adds no formal test to the project count.

## 1. Why

The Stage 13 flag (sale from nine months before the latest guidance to the announcement) gave an alpha of +3.4% a year (t 0.79, n 145); the original flag gave -0.6% (n 42). Both are statistically zero. The author's hypothesis for the positive sign: markets reward turnaround stories (a better CEO replaces the departing one) and flagged firms already trade at low multiples, so a value effect is in play. The calendar-time four-factor model already includes the value factor (HML); the six-factor secondary (adds profitability and investment) barely moved the estimate (+3.6%). This stage asks the question at the event level, with firm-specific valuation and turnaround proxies, which a portfolio loading on HML cannot capture.

**What I looked at before registering (no outcomes):** the columns of the events file (it has size, prior-12-month return, announcement-window return and 12-month buy-and-hold return, but no book-to-market, earnings or profitability variable), the number of events with each (pre-event return 4,633; announcement-window return 4,633; 12-month buy-and-hold return 4,223), and the columns of the cached ExecuComp file (no join date, so successor type can only be proxied).

## 2. Samples and flags

All departure events announced 2004 to 2024 with a 12-month buy-and-hold return (BHAR, from day +2 to +253, as in Stage 1; windows end 2024 because of the earlier price files). Two flags, both with abruptness from the annotator label where one exists and the rule classifier otherwise (as in Stage 13): **old** (guidance within 180 days, a sale after the latest guidance, abrupt; 42 events with a window) and **new** (the Stage 13 wider sale condition; 145). Each is analysed on its own.

## 3. Variables (all dated strictly before the announcement)

- **Valuation:** book-to-market $BM=$ book equity (ceq) over market capitalisation at the announcement; earnings yield $EP=$ income before extraordinary items over market capitalisation. Book equity and earnings come from the last Compustat annual report whose fiscal year end is at least **90 days before** the announcement (the publication lag; no look-ahead). A firm with non-positive book equity gets $\log BM$ missing and an indicator; earnings are winsorised at 1% and 99%.
- **Turnaround and distress proxies (pre-event only):** prior-12-month return (in the events file), return on assets $ROA=$ income over assets and its change from the previous fiscal year, a negative-earnings indicator.
- **Size:** log market capitalisation at the announcement (in the events file).
- **Outcome:** BHAR winsorised at 1% and 99%.

## 4. Analyses

**A. Announcement-window reaction.** The cumulative abnormal return from the day before to the day after the announcement (in the events file), flagged against all other departures: difference in means with errors clustered by announcement quarter. This tests directly whether the market rewards these exits, on the one window the post-announcement alphas exclude.

**B. Valuation- and characteristic-matched comparison.** Each flagged event is matched to its $k=5$ nearest other departures **announced in the same year with the same role**, on standardised log market capitalisation, prior-12-month return, $\log BM$ and $EP$ (nearest in Euclidean distance; events missing a matching variable are dropped). Statistic: the mean BHAR of flagged events minus the mean BHAR of their matches, with a 95% interval from a bootstrap over flagged events (2,000 draws, seed 0). Reported against the unmatched (size and prior-return only) version for comparison.

**C. Event-level regression.** BHAR on the flag with year fixed effects, controls (log market capitalisation, prior-12-month return, $\log BM$, its missing indicator, $EP$, $ROA$, change in $ROA$, negative-earnings indicator, role), errors clustered by announcement quarter, over all events with a BHAR. Reported: the flag coefficient with (i) no controls, (ii) size and prior return only, (iii) the full set. **The difference between (i) and (iii) is what the controls change.**

**D. Successor type (descriptive, not a control).** For CEO departures, the successor is the other executive at the firm listed as CEO in the departure year or the next. Proxy for an **outside hire**: the successor has no earlier ExecuComp record at the firm (ExecuComp covers only the top five executives, so an insider who was never in the top five is misclassified as outside; this is noisy and labelled as such). Report the share of outside successors among flagged and other departures, and the mean BHAR of flagged events by successor type. **Not used as a control**: successor quality is partly a consequence of leaving, so conditioning on it changes the question being tested.

## 5. Interpretation (fixed now)

- A result is **supported** as an explanation only if the flag coefficient in C(iii) differs from C(i) by more than its own standard error **and** B shows the matched difference closer to the direction the author's story predicts (flagged firms' positive return explained by valuation). Otherwise valuation and turnaround do not account for the sign.
- If every flag coefficient stays statistically indistinguishable from zero with or without controls, the answer is "nothing to explain": the estimate was noise, not a turnaround or value effect.
- A significant announcement-window reaction (A) with no later drift is the standard picture of a market that prices the news at once; it does not imply a post-announcement alpha and is not treated as one.
- All results are exploratory and reported in full; no number from this stage enters the project-wide test count.

## 6. Known limits fixed now

Book-to-market and earnings from annual statements are stale by up to a year; the successor proxy is noisy; 12-month BHAR is skewed and its clustering by quarter is a convenience (events cluster by announcement date); the 42 old-flag events give little power for the matched comparison; the data end 2024 for BHAR. Compustat fundamentals are pulled through the WRDS REST API, which ignores firm filters, so the table is streamed and filtered locally.

## 7. Results (run once on 2026-11-09 after the spec above was fixed; `scripts/run_stage14.py`, output `outputs/results/stage14_results.json`)

One code error stopped the first attempt before anything was computed (a string executive ID stored in a numeric column); it was fixed test-first and the run repeated. Nothing was changed because of a result.

**Samples.** 5,187 departure events, 4,223 with a 12-month BHAR; the **old flag** has 42 flagged events with a BHAR and the **new flag** 141 (the calendar-time alpha of Stage 13 uses 145 because window availability differs slightly); valuation variables exist for 4,215 of the 4,223. 82,865 Compustat firm-years for 2,409 firms. Units: BHAR and CAR are fractions (0.01 = 1 percentage point).

**A. Announcement-window reaction (CAR from day -1 to +1, flagged against all other departures, 4,633 events).** Old flag: flagged mean -1.53%, others -0.69%, difference -0.84 points (se 0.98, p 0.39). New flag: flagged -1.78%, others -0.67%, difference **-1.11 points (se 0.64, two-sided p 0.084)**. The market does **not** reward these exits on announcement; the point estimate is slightly negative and not significant at 5%.

**B. Matched comparison (mean flagged BHAR minus the mean of five same-year, same-role matches; 95% bootstrap interval).** Old flag: size and prior return only +0.9 points [-10.1, +11.5]; adding book-to-market and earnings yield **+5.2 points [-5.2, +15.8]**, n 42. New flag: +1.3 points [-5.7, +9.1] and **+1.5 points [-5.8, +9.2]**, n 139 (2 dropped for missing variables). All intervals contain zero and are wide.

**C. Event-level regression (BHAR on the flag, year effects, errors clustered by announcement quarter).** Old flag: no controls +0.0002 (se 0.0678); size and prior return +0.0119 (0.0670); full controls **+0.0107 (0.0673)**. New flag: no controls +0.0036 (0.0416); size and prior return +0.0080 (0.0414); full controls **+0.0079 (0.0414)**. **The controls move the flag coefficient by about one percentage point, a fifth or less of its standard error (4 to 7 points).**

**D. Successor type (descriptive; outside-hire proxy is noisy).** Share of CEO departures with an outside successor: old flag 26% (n 42) against 17% for others; new flag 17% (n 150) against 17%. Mean BHAR of flagged events: old flag outside successor -13.4% (n 11) against inside +7.8% (n 30); new flag outside -4.3% (n 23) against inside +1.2% (n 115). Small samples; if anything outside hires do worse, the opposite of a turnaround story.

**Descriptives, flagged against other departures (old flag / new flag).** Larger (log market cap 22.3 / 21.8 against 21.5), **less distressed** (negative-earnings share 19% / 14% against 24%), **higher** profitability (ROA 4.6% / 5.4% against 2.0%) and **earnings yield** (+4.1% / +5.2% against -2.8% / -3.0%), **lower** log book-to-market (-1.11 / -0.87 against -0.78), prior-year return -3.7% (old) and -12.3% (new) against -8.1% / -8.0%. The flagged firms are not value stocks on book-to-market and are not distressed; they look more profitable than other departing-executive firms.

**Reading under Section 5.** Valuation and turnaround proxies do **not** account for the sign of the result: the flag coefficient in C(iii) differs from C(i) by far less than its own standard error for both flags, and B's matched differences keep zero inside their intervals. Every flag coefficient is statistically indistinguishable from zero with or without controls, so under the "nothing to explain" clause the earlier small positive Stage 13 alpha (+3.4%, t 0.79) was noise and not a turnaround or value effect. The announcement reaction (A) is not positive. The premise that flagged firms trade at low multiples is not supported either: they are higher on book multiples and cheaper on earnings. Exploratory; no number from this stage enters the project-wide test count.
