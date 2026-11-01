# Guidance Reversal: Limitations and Future Work

Status: rewritten 2026-10-24 after the measurement rebuild and the pre-specified test (`PREREGISTRATION.md`, `MEMO.md`).
Tags: **[run]** observed in a run, **[code]** verified in code, **[open]** not yet checked or fixed.
Severity: **High** = could change the conclusion; **Med** = biases or weakens it; **Low** = polish.

## 0. What can and cannot be claimed

**Can claim:** in S&P 1500 CEO/CFO departures 2004-2025, firms whose departing CEO issued guidance, sold stock and left abruptly
(42 usable events) show no detectable negative risk-adjusted return (alpha -0.6% a year, placebo p = 0.39), and the broader group that
issued guidance, sold and left in any manner (267 events) shows no underperformance beyond about 4% a year.

**Cannot claim:** that a *modest* effect for the strict flag does not exist (the primary test could not reliably detect less than about -20% a year);
that executives acted with intent; anything about executives who guided and sold but **stayed**; anything about firms outside the S&P 1500.

**Stage 2 addendum (2026-10-24).** Wider samples (157-1,114 events) are null for the group as a whole. One optimism-split test (L3 x U1, -6.1%/yr, Holm 0.048) is borderline and fragile (see `MEMO.md`).
Added limitations: (a) Stage 2 was designed after the Stage 1 null (forking path; ~28 formal tests project-wide, Holm covers only Stage 2); (b) "optimistic" = guidance midpoint above consensus is a proxy for intent, not intent, and I/B/E/S guidance
is often non-GAAP; (c) the U2 universe is almost entirely CEOs (ExecuComp `leftofc` rarely populated for CFOs); (d) L3/L4 links ("any sale in the prior 365 days") are loose and can include routine or planned sales; (e) the Stage 2 sample overlaps Stage 1's S6 heavily, so results are not independent.

**Stage 3 addendum (2026-10-24).** Stayers are now in the design (CEO-quarter panel, 5,388 stayer events), so limitation 2.2 below is **partly resolved for CEOs**; CFOs remain out (ExecuComp records an exit date for only 6% of CFO episodes). Result: null on hazard (P1), earnings misses (P2) and leaver-vs-stayer alpha (P3).
Open items added: (a) power for interaction terms is modest (detectable P2 effect 0.43 pp of price, about the size of the exit main effect; P1 0.75 pp per quarter, 30% of baseline); (b) the optimism proxy and the "any sale" definition remain loose (10b5-1 status unobserved); (c) calibration showed the standard errors may be 5-10% too small; (d) 31 formal tests project-wide; (e) a new CEO's write-offs cannot be separated from intent except through the triple interaction; (f) exits are announced before `leftofc`, which the skip-quarter and 90-day checks only partly address; (g) stationarity is assumed rather than tested, with subperiod splits as the check.

**Stage 4 addendum (2026-10-24).** Exploratory iteration with a 2004-2014 / 2015-2025 hold-out found no robust pattern. Added limitations: (a) **ExecuComp exit reason is missing for 87% of exits and for nearly all exits after 2009**, so it cannot define exit type; the text classifier (Capital IQ) is used instead and covers 87-90% of exits; (b) deaths after 2008 are not excluded in the Stage 3 panels (about 1% of exits); (c) the routine-sale rule (same month in each of the three prior years) classifies only 4.8% of CEO sales as routine, so the routine/non-routine contrast is not informative; (d) the two halves have only 44 and 41 quarterly time clusters, so two-way clustered inference is imprecise; (e) the event-time profile and the sale-timing reading are descriptive hypotheses, not tested effects; (f) iteration after a null is a forking path; only the hold-out protocol limits it.

## 1. Problems found and fixed (so older numbers are not reused)

| Problem | Effect | Status |
|---|---|---|
| WRDS REST API silently truncated large pulls (270,000 rows / 56 firms returned for a query with 485,235 rows / 100 firms) | About half of all firms had no price history in every earlier run; all earlier return results invalid | Fixed: Postgres preferred, API fallback repairs truncation, caches versioned `_v2` |
| Departure date filter ignored by the API | 46% of events were pre-2010 with no data | Fixed: filtered locally |
| Executive names mis-parsed (12% of names, e.g. "J.D." read as last name "D") | Broke text matching and insider matching | Fixed (`gg_labels.parse_name`, unit-tested) |
| Sentence splitter cut "William H. Osborne" at the initial | Departures undetected | Fixed |
| Missing announcement text counted as "abrupt" | Old abrupt rule: 23% precision, 57% flagged vs 21% true rate | Fixed: validated classifier + two independent reads |
| Old "government destination" label (keyword match) | 0 of 40 sampled cases genuine | Removed; only an explicit-phrase rule remains |
| Guidance stored on a split-adjusted basis, actuals and prices as reported | Guidance "misses" of up to +940% of price | Fixed with CRSP split factors |
| "Sale after guidance" measured from the first guidance in a year, not the latest | Mis-timed sales | Fixed: sale between latest guidance and announcement |
| Stale caches keyed by firm count / not by firm set | Subset runs reused wrong files | Fixed: keyed by hash of the id set, superset reuse |
| Flag required a bad outcome (circular comparison) | Headline -21% vs -4% gap was partly built in | Fixed: flag uses only announcement-date information |
| Return windows ignored delisting returns | Failures biased upward | Fixed (`apply_delisting`) |

## 2. Open limitations, most important first

| # | Limitation | Sev |
|---|---|---|
| 2.1 | **Power.** Only 42 usable flagged events over 19 years (average 2.3 open positions in the calendar-time portfolio). Placebo sd is 8%/yr, so the minimum detectable alpha at 80% power is about -20%/yr. A null result excludes large effects only. All flagged firms are CEOs (no CFOs) and about half are in technology. **[run]** | High |
| 2.2 | **No control group of stayers.** The comparison is among *departures*. Executives who guided and sold but did not leave are not observed, so we cannot say whether leaving adds anything. A firm-quarter panel from ExecuComp executives and Form 4 would fix this. **[open]** | High |
| 2.3 | **Universe and selection.** S&P 1500 only (ExecuComp); only firms that give EPS guidance (44% of announcements fall within 180 days of guidance); events need an identified announcement and CRSP data. Micro-caps, where abuse is more plausible, are absent. **[code]** | Med |
| 2.4 | **Measurement error that remains.** On the held-out half of the gold set the new rule classifier has 59% precision and 53% recall for "abrupt"; that is why every candidate was also read by annotators. The annotators are language models, not people (94% agreement between two reads; 90% departure-type agreement, kappa 0.86). "Unclear" counts as not abrupt, so recall of the final flag is unknown. A human spot-check of `validation/gold_labels.csv` and of the 45 flagged events is recommended. **[run]** | Med |
| 2.5 | **Insider name matching is unit-tested but not hand-audited.** The match requires last name plus a full first name or nickname (score 3). Only 5.6% of events show a sale between the latest guidance and the announcement; some true sales may be missed (nicknames, name changes) or family members matched. A 50-pair audit is still to do. **[open]** | Med |
| 2.6 | **Insider data limits.** Form 4 open-market sales only: no pledges, hedges, gifts or option-related trades; 10b5-1 plan status not observed before 2023, so planned sales cannot be separated from discretionary ones. **[code]** | Med |
| 2.7 | **Announcement date.** The announcement is the earliest Capital IQ item where the executive is the one leaving (rule: 70% found; annotators for flagged events, 98% agreement on the item). Earlier leaks, 8-K timing and after-hours releases are not observed. **[run]** | Low-Med |
| 2.8 | **Guidance quality.** EPS guidance only; I/B/E/S guidance is often on an adjusted (non-GAAP) basis while the actual is GAAP, so even after split rebasing about 28% of guidance rows differ from the actual by over 30%. The guidance-miss mechanism is therefore not tested directly. **[run]** | Med |
| 2.9 | **Statistics.** Equal-weighted calendar-time portfolios with few open positions are noisy; factor models (Carhart) may omit relevant risks; Newey-West with 10 lags; the placebo draws from other departures matched on announcement-year counts but not on size or sector. 2025 events are truncated (under 12 months of data; calendar ends 2024-12-31). **[code]** | Med |
| 2.10 | **Pre-registration was not blind.** I had seen earlier versions of the project (day-0 flag, no gap) and, through a bug in my dry run, two secondary results (S4, S6: both null) before the primary test. Disclosed in `PREREGISTRATION.md`. **[run]** | Low |
| 2.11 | **Exploratory items are not evidence.** Matched-control differences, event-time BHARs and return paths are labelled exploratory and none is significant. **[run]** | Low |
| 2.12 | **Interpretation.** A flag shows a pattern, not intent. Departure is endogenous to performance; no counterfactual for what would have happened without the sale or the exit. **[code]** | Med |
| 2.13 | **Engineering.** No continuous-integration; `test_gg.py` covers the new modules only. The old `gg_cache/` files (truncated pulls) are obsolete and can be deleted; `guide_and_go.py` still contains legacy analysis code (`analyze`, RAG index) that predates the rebuild and is not used for inference. Secrets are in `.env` inside a OneDrive-synced folder. **[code]** | Low |

## 3. Future work, ranked

1. **Control group of stayers** (2.2): firm-quarter panel of CEOs/CFOs who guided and sold, with and without departure.
2. **Widen the universe** (2.1, 2.3): add micro-caps and CFO/other officers, or non-U.S. issuers, to raise the number of flagged events.
3. **Human audit** (2.4, 2.5): 50-100 flagged and unflagged cases and 50 insider matches, then re-estimate classifier precision.
4. **Direct mechanism test** (2.8): GAAP-matched guidance misses after the executive leaves, measured against firms whose executive stayed.
5. **Earnings-call language** (Larcker and Zakolyukina) between guidance and exit as an additional signal.
6. **Retire legacy code** in `guide_and_go.py` and add tests for the data layer.
