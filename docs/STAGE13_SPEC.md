# Stage 13: a wider sale condition for the Guidance Reversal flag, and the event-level tests rerun

Status: **registered 2026-11-09, before any Stage 13 return, alpha or p-value was computed.** Author: Nayrb (requested by the author on 2026-11-09, after the Fiserv case was checked against the flag). Numbering: Stages 11, 12 and 12b were registered in parallel sessions; Stage 10 (large-firm lead) is this project's earlier pre-registration. This is Stage 13.

## 1. The change, and why it is post hoc

The original flag needs the executive's sale to fall **after the latest guidance**. Fiserv, the motivating case, fails that: Frank Bisignano sold $25.7M in September 2024, seven months before the guidance of April 2025 and the abrupt exit of May 2025. **New sale condition:** a sale from **9 months before the latest guidance up to the announcement date** (both ends inclusive). Guidance within 180 days before the announcement and an abrupt exit are unchanged. I read "9 months before to after" this way, and not as a window around the announcement, because the flag is built only from information known on the announcement date and a sale after it is not.

**This is a forking path and I am recording it as one.** The change was prompted by one case I already knew about, so it is fitted to that case. Events and returns are the same ones Stages 1 to 5 used, so nothing here is confirmatory. The definition does not replace the original: the original columns are never overwritten, and the original results stay in the memo as the pre-registered ones.

## 2. What I looked at before registering (counts and booleans only, no returns)

`scripts/count_stage13.py` (results in `outputs/results/stage13_counts.json`): of 5,187 departure events, 289 were candidates under the original sale rule and **832** under the wider rule (543 added); 161 of the 832 have an abrupt exit by the rule classifier, of which 114 are added. Of the added abrupt candidates, 99 have a 12-month return window (the last event with a window is 2024-03-14, since the earlier stages' CRSP prices end 2024-12-31). The original flag had 45 annotated events, 42 with a window. Fiserv is a new candidate and a new rule flag (sale value in the window $25.7M) but has no return window. Added events are 112 CEOs and 2 CFOs.

## 3. Abruptness for the added events

The original flag used the annotators' two-pass abrupt label where it existed (it overrode the rule classifier). **The 114 added abrupt candidates have no annotator label.** The primary flag therefore uses, for each event, the annotator label where one exists and the rule classifier otherwise. This mixes two sources, so the sensitivity S4 below uses the rule classifier for every event.

## 4. The test (one formal test)

**P13.** Flagged events (wider sale condition, guided within 180 days, abrupt exit as in Section 3) with a usable 12-month window; calendar-time four-factor alpha of the equal-weighted portfolio, **placebo-portfolio p-value, one-sided, $p<0.05$**, exactly the machinery and 2004 to 2025 announcement years of the Stage 1 primary test (`src/gg_study.py`, 2,000 placebo draws, Newey-West 10 lags, weights = open positions). The Stage 1 primary needed at least 40 flagged events with windows; this one will be reported with its number whatever it is. This is one new formal test.

## 5. Secondary (outside the count; Holm among themselves)

The Stage 1 secondaries on the new flag: S1 CEO only; S2 six-month hold; **S3 (lenient insider match) is dropped**, because the lenient variant is defined only for the original sale rule; S4 rule-based abruptness for all events; S5 six-factor model; S6 guidance and sale with any exit type; S7 excluding 2008 to 2009 and 2020; S8a and S8b announcements 2004 to 2014 and 2015 to 2025. Also reported: the original flag's estimate beside the new one, and the new-only events (added events) on their own.

## 6. What is not rerun, and why

- **Stage 2** (14 link and universe variants) already contains sale conditions that do not require the sale to follow the latest guidance (L3 and L4); a rerun under the new flag would duplicate them.
- **Stage 3 panels and Stages 6 to 10** define guidance and sales per release or per sale, not per flagged event, so the sale-after-guidance condition does not appear in them in the same way; changing them is a different design question and is not made here.
- **Stage 5 tests** split the flagged events by transparency $T$; Test 1 can be rerun on the new flag with the same code if P13 has enough events, and will be reported as exploratory if so.

## 7. Interpretation (fixed now)

Supported: P13 placebo p below 0.05 with a negative alpha. Not supported: anything else, reported with the number of events and the minimum detectable alpha implied by the placebo standard deviation. A significant result would still be a post hoc redefinition on seen data: it would be a lead for new data, not a finding. A null changes nothing already reported. The 99 added events are mostly events where the sale came more than a few weeks before the guidance, so any effect that appears may reflect sales unrelated to the guidance; I will state the median gap between sale and guidance for the added events when reporting.
