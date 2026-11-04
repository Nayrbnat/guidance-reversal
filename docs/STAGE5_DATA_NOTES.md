# Stage 5 data notes

Written 2026-11-01 by the Task 2 live run (`build_filing_index`). Numbers are from the event and panel firms of Stages 1-4
(2,927 firms: `events.csv`, `panel_A_quarters`, `panel_B_sample`).

## Filing source and index

- **Source:** `wrdssec.wrds_forms` (cik, form, fdate, accession, fname), 10-K, 10-K405 and their amendments only. Postgres was
  unreachable from this machine, so the pull ran through the WRDS REST API, which ignores filters on `form`; every form is
  downloaded and reduced to 10-K types chunk by chunk (small cache `gg_cache/stage5_forms10k/`).
- **Link:** `wrdssec.wciklink_gvkey` (dated), hulled per gvkey-CIK pair; undated duplicate rows dropped when a dated row exists.
  SIC from `comp.company`.
- **Result:** 70,180 filings for 2,921 of the 2,927 firms (6 firms have no CIK link or no 10-K in the table).
  Local copy: `gg_cache/filing_index_stage5.pkl`. Raw report: `gg_cache/stage5_live_index_report.json`.

## Coverage of the timing rule (10-K filed strictly before `dep_ann`)

98.4% of the 5,187 departure events have a 10-K before the announcement; the 85 that do not get `T = NaN` with reason `no_filing`
and are counted, never zero-filled.

| Year | Events | Share with a prior 10-K | Year | Events | Share with a prior 10-K |
|---|---|---|---|---|---|
| 2003 | 23 | 95.7% | 2015 | 297 | 99.0% |
| 2004 | 200 | 94.0% | 2016 | 258 | 98.8% |
| 2005 | 214 | 96.7% | 2017 | 244 | 100.0% |
| 2006 | 233 | 93.6% | 2018 | 257 | 98.4% |
| 2007 | 247 | 94.3% | 2019 | 262 | 99.6% |
| 2008 | 250 | 96.8% | 2020 | 223 | 99.1% |
| 2009 | 218 | 99.1% | 2021 | 193 | 99.0% |
| 2010 | 202 | 99.5% | 2022 | 240 | 99.6% |
| 2011 | 216 | 99.5% | 2023 | 219 | 99.1% |
| 2012 | 243 | 99.6% | 2024 | 259 | 100.0% |
| 2013 | 232 | 99.1% | 2025 | 236 | 99.2% |
| 2014 | 221 | 99.5% | | | |

Coverage is lowest in 2004-2008 (93.6% to 96.8%), where filings are older or links are thinner; those events will have `T` missing at
a somewhat higher rate. Whether missingness relates to the flag is checked when `T` is attached (Task 9, step 4).

## Link validation

- 313 gvkeys map to more than one CIK over time and 41 CIKs map to more than one gvkey. Both are expected over 20 years of
  mergers and re-listings; the dated link windows decide which gvkey a filing counts for. These lists are in the raw report.
- A random 20-pair spot check (Compustat name beside each CIK) is in the raw report. **The comparison against the EDGAR
  conformed name is not done yet:** the filing table pull did not keep `coname`. To do before the full scoring run.

## S&P 1500 membership (point in time)

- The Stage 1-4 event universe uses Execucomp coverage as its "S&P 1500" proxy; it is not point-in-time index membership.
- A point-in-time check was attempted with `comp.idxcst_his` (S&P 500 `000003`, MidCap 400 `024248`, SmallCap 600 `030824`;
  the S&P 1500 composite is `031855`). **The result cannot be trusted:** through the REST API the table returns about 1,155 S&P 1500
  rows, `thru` is empty on every row, and 28% of events fall on a constituent date. That is the signature of a current-constituents
  extract, not the full history. Re-run on Postgres or from a WRDS web query before relying on any membership figure.
- **Limitation recorded:** the universe is Execucomp-covered firms, as in Stages 1-4; the spec's point-in-time check is still open.
  Stages 1-4 are not reopened (spec 5).
