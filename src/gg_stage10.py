"""Stage 10 interim-run machinery (docs/STAGE10_SPEC.md): the validation gate, the universe restriction, the per-test samples, Holm, the filing chain to label, and the
refusal checks for extending the phase and score tables. Pure functions, so the logic the unattended scripts rely on is unit-tested (tests/test_gg10.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def gate_passed(rep: dict, min_corr: float = 0.99, min_share: float = 0.95) -> bool:
    """The spec's bar (Amendment 1, unchanged by Amendment 2): correlation at least min_corr AND share of returns within 0.001 at least min_share.
    A missing or NaN statistic fails."""
    c, s = rep.get("corr"), rep.get("share_within")
    if c is None or s is None or not np.isfinite(c) or not np.isfinite(s):
        return False
    return bool(c >= min_corr and s >= min_share)


def restrict_universe(new: pd.DataFrame, universe_cusip6: set, from_date: str = "2026-01-01", cleanse=("R", "H")) -> pd.DataFrame:
    """The streamed insider rows added to the cached file: dated from `from_date`, cleansed records only, and only firms whose CUSIP is in the original universe."""
    t = pd.to_datetime(new["trandate"], errors="coerce")
    return new[(t >= pd.Timestamp(from_date)) & new["cleanse"].isin(list(cleanse)) & new["cusip6"].isin(universe_cusip6)]


def holm_adjust(p: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p[k]))
        out[k] = run
    return out


def winsor(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def make_sample(x: pd.DataFrame) -> pd.DataFrame:
    """A test sample: complete cases on the raw outcome (Yraw), Z, pre-sale volatility and industry; the outcome winsorised at 1% and 99% and V standardised
    ON THE SAMPLE ITSELF (as the spec fixes), so each test has its own scaling."""
    b = x[x["Yraw"].notna() & x["Z"].notna() & x["vol"].notna() & x["sic2"].notna()].copy().reset_index(drop=True)
    b["Y63"] = winsor(b["Yraw"])
    b["V"] = (b["vol"] - b["vol"].mean()) / b["vol"].std()
    return b


def stage_samples(d: pd.DataFrame, new_from: str = "2025-10-01", large_cut: float = 22.298):
    """(pooled, new months all firms, new months large firms): large means log market cap at least `large_cut` (inclusive); new months are on or after `new_from`."""
    after = d["trandate"] >= pd.Timestamp(new_from)
    return make_sample(d), make_sample(d[after]), make_sample(d[after & (d["lmcap"] >= large_cut)])


def filings_to_label(not_scored: pd.DataFrame, tk: pd.DataFrame, last_have, have_accessions: set, end: str) -> pd.DataFrame:
    """The 10-K filings to label for insider-days that have no score: for every firm whose last 10-K before an unscored sale exists, ALL of its 10-Ks filed after its
    last already-labelled filing (all of them when it has none) and on or before `end`, minus accessions already labelled. `not_scored` has gvkey and trandate;
    `tk` has gvkey, accession, filing_date, cik, url (10-K forms)."""
    t = tk.copy()
    t["filing_date"] = pd.to_datetime(t["filing_date"]).astype("datetime64[ns]")
    t["gvkey"] = t["gvkey"].astype(str).str.zfill(6)
    n = not_scored[["gvkey", "trandate"]].copy()
    n["gvkey"] = n["gvkey"].astype(str).str.zfill(6)
    n["trandate"] = pd.to_datetime(n["trandate"]).astype("datetime64[ns]")
    m = pd.merge_asof(n.sort_values("trandate"), t.sort_values("filing_date")[["gvkey", "filing_date", "accession"]], left_on="trandate", right_on="filing_date",
                      by="gvkey", direction="backward", allow_exact_matches=False)
    firms = sorted(set(m.dropna(subset=["accession"])["gvkey"]))
    lh = {str(k).zfill(6): pd.Timestamp(v) for k, v in dict(last_have).items()}
    rows = []
    for g in firms:
        f = t[(t["gvkey"] == g) & (t["filing_date"] <= pd.Timestamp(end))]
        if g in lh:
            f = f[f["filing_date"] > lh[g]]
        rows.append(f)
    out = pd.concat(rows) if rows else t.iloc[:0]
    out = out[~out["accession"].isin(have_accessions)].drop_duplicates("accession")
    return out[["gvkey", "accession", "filing_date", "cik", "url"]].reset_index(drop=True)


def changed_scores(new: pd.DataFrame, old: pd.DataFrame, col: str = "t_final") -> list:
    """Accessions that exist in `old` whose score in `new` differs (NaN equals NaN) or that are missing from `new`: an extension must change none of them."""
    a = old.set_index("accession")[col]
    b = new.drop_duplicates("accession").set_index("accession")[col].reindex(a.index)
    gone = b.index[~a.index.isin(new["accession"])]
    diff = a.index[(a.fillna(-1e18) != b.fillna(-1e18)) | a.index.isin(gone)]
    return sorted(diff.tolist())


def changed_phases(new: pd.DataFrame, old: pd.DataFrame, cols=("level", "phase", "phase_first")) -> list:
    """Accessions in `old` whose level, phase or phase_first differ in `new` (or that are missing from it)."""
    a = old.set_index("accession")[list(cols)]
    b = new.drop_duplicates("accession").set_index("accession")[list(cols)].reindex(a.index)
    bad = ~((a == b) | (a.isna() & b.isna())).all(axis=1)
    return sorted(a.index[bad].tolist())
