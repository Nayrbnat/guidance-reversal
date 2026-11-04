"""
gg_filings: the 10-K filing index and the no-look-ahead timing rule for Stage 5 (transparency score T).

A departure event uses the last 10-K filed strictly before the announcement date; a CEO-quarter uses the last
10-K filed on or before the quarter start (docs/STAGE5_SPEC.md, section 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORMS_OK = ("10-K", "10-K405")
AMENDMENTS = tuple(f + "/A" for f in FORMS_OK)
AMEND_WINDOW = pd.Timedelta(days=400)     # an original filed this long before the amendment is "its" original

# candidate WRDS tables for the filing index, tried in order: (schema, table)
FILING_TABLES = [("wrdssec", "wrds_forms"), ("wrdssec_all", "wrds_forms"), ("wrdssec", "forms")]
# column aliases we accept for each field of the index
_ALIASES = {"cik": ("cik",), "form": ("form", "formtype"), "filing_date": ("fdate", "filing_date", "filingdate"),
            "accession": ("accession", "accession_number"), "fname": ("fname",)}
SEC_ARCHIVE = "https://www.sec.gov/Archives/"


def last_10k_before(index: pd.DataFrame, gvkey: str, date: pd.Timestamp, strict: bool = True) -> pd.Series | None:
    """Latest 10-K for `gvkey` filed before `date` (`<=` when strict is False), or None.

    Amendments (10-K/A) are used only when no original 10-K was filed in the 400 days before them.
    """
    d = index[index["gvkey"] == gvkey]
    d = d[d["filing_date"] < date] if strict else d[d["filing_date"] <= date]
    orig = d[d["form"].isin(FORMS_OK)]
    amend = d[d["form"].isin(AMENDMENTS)]
    if len(amend) and len(orig):
        of = orig["filing_date"].to_numpy()
        keep = [not ((of >= a - AMEND_WINDOW) & (of <= a)).any() for a in amend["filing_date"].to_numpy()]
        amend = amend[keep]
    d = pd.concat([orig, amend])
    if d.empty:
        return None
    return d.sort_values(["filing_date", "accession"]).iloc[-1]


def apply_link_window(index: pd.DataFrame) -> pd.DataFrame:
    """Keep a filing for a gvkey only inside the validity window of the gvkey-CIK link (open-ended if NaT)."""
    if "link_start" not in index.columns:
        return index
    after = index["link_start"].isna() | (index["filing_date"] >= index["link_start"])
    before = index["link_end"].isna() | (index["filing_date"] <= index["link_end"])
    return index[after & before].reset_index(drop=True)


def validate_links(index: pd.DataFrame, db=None, n_spot: int = 20, seed: int = 0) -> dict:
    """GVKEY-CIK link checks: gvkeys with several CIKs, CIKs shared by several gvkeys, and (when `db` is given)
    a random spot-check list pairing the Compustat name with the CIK for a manual name comparison."""
    pairs = index[["gvkey", "cik"]].dropna().drop_duplicates()
    per_g = pairs.groupby("gvkey")["cik"].nunique()
    per_c = pairs.groupby("cik")["gvkey"].nunique()
    rep = {"n_gvkeys": int(pairs["gvkey"].nunique()), "n_ciks": int(pairs["cik"].nunique()),
           "gvkey_multi_cik": sorted(per_g[per_g > 1].index), "cik_multi_gvkey": sorted(per_c[per_c > 1].index),
           "spot_check": []}
    if db is not None and len(pairs):
        names = _company_names(db)
        samp = pairs.sample(min(n_spot, len(pairs)), random_state=seed)
        rep["spot_check"] = [{"gvkey": g, "cik": c, "compustat_name": names.get(g, "")}
                             for g, c in zip(samp["gvkey"], samp["cik"])]
    return rep


def _company_names(db) -> dict:
    df = db.fetch("comp", "company", ["gvkey", "conm"], [])
    return dict(zip(df["gvkey"].astype(str).str.zfill(6), df["conm"]))


def _find_source(db):
    """First existing (schema, table) among FILING_TABLES with usable columns, plus the column map; else None."""
    for schema, table in FILING_TABLES:
        have = db.columns(schema, table)
        if not have:
            continue
        cols = {}
        for field, aliases in _ALIASES.items():
            hit = next((a for a in aliases if a in have), None)
            if hit:
                cols[field] = hit
        if {"cik", "form", "filing_date"} <= set(cols):
            return schema, table, cols
    return None


def _cik10(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def _clean_link(link: pd.DataFrame) -> pd.DataFrame:
    """One row per gvkey-cik pair with the hull of its dated windows. Undated rows are dropped when a dated row exists
    for the pair (they would otherwise reopen the window); a pair with only undated rows stays open-ended."""
    link = link.dropna(subset=["cik"]).assign(gvkey=lambda d: d["gvkey"].astype(str).str.zfill(6),
                                              cik=lambda d: _cik10(d["cik"]),
                                              link_start=lambda d: pd.to_datetime(d["link_start"]),
                                              link_end=lambda d: pd.to_datetime(d["link_end"]))
    link["_dated"] = link["link_start"].notna() | link["link_end"].notna()
    has_dated = link.groupby(["gvkey", "cik"])["_dated"].transform("any")
    link = link[~has_dated | link["_dated"]]
    g = link.groupby(["gvkey", "cik"], as_index=False)
    out = g.agg(link_start=("link_start", "min"), link_end=("link_end", "max"))
    # a pair with any open-ended dated row stays open at that end
    open_end = link[link["_dated"] & link["link_end"].isna()].groupby(["gvkey", "cik"]).size().rename("_oe")
    out = out.merge(open_end, on=["gvkey", "cik"], how="left")
    out.loc[out["_oe"].notna(), "link_end"] = pd.NaT
    return out.drop(columns="_oe")


def _fetch_10k_rows(db, schema, table, cols, ciks, cache_dir=None, chunk=100, forms=None):
    """Rows of the filing table for `ciks`, reduced to 10-K-type forms chunk by chunk. The WRDS REST API ignores
    filters on `form`, so every form comes back; filtering as each chunk arrives keeps memory and the on-disk cache
    small (only 10-K rows are kept in `cache_dir`)."""
    import hashlib
    from pathlib import Path
    forms = tuple(forms) if forms else FORMS_OK + AMENDMENTS
    parts = []
    for k in range(0, len(ciks), chunk):
        ch = ciks[k:k + chunk]
        path = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            path = Path(cache_dir) / ("forms10k_" + hashlib.md5(repr((schema, table, ch, forms)).encode()).hexdigest() + ".pkl")
            if path.exists():
                parts.append(pd.read_pickle(path))
                continue
        if hasattr(db, "_fetch_complete"):
            df = db._fetch_complete(schema, table, list(cols.values()), cols["cik"], ch, [])
        else:
            df = db.fetch_in(schema, table, list(cols.values()), cols["cik"], ch, [])
        df = df.rename(columns={v: k_ for k_, v in cols.items()})
        df = df[df["form"].isin(forms)]
        if path is not None:
            df.to_pickle(path)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(cols))


def build_filing_index(db, gvkeys=None, cache_dir=None, forms=None) -> pd.DataFrame:
    """Index of 10-K filings per gvkey: gvkey | cik | sic | form | filing_date | accession | url | link_start | link_end.

    Source: the WRDS SEC filing table (`wrdssec.wrds_forms`). The gvkey-CIK link comes from the dated
    `wrdssec.wciklink_gvkey` when it exists, so a filing counts for a gvkey only inside the link window; otherwise from
    `comp.company` (no dates, so the window is open). `gvkeys` restricts the pull to those firms. Stops with an error
    when no filing table exists, so the author can decide whether to use EDGAR form.idx directly.
    """
    src = _find_source(db)
    if src is None:
        raise RuntimeError("no SEC filing index table found in WRDS (tried "
                           f"{FILING_TABLES}); stop and ask the author whether to use EDGAR form.idx")
    schema, table, cols = src
    want = None if gvkeys is None else sorted({str(g).zfill(6) for g in gvkeys})
    comp = _fetch_by_gvkey(db, "comp", "company", ["gvkey", "sic", "cik"], want)
    comp["gvkey"] = comp["gvkey"].astype(str).str.zfill(6)
    sic = comp[["gvkey", "sic"]].drop_duplicates("gvkey")
    if db.columns("wrdssec", "wciklink_gvkey"):
        link = _fetch_by_gvkey(db, "wrdssec", "wciklink_gvkey", ["gvkey", "cik", "link_start_date", "link_end_date"], want)
        link = link.rename(columns={"link_start_date": "link_start", "link_end_date": "link_end"})
    else:
        link = comp[["gvkey", "cik"]].assign(link_start=pd.NaT, link_end=pd.NaT)
    link = _clean_link(link)
    f = _fetch_10k_rows(db, schema, table, cols, sorted(set(link["cik"])), cache_dir, forms=forms)
    f["cik"] = _cik10(f["cik"])
    f["filing_date"] = pd.to_datetime(f["filing_date"])
    if "accession" not in f.columns:
        f["accession"] = f["cik"] + "_" + f["filing_date"].dt.strftime("%Y%m%d") + "_" + f["form"]
    f["url"] = (SEC_ARCHIVE + f["fname"].astype(str)) if "fname" in f.columns else ""
    out = f.merge(link, on="cik", how="inner").merge(sic, on="gvkey", how="left")
    cols_out = ["gvkey", "cik", "sic", "form", "filing_date", "accession", "url", "link_start", "link_end"]
    out = apply_link_window(out[cols_out])
    return out.drop_duplicates(["gvkey", "accession"]).sort_values(["gvkey", "filing_date"]).reset_index(drop=True)


def _fetch_by_gvkey(db, schema, table, cols, gvkeys):
    have = [c for c in cols if c in db.columns(schema, table)]
    if gvkeys is None:
        return db.fetch(schema, table, have, [])
    return db.fetch_in(schema, table, have, "gvkey", gvkeys, [])


# ---------------------------------------------------------------- fetch and cache of extracted text
_last_request = [0.0]


def _http_get(url: str, user_agent: str) -> str:
    import requests
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60)
    r.raise_for_status()
    return r.text


def _filing_url(row: pd.Series) -> str:
    url = str(row.get("url", "") or "")
    if url:
        return url
    return f"{SEC_ARCHIVE}edgar/data/{int(row['cik'])}/{row['accession']}.txt"


def load_sections(row: pd.Series, names: list, cache_dir, user_agent: str, min_interval: float = 0.15,
                  fetch=None, tickers=()) -> tuple[str, str]:
    """(text, status) for one filing, status in {ok, limited_disclosure, extraction_failed, fetch_failed}.

    Only the extracted, identity-stripped text is cached (gzip JSON with the extraction version and the text's sha256);
    never the raw HTML. A cache entry from another extraction version or another name list is a miss. `fetch` is a
    `(url, user_agent) -> html` callable, injectable for tests.
    """
    import gzip
    import hashlib
    import json
    import time
    from pathlib import Path
    import gg_transparency as T
    cache_dir = Path(cache_dir)
    names_sha = hashlib.sha256(("\n".join(sorted(n for n in names if n)) + "|" + "\n".join(sorted(t for t in tickers if t))).encode()).hexdigest()[:16]
    path = cache_dir / f"{row['accession']}.json.gz"
    if path.exists():
        rec = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        if rec.get("extraction_version") == T.EXTRACTION_VERSION and rec.get("names_sha") == names_sha:
            return rec["text"], rec["status"]
    fetch = fetch or _http_get
    wait = min_interval - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.time()
    try:
        html = fetch(_filing_url(row), user_agent)
    except Exception:
        return "", "fetch_failed"
    text, status = T.extract_sections(T.select_docs(html, ("10-K", "EX-13")) or html)
    text = T.strip_identity(text, list(names), list(tickers))
    cache_dir.mkdir(parents=True, exist_ok=True)
    rec = {"accession": row["accession"], "status": status, "text": text, "extraction_version": T.EXTRACTION_VERSION,
           "names_sha": names_sha, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(rec))
    return text, status


def effective_index(index: pd.DataFrame) -> pd.DataFrame:
    """The filings last_10k_before can ever return: originals, plus amendments with no original filed in the 400 days
    up to them. Equivalent to applying the amendment rule per lookup, but vectorised for attaching T to whole panels."""
    orig = index[index["form"].isin(FORMS_OK)]
    amend = index[index["form"].isin(AMENDMENTS)]
    if amend.empty or orig.empty:
        return index[index["form"].isin(FORMS_OK + AMENDMENTS)].reset_index(drop=True) if orig.empty else orig.reset_index(drop=True)
    o = orig[["gvkey", "filing_date"]].rename(columns={"filing_date": "orig_date"}).sort_values("orig_date")
    a = amend.sort_values("filing_date")
    m = pd.merge_asof(a, o, left_on="filing_date", right_on="orig_date", by="gvkey", direction="backward")
    keep = m["orig_date"].isna() | ((m["filing_date"] - m["orig_date"]) > AMEND_WINDOW)
    return pd.concat([orig, m.loc[keep, a.columns]], ignore_index=True)


def sec_user_agent() -> str:
    """User-Agent for EDGAR requests (the SEC requires a descriptive agent with a contact). The contact comes from the local
    environment variable SEC_CONTACT_EMAIL (set in .env) and is used for SEC requests only."""
    import os
    contact = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not contact:
        raise RuntimeError("SEC_CONTACT_EMAIL is not set; EDGAR requires a contact in the User-Agent")
    return f"guidance-reversal research {contact}"



# ---------------------------------------------------------------- all company materials: 10-K + 10-Qs + 8-K exhibits
def bundle_documents(doc_index: pd.DataFrame, tenk_row: pd.Series, max_10q: int = 2, max_8k: int = 4, window_days: int = 365,
                     lead_days: int = 25):
    """The other materials known when `tenk_row` was filed: the latest `max_10q` 10-Qs filed in the `window_days` before it,
    and up to `max_8k` 8-Ks filed in the `lead_days` before a 10-Q or the 10-K (the earnings release and presentation
    exhibits). Nothing filed after the 10-K is ever included, so T keeps its no-look-ahead timing. Returns (10-Qs, 8-Ks)."""
    d = tenk_row["filing_date"]
    c = doc_index[(doc_index["gvkey"] == tenk_row["gvkey"]) & (doc_index["filing_date"] <= d)
                  & (doc_index["filing_date"] > d - pd.Timedelta(days=window_days))]
    qs = c[c["form"] == "10-Q"].sort_values("filing_date")
    anchors = list(qs["filing_date"]) + [d]
    k8 = c[c["form"] == "8-K"]
    mask = np.array([any(0 <= (a - x).days <= lead_days for a in anchors) for x in k8["filing_date"]], dtype=bool)
    near = k8[mask]                                          # a boolean mask keeps the columns even when k8 is empty
    return qs.tail(max_10q), near.sort_values("filing_date").tail(max_8k)


def load_supplement(row: pd.Series, kind: str, cap: int, names: list, cache_dir, user_agent: str, min_interval: float = 0.15,
                    fetch=None, tickers=()):
    """Cached, identity-stripped operating-statistics text from one 10-Q or 8-K filing ('' when it has none); None when the
    document could not be fetched (not cached, so a later run retries)."""
    import gzip
    import hashlib
    import json
    import time
    from pathlib import Path
    import gg_transparency as T
    cache_dir = Path(cache_dir)
    names_sha = hashlib.sha256(("\n".join(sorted(n for n in names if n)) + "|" + "\n".join(sorted(t for t in tickers if t))).encode()).hexdigest()[:16]
    path = cache_dir / f"{row['accession']}.sup.json.gz"
    if path.exists():
        rec = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        if rec.get("extraction_version") == T.EXTRACTION_VERSION and rec.get("names_sha") == names_sha and rec.get("cap") == cap:
            return rec["text"]
    fetch = fetch or _http_get
    wait = min_interval - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.time()
    try:
        raw = fetch(_filing_url(row), user_agent)
    except Exception:
        return None
    prefixes = ("10-Q",) if kind == "10-Q" else ("EX-99",)
    text = T.strip_identity(T.extract_supplement(T.select_docs(raw, prefixes), cap), list(names), list(tickers))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"accession": row["accession"], "kind": kind, "cap": cap, "text": text, "names_sha": names_sha,
                             "extraction_version": T.EXTRACTION_VERSION}))
    return text


def load_bundle(tenk_row: pd.Series, doc_index: pd.DataFrame, names: list, cache_dir, user_agent: str, min_interval: float = 0.15,
                fetch=None, tickers=(), max_10q: int = 2, max_8k: int = 4, cap_10q: int = 4_000, cap_8k: int = 6_000):
    """(text, status, used): the 10-K text (load_sections) followed, after BUNDLE_MARKER, by operating-statistics text from the
    10-Qs and from the earnings-release / presentation exhibits of the 8-Ks filed before it. `status` is the 10-K's; a
    supplementary document that cannot be fetched or has no statistics is skipped. `used` lists the accessions that
    contributed text."""
    import gg_transparency as T
    text, status = load_sections(tenk_row, names, cache_dir, user_agent, min_interval, fetch, tickers)
    if status in ("fetch_failed", "extraction_failed"):
        return "", status, []
    qs, earn = bundle_documents(doc_index, tenk_row, max_10q, max_8k)
    parts, used = [], []
    for kind, rows, cap, label in (("10-Q", qs, cap_10q, "10-Q"), ("8-K", earn, cap_8k, "8-K exhibit")):
        for r in rows.itertuples():
            sup = load_supplement(pd.Series({"accession": r.accession, "url": getattr(r, "url", ""), "cik": r.cik}), kind, cap, names,
                                  cache_dir, user_agent, min_interval, fetch, tickers)
            if sup:
                parts.append(f"=== {label} filed {str(r.filing_date)[:10]} ===\n{sup}")
                used.append(r.accession)
    return text + (T.BUNDLE_MARKER + "\n\n".join(parts) if parts else ""), status, used
