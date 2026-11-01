#!/usr/bin/env python3
"""
guide_and_go.py  --  "Guidance Reversal" screener
=============================================

Research question
-----------------
How often do executives (CEO / CFO) issue forward guidance, sell stock, and then
leave the firm (for government, a rival, PE, or "to pursue other interests")
before that guidance comes due -- and does the market punish the stock afterwards?

Can an *ex-ante* score built only from information available on the departure
date (recent guidance, guidance optimism vs. consensus, insider selling after
guidance, abruptness, destination) predict *ex-post* outcomes (guidance missed
after the exec is gone, negative long-run abnormal returns)?

Literature this design leans on
-------------------------------
* Cohen, Malloy & Pomorski (2012, JF) "Decoding Inside Information" --
  routine vs. opportunistic insider trades (used to classify sales).
* Jagolinzer (2009, Mgmt Sci); Larcker, Lynch, Quinn, Tayan & Taylor (2021)
  "Gaming the System" -- strategic 10b5-1 plan trading; SEC amended Rule 10b5-1
  in Dec 2022 (cooling-off periods) because of it.
* Rogers & Stocken (2005, TAR) -- credibility / bias of management forecasts.
* Larcker & Zakolyukina (2012, JAR) -- detecting deceptive conference-call language.
* Luechinger & Moser (2014, JPubE) -- stock-price effects of the "revolving door"
  between corporations and government appointments.
* Jenter & Kanaan (2015, JF); Gentry et al. (2021, SMJ) -- CEO turnover /
  dismissal classification.
* Cohen, Malloy & Nguyen (2020, JF) "Lazy Prices"; Lopez-Lira & Tang (2023);
  Kim, Muhn & Nikolaev (2024) -- text / LLM signals in disclosures, motivating the
  embedding + RAG layer below.

Pipeline
--------
1. WRDS pulls (cached to ./gg_cache):
   Execucomp departures, CCM link, CRSP daily + index, I/B/E/S management
   guidance + actuals, Thomson/Refinitiv insider filings (Form 4),
   Capital IQ Key Developments (exec-change / guidance press-release text),
   optional BoardEx (where the exec went next).
2. Feature engineering per departure event.
3. Event study: market-model CAR[-1,+1], BHAR[0,+126], BHAR[0,+252], CAR paths.
4. Embeddings: zero-shot classification of departure reason / destination
   (government, poached, forced, scandal, retirement, health...) from CIQ text.
5. Statistics: group tests, bootstrap CIs, OLS with clustered SEs,
   out-of-time gradient boosting vs. logit (AUC, permutation importance).
6. Screen: ranked list of Guidance Reversal cases + counts by year / sector.
7. RAG: vector index over "case cards" + CIQ text; `search` command answers
   natural-language queries, optionally with an LLM (Claude) over retrieved context.

Usage
-----
    pip install pandas numpy requests sqlalchemy psycopg2-binary scikit-learn statsmodels \
                matplotlib scipy python-dotenv sentence-transformers anthropic

    # credentials -- put in a .env file in the repo root (one level above src/):
    WRDS_API_KEY=...              # preferred: WRDS REST API token
    #   (or, instead: WRDS_USERNAME=... and WRDS_PASSWORD=...  for direct Postgres)
    # run everything from the repo root
    python src/guide_and_go.py test-api                               # verify the key first
    export ANTHROPIC_API_KEY=...  # optional, only for `search --llm`

    python src/guide_and_go.py build --start 2000 --end 2025          # full run
    python src/guide_and_go.py build --demo                           # synthetic dry run
    python src/guide_and_go.py build --start 2010 --max-events 500    # quick test
    python src/guide_and_go.py search "CEO raised guidance, sold shares, then left for a government post"
    python src/guide_and_go.py search "CFO resigned abruptly before guidance miss" --llm

Outputs go to ./outputs: data/ (screen.csv, events.csv, car_paths.pkl), results/ (report.md, figures), rag/ (index).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so gg_labels / gg_tests import from this folder
import gg_labels as GL
import gg_tests as GT

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # repo root (this file lives in src/)
CACHE = ROOT / "gg_cache"               # raw WRDS pulls
OUT = ROOT / "outputs"
DATA = OUT / "data"                     # events, screen, panels, return windows (inputs to the tests)
RESULTS = OUT / "results"               # *.json results, report, figures
RAG = OUT / "rag"                       # legacy RAG index


def make_output_dirs() -> None:
    for d in (DATA, RESULTS, RAG):
        d.mkdir(parents=True, exist_ok=True)


try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


# =============================================================================
# 1. WRDS access layer  (two interchangeable backends)
# =============================================================================
#   * API key  (preferred if set):  WRDS_API_KEY  -> REST API at wrds-api.wharton.upenn.edu
#   * Postgres (fallback):          WRDS_USERNAME + WRDS_PASSWORD (or ~/.pgpass)
#
# Every loader below calls  db.fetch(schema, table, columns, filters)  where filters are
# (column, op, value) tuples with op in {eq, in, gte, lte, icontains}. The API backend turns
# them into Django-REST style query params (col=v, col__in=a,b, col__gte=v, col__icontains=v);
# the Postgres backend turns them into a parameterised WHERE clause.
#
# API settings (from the WRDS "Accessing Data via the WRDS API" guide; override in .env):
#   WRDS_API_URL      default https://wrds-api.wharton.upenn.edu
#   WRDS_API_PATH     default /data/{schema}.{table}/
#   WRDS_API_AUTH     "param" (default: ?wrds_token=...) or "header" (Authorization: Token ...)
#   WRDS_API_LIMIT    rows per page, default 10000 (API default is only 10!)
#   WRDS_API_LOOKUPS  "1"/"0" to skip the start-up probe of date__gte / __in support
OPS = {"eq", "in", "gte", "lte", "icontains"}


class _Base:
    in_chunk = 400

    def __init__(self, refresh: bool = False):
        self.refresh = refresh
        CACHE.mkdir(exist_ok=True)
        self._cols: dict = {}

    def cached(self, name: str, fn):
        p = CACHE / f"{name}.pkl"
        if p.exists() and not self.refresh:
            return pd.read_pickle(p)
        print(f"  [wrds] pulling {name} ...", flush=True)
        df = fn()
        df.to_pickle(p)
        print(f"  [wrds] {name}: {len(df):,} rows", flush=True)
        return df

    def _superset_cache(self, prefix: str, want: set):
        """Path of an earlier pull (with a .vals sidecar) whose ID set contains `want`, else None."""
        if self.refresh:
            return None
        for q in sorted(CACHE.glob(f"{prefix}_*.vals")):
            if want <= set(q.read_text().split("\n")) and q.with_suffix(".pkl").exists():
                return q.with_suffix(".pkl")
        return None

    def has_cache_in(self, prefix: str, values, norm=str) -> bool:
        vals = sorted({norm(v) for v in values})
        return (CACHE / f"{prefix}_{idkey(vals)}.pkl").exists() or \
            self._superset_cache(prefix, set(vals)) is not None

    def cached_in(self, prefix: str, values, col: str, fn, norm=str):
        """Cache for a pull filtered to a set of IDs. A narrower request (e.g. fewer firms after a date
        filter, or a --max-events run) reuses any earlier pull whose ID set is a superset, filtering rows
        to the requested IDs, so it never re-downloads data it already has."""
        vals = sorted({norm(v) for v in values})
        p = CACHE / f"{prefix}_{idkey(vals)}.pkl"
        if not self.refresh and p.exists():
            return pd.read_pickle(p)
        src = self._superset_cache(prefix, set(vals))
        if src is not None:
            df = pd.read_pickle(src)
            if col in df.columns:
                print(f"  [cache] {prefix}: reusing {src.stem} ({len(vals):,} requested ids)", flush=True)
                return df[df[col].map(norm).isin(set(vals))].reset_index(drop=True)
        print(f"  [wrds] pulling {prefix} ...", flush=True)
        df = fn()
        df.to_pickle(p)
        p.with_suffix(".vals").write_text("\n".join(vals))
        print(f"  [wrds] {prefix}: {len(df):,} rows", flush=True)
        return df

    def find(self, candidates):
        for s, t in candidates:
            if self.columns(s, t):
                return s, t
        return None

    def pick(self, schema, table, wanted):
        have = self.columns(schema, table)
        return [c for c in wanted if c in have]

    truncates = False     # True for the REST API: it silently cuts off large result sets (see _fetch_complete)

    @staticmethod
    def _nid(v):
        try:
            return str(int(float(v)))
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _skey(s):
        try:
            return (0, float(s), "")
        except ValueError:
            return (1, 0.0, s)

    def _fetch_complete(self, schema, table, cols, col, ids, filters, depth=0):
        """fetch() for `col IN ids` that survives the REST API's silent result truncation.

        The API stops after roughly its own row-count estimate (a few hundred thousand rows) and returns results
        ordered by `col`, so the LAST firms in a large IN-list are simply missing and the last one returned may be
        cut mid-way. (This quietly dropped ~half of all firms' price histories in earlier runs.) So: keep everything
        before the last returned id, re-fetch the last id on its own, and recurse on the ids that sort after it.
        """
        df = self.fetch(schema, table, cols, [(col, "in", list(ids)), *filters])
        if not self.truncates or df.empty or col not in df.columns or len(ids) <= 1:
            return df
        ret = df[col].map(self._nid)
        by_id = {self._nid(v): v for v in ids}
        last = ret.iloc[-1]
        keys = [self._skey(s) for s in ret.values]
        monotonic = all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))
        if monotonic:
            lk = self._skey(last)
            suspects = [v for n, v in by_id.items() if self._skey(n) > lk]
        else:                                           # unknown server order: treat every absent id as suspect
            got = set(ret.values)
            suspects = [v for n, v in by_id.items() if n not in got]
        parts = [df[ret != last]]
        parts.append(self.fetch(schema, table, cols, [(col, "in", [by_id.get(last, last)]), *filters]))
        if suspects:
            parts.append(self._fetch_complete(schema, table, cols, col, suspects, filters, depth + 1))
        return pd.concat(parts, ignore_index=True)

    def fetch_in(self, schema, table, cols, col, values, filters=()):
        """fetch() with a long IN list split into chunks."""
        values = sorted({v for v in values if pd.notna(v)}, key=str)
        if not values:
            return pd.DataFrame(columns=cols)
        # Each chunk is checkpointed to disk, so a dropped connection / Ctrl-C mid-pull resumes
        # where it stopped instead of re-downloading everything.
        import hashlib
        ck = CACHE / "chunks"
        ck.mkdir(exist_ok=True)
        import time
        parts, t0, rows, done_new = [], time.time(), 0, 0
        todo = list(chunks(values, self.in_chunk))
        for n, ch in enumerate(todo, 1):
            # "v2": checkpoints written before the truncation fix (see _fetch_complete) are never reused
            key = hashlib.md5(repr(("v2", schema, table, list(cols), col, [str(v) for v in ch],
                                    [(c, o, str(v)) for c, o, v in filters])).encode()).hexdigest()
            p = ck / f"{schema}_{table}_{key}.pkl"
            if p.exists() and not self.refresh:
                parts.append(pd.read_pickle(p))
                rows += len(parts[-1])
                print(f"  [{schema}.{table}] chunk {n}/{len(todo)} from checkpoint "
                      f"(total {rows:,} rows)", flush=True)
                continue
            df = self._fetch_complete(schema, table, cols, col, ch, filters)
            df.to_pickle(p)
            parts.append(df)
            rows += len(df)
            done_new += 1
            el = time.time() - t0
            eta = el / done_new * (len(todo) - n)
            print(f"  [{schema}.{table}] chunk {n}/{len(todo)} done: {len(df):,} rows "
                  f"(total {rows:,}) | elapsed {el / 60:.1f} min, ETA ~{eta / 60:.0f} min", flush=True)
        return pd.concat(parts, ignore_index=True)


class WRDSPostgres(_Base):
    HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"

    def __init__(self, refresh=False):
        super().__init__(refresh)
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
        user, pw = os.environ.get("WRDS_USERNAME"), os.environ.get("WRDS_PASSWORD")
        if not user:
            sys.exit("Set WRDS_API_KEY, or WRDS_USERNAME (+ WRDS_PASSWORD / ~/.pgpass), in .env")
        url = URL.create("postgresql+psycopg2", username=user, password=pw, host=self.HOST,
                         port=self.PORT, database=self.DB, query={"sslmode": "require"})
        self.engine = create_engine(url, pool_pre_ping=True)
        print("  [wrds] backend: Postgres")

    def _sql(self, q, params=None, expanding=()):
        from sqlalchemy import text, bindparam
        stmt = text(q)
        if expanding:
            stmt = stmt.bindparams(*[bindparam(k, expanding=True) for k in expanding])
        with self.engine.connect() as c:
            return pd.read_sql(stmt, c, params=params or {})

    def columns(self, schema, table):
        key = (schema, table)
        if key not in self._cols:
            df = self._sql("select column_name from information_schema.columns "
                           "where table_schema=:s and table_name=:t", {"s": schema, "t": table})
            self._cols[key] = set(df.column_name.str.lower())
        return self._cols[key]

    def fetch(self, schema, table, cols, filters=()):
        have = self.columns(schema, table)
        cols = [c for c in cols if c in have]
        where, params, exp = [], {}, []
        for i, (c, op, v) in enumerate(filters):
            assert op in OPS and re.fullmatch(r"[a-z0-9_]+", c), (c, op)
            k = f"p{i}"
            def _py(x):                      # numpy scalars / integral floats -> plain Python (psycopg2 can't adapt np types)
                if hasattr(x, "item"):
                    x = x.item()
                if isinstance(x, float) and x.is_integer():
                    x = int(x)
                return x
            if op == "in":
                where.append(f"{c} in :{k}"); params[k] = [_py(x) for x in v]; exp.append(k)
            elif op == "icontains":
                where.append(f"{c} ilike :{k}"); params[k] = f"%{v}%"
            else:
                where.append(f"{c} {dict(eq='=', gte='>=', lte='<=')[op]} :{k}"); params[k] = _py(v)
        q = f"select {', '.join(cols)} from {schema}.{table}"
        if where:
            q += " where " + " and ".join(where)
        return self._sql(q, params, exp)


class WRDSApi(_Base):
    """WRDS REST API, per the WRDS 'Accessing Data via the WRDS API' guide:
         GET https://wrds-api.wharton.upenn.edu/data/<schema>.<table>/?col=val&limit=10000&wrds_token=<token>
       * token goes in the query string as `wrds_token` (header auth optional via WRDS_API_AUTH=header)
       * WITHOUT `limit` the API returns only 10 rows -- we always send it and follow `next` links
       * rows are under `results`
       Range/list lookups (date__gte, permno__in, ...) are NOT documented in that guide, so on start-up
       we probe them once. If the API ignores them we fall back to exact-match requests (one per value)
       and apply ranges client-side, so nothing silently downloads a whole table unfiltered."""
    in_chunk = 100   # keep URLs short
    truncates = True # the REST API silently truncates large results; fetch_in() compensates

    def __init__(self, refresh=False, probe=True):
        super().__init__(refresh)
        import requests
        self.base = os.environ.get("WRDS_API_URL", "https://wrds-api.wharton.upenn.edu").rstrip("/")
        self.path = os.environ.get("WRDS_API_PATH", "/data/{schema}.{table}/")
        self.token = os.environ["WRDS_API_KEY"].strip().rstrip(".")   # guide warns about a trailing "."
        self.auth = os.environ.get("WRDS_API_AUTH", "param").lower()   # "param" (documented) or "header"
        self.limit = int(os.environ.get("WRDS_API_LIMIT", "10000"))
        self.s = requests.Session()
        # "close": don't reuse pooled connections; the server appears to drop idle keep-alives,
        # which shows up as RemoteDisconnected on the next page request.
        self.s.headers.update({"Accept": "application/json", "Connection": "close"})
        if self.auth == "header":
            self.s.headers["Authorization"] = f"{os.environ.get('WRDS_API_AUTH_SCHEME', 'Token')} {self.token}"
        print(f"  [wrds] backend: REST API ({self.base}, token via {self.auth})")
        env = os.environ.get("WRDS_API_LOOKUPS")          # "1"/"0" skips the probe
        self.lookups_ok = (env == "1") if env in ("0", "1") else (self.probe_lookups() if probe else True)

    # -- low level -------------------------------------------------------------
    def _url(self, schema, table):
        return self.base + self.path.format(schema=schema, table=table)

    def _auth_params(self, url=""):
        return {} if self.auth == "header" or "wrds_token=" in url else {"wrds_token": self.token}

    def _get(self, url, params=None):
        import time
        params = {**(params or {}), **self._auth_params(url)}
        import requests
        for attempt in range(8):
            try:
                r = self.s.get(url, params=params, timeout=180)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                # server dropped the connection (long pulls do this) -> back off and retry
                wait = min(2 ** attempt, 60)
                self.n_drops = getattr(self, "n_drops", 0) + 1
                if attempt >= 1:   # a single quick retry is routine; only report repeated failures
                    print(f"    [wrds] connection still failing ({type(e).__name__}); "
                          f"retry {attempt + 1}/8 in {wait}s (total drops this run: {self.n_drops})",
                          flush=True)
                time.sleep(wait)
                r = None
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt); continue
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                sys.exit(f"WRDS API auth failed ({r.status_code}): {r.text[:300]}\n"
                         "Check WRDS_API_KEY (no trailing '.'), then run `python src/guide_and_go.py test-api`.")
            if r.status_code == 400:
                raise ValueError(f"WRDS API rejected query {r.url.split('wrds_token')[0]}: {r.text[:300]}")
            r.raise_for_status()
            return r.json()
        if r is None:
            raise ConnectionError(f"WRDS API unreachable after 8 attempts: {url}")
        r.raise_for_status()

    def _pages(self, url, params):
        rows, first, n = [], True, 0
        params = {**params, "limit": self.limit}
        while url:
            js = self._get(url, params if first else None)
            first = False
            if js is None:
                break
            if isinstance(js, list):
                rows.extend(js); break
            rows.extend(js.get("results", []))
            n += 1
            if n % 20 == 0:
                print(f"    ... {len(rows):,} rows" + (f" of {js['count']:,}" if "count" in js else ""), flush=True)
            url = js.get("next")
        return rows

    @staticmethod
    def _params(filters):
        p = {}
        for c, op, v in filters:
            assert op in OPS, op
            if op == "in":
                p[f"{c}__in"] = ",".join(str(x) for x in v)
            elif op == "eq":
                p[c] = str(v)
            else:
                p[f"{c}__{op}"] = str(v)
        return p

    def probe_lookups(self) -> bool:
        """Does the API honour date__gte / date__lte / __in?  Tested on the small CRSP index file."""
        try:
            js = self._get(self._url("crsp", "dsi"),
                           {"date__gte": "2024-01-02", "date__lte": "2024-01-05", "limit": 50})
            recs = (js or {}).get("results", js if isinstance(js, list) else [])
            dates = sorted(str(r.get("date"))[:10] for r in recs)
            ok = bool(dates) and dates[0] >= "2024-01-02" and dates[-1] <= "2024-01-05" and len(dates) <= 5
        except Exception as e:
            print(f"  [wrds] lookup probe error: {e}")
            ok = False
        print(f"  [wrds] range/list filters {'supported' if ok else 'NOT supported -> exact-match fallback'}")
        return ok

    def columns(self, schema, table):
        key = (schema, table)
        if key not in self._cols:
            try:
                js = self._get(self._url(schema, table), {"limit": 1})
            except Exception:
                js = None
            recs = js if isinstance(js, list) else (js or {}).get("results", [])
            self._cols[key] = {k.lower() for k in recs[0]} if recs else set()
        return self._cols[key]

    # -- public --------------------------------------------------------------------
    def fetch(self, schema, table, cols, filters=()):
        url = self._url(schema, table)
        if self.lookups_ok:
            rows = self._pages(url, self._params(filters))
            df = pd.DataFrame(rows)
        else:
            eq = [f for f in filters if f[1] == "eq"]
            ins = [f for f in filters if f[1] == "in"]
            local = [f for f in filters if f[1] in ("gte", "lte", "icontains")]
            if not eq and not ins:
                print(f"  [warn] {schema}.{table}: no exact-match filter; downloading the whole table "
                      f"(range filters applied locally). This can be slow.")
            # expand one IN-list into exact-match requests (others applied locally)
            if ins:
                col, _, vals = ins[0]
                local += ins[1:]
                parts = [pd.DataFrame(self._pages(url, self._params(eq + [(col, "eq", v)]))) for v in vals]
                df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            else:
                df = pd.DataFrame(self._pages(url, self._params(eq)))
            df.columns = [c.lower() for c in df.columns]
            for c, op, v in local:
                if c not in df or df.empty:
                    continue
                if op == "in":
                    df = df[df[c].astype(str).isin([str(x) for x in v])]
                elif op == "icontains":
                    df = df[df[c].astype(str).str.contains(str(v), case=False, na=False, regex=False)]
                else:
                    x = pd.to_numeric(df[c], errors="coerce")
                    vv = pd.to_numeric(pd.Series([v]), errors="coerce")[0]
                    if pd.isna(vv) or x.isna().all():   # treat as dates / strings
                        x, vv = df[c].astype(str).str[:10], str(v)[:10]
                    df = df[x >= vv] if op == "gte" else df[x <= vv]
        if df.empty:
            return pd.DataFrame(columns=cols)
        df.columns = [c.lower() for c in df.columns]
        return df[[c for c in cols if c in df.columns]]


def WRDS(refresh: bool = False):
    """Backend choice. Direct Postgres is preferred when WRDS_USERNAME/WRDS_PASSWORD are set: it runs the query
    server-side and returns COMPLETE results. The REST API silently truncates large results (it quietly dropped
    about half of all firms' CRSP histories), so it is only a fallback (GG_BACKEND=api forces it)."""
    want_pg = bool(os.environ.get("WRDS_USERNAME") and os.environ.get("WRDS_PASSWORD")) and \
        os.environ.get("GG_BACKEND", "").lower() != "api"
    if want_pg:
        for attempt in range(3):
            try:
                pg = WRDSPostgres(refresh)
                pg._sql("select 1")
                return pg
            except Exception as e:
                print(f"  [wrds] Postgres connection attempt {attempt + 1}/3 failed: {str(e).strip().splitlines()[0][:120]}")
                import time
                time.sleep(5 * (attempt + 1))
        print("  [wrds] Postgres unavailable -> falling back to the REST API (with truncation repair)")
    return WRDSApi(refresh) if os.environ.get("WRDS_API_KEY") else WRDSPostgres(refresh)


def test_api():
    """Mirrors the WRDS guide's example (Netflix, CUSIP 64110L10 in crsp.dsf), then probes filters."""
    if not os.environ.get("WRDS_API_KEY"):
        sys.exit("WRDS_API_KEY not set (put it in .env next to this script).")
    db = WRDSApi(probe=False)
    url = db._url("crsp", "dsf")
    print("GET", url, "?cusip=64110L10&limit=5&wrds_token=***")
    r = db.s.get(url, params={"cusip": "64110L10", "limit": 5, **db._auth_params()}, timeout=60)
    print("status:", r.status_code)
    print(r.text[:1200])
    if not r.ok:
        sys.exit("\nFailed. 401/403 = bad token (copy it again without the trailing '.'); "
                 "404 = no CRSP subscription via the API, or WRDS_API_PATH differs.")
    js = r.json()
    print("\nkeys in response:", list(js) if isinstance(js, dict) else "list",
          "| rows:", js.get("count", "n/a") if isinstance(js, dict) else len(js))
    ok = db.probe_lookups()
    print("\nOK - token works." + ("" if ok else
          " Range filters unsupported: the build will use exact-match requests (slower but correct)."))
    print("Next: python src/guide_and_go.py build --start 2010 --max-events 300")


def _nint(v) -> str:
    return str(int(float(v)))


def _ngv(v) -> str:
    return str(v).zfill(6)


def idkey(ids) -> str:
    """Short stable hash of an ID set, so firm-specific caches never collide across different samples
    (e.g. a 300-event test run vs the full build)."""
    import hashlib
    return hashlib.md5(",".join(sorted(str(i) for i in ids)).encode()).hexdigest()[:10]


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _range(col, a, b):
    return [(col, "gte", a), (col, "lte", b)]


# --- individual pulls --------------------------------------------------------
def load_departures(db, start: int, end: int, roles: set) -> pd.DataFrame:
    loc = db.find([("comp_execucomp", "anncomp"), ("execcomp", "anncomp"), ("comp", "anncomp")])
    if loc is None:
        sys.exit("Execucomp anncomp not found in your WRDS subscription.")
    s, t = loc
    cols = ["gvkey", "year", "execid", "exec_fullname", "coname", "leftofc", "leftco", "becameceo",
            "reason", "ceoann", "cfoann", "pceo", "pcfo", "titleann"]
    d = db.cached(f"execucomp_{start}_{end}", lambda: db.fetch(
        s, t, cols, _range("leftofc", f"{start}-01-01", f"{end}-12-31")))
    d.columns = d.columns.str.lower()
    # The REST API ignores the leftofc range filter on this table, so apply the date window here;
    # otherwise --start 2010 still returns departures back to 1990 that have no price/guidance data.
    d["leftofc"] = pd.to_datetime(d.leftofc, errors="coerce")
    n_dated = int(d.leftofc.notna().sum())
    d = d[(d.leftofc >= pd.Timestamp(f"{start}-01-01")) & (d.leftofc <= pd.Timestamp(f"{end}-12-31"))].copy()
    print(f"  departure date filter {start}-{end}: kept {len(d):,} of {n_dated:,} dated ExecuComp departure rows")
    blank = pd.Series("", index=d.index)
    title = d.get("titleann", blank).fillna("").str.upper()
    is_ceo = (d.get("ceoann", blank).fillna("").eq("CEO") | d.get("pceo", blank).fillna("").eq("CEO")
              | title.str.contains(r"\bCEO\b|CHIEF EXECUTIVE"))
    is_cfo = (d.get("cfoann", blank).fillna("").eq("CFO") | d.get("pcfo", blank).fillna("").eq("CFO")
              | title.str.contains(r"\bCFO\b|CHIEF FINANCIAL"))
    d["role"] = np.where(is_ceo, "CEO", np.where(is_cfo, "CFO", "OTHER"))
    d["role_rank"] = d.role.map({"CEO": 0, "CFO": 1, "OTHER": 2})
    # role in the executive's LAST ExecuComp row (what they were when they left). `role` above gives CEO priority if the
    # executive held that role in ANY year, so ex-CEOs leaving as chairman/advisor are labelled CEO.
    key = ["gvkey", "execid", "leftofc"]
    last_role = d.sort_values("year").groupby(key).role.last().rename("role_last")
    d = d.sort_values("role_rank").drop_duplicates(key)
    d = d.join(last_role, on=key)
    d = d[d.role.isin(roles)].copy()
    d["leftofc"] = pd.to_datetime(d.leftofc)
    return pd.DataFrame({
        "gvkey": d.gvkey.astype(str).str.zfill(6), "execid": d.execid.astype(str),
        "exec_name": d.exec_fullname, "role": d.role, "role_last": d.role_last, "leftofc": d.leftofc,
        "reason_ec": d.get("reason", pd.Series(np.nan, index=d.index)),
        "conm": d.get("coname", blank),
    }).reset_index(drop=True)


def link_gvkey_permno(db, dep: pd.DataFrame) -> pd.DataFrame:
    lk = db.cached("ccm_link", lambda: db.fetch(
        "crsp", "ccmxpf_lnkhist", ["gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"],
        [("linktype", "in", ["LU", "LC"]), ("linkprim", "in", ["P", "C"])]))
    lk = lk.rename(columns={"lpermno": "permno"}).dropna(subset=["permno"])
    lk["gvkey"] = lk.gvkey.astype(str).str.zfill(6)
    lk["linkdt"] = pd.to_datetime(lk.linkdt)
    lk["linkenddt"] = pd.to_datetime(lk.linkenddt, errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    m = dep.merge(lk[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    m = m[(m.leftofc >= m.linkdt) & (m.leftofc <= m.linkenddt)]
    return m.drop(columns=["linkdt", "linkenddt"]).drop_duplicates(["gvkey", "execid", "leftofc"])


def load_crsp(db, permnos, start: int, end: int):
    permnos = sorted({int(p) for p in permnos})
    a, b = f"{start - 2}-01-01", f"{min(end + 2, 2100)}-12-31"
    crsp = db.cached_in(f"crsp_dsf_v2_{start}_{end}", permnos, "permno", lambda: db.fetch_in(
        "crsp", "dsf", ["permno", "date", "ret", "prc", "shrout"], "permno", permnos, _range("date", a, b)),
        norm=_nint)
    mkt = db.cached(f"crsp_dsi_{start}_{end}", lambda: db.fetch(
        "crsp", "dsi", ["date", "vwretd"], _range("date", a, b)).rename(columns={"vwretd": "mkt"}))
    names = db.cached_in("crsp_stocknames_v2", permnos, "permno", lambda: db.fetch_in(
        "crsp", "stocknames", ["permno", "ncusip", "ticker", "comnam", "siccd", "namedt", "nameenddt"],
        "permno", permnos), norm=_nint)
    for df in (crsp, mkt):
        df["date"] = pd.to_datetime(df.date)
    crsp["permno"] = crsp.permno.astype(int)
    names["permno"] = names.permno.astype(int)
    for c in ["ret", "prc", "shrout"]:
        crsp[c] = pd.to_numeric(crsp[c], errors="coerce")
    mkt["mkt"] = pd.to_numeric(mkt.mkt, errors="coerce")
    crsp["prc"] = crsp.prc.abs()
    # fold CRSP delisting returns into the daily series, so failing firms keep their final crash
    if db.columns("crsp", "dsedelist"):
        dl = db.cached_in("crsp_dsedelist_v2", permnos, "permno", lambda: db.fetch_in(
            "crsp", "dsedelist", ["permno", "dlstdt", "dlstcd", "dlret"], "permno", permnos), norm=_nint)
        n0 = int(crsp.ret.notna().sum())
        crsp = GT.apply_delisting(crsp, dl)
        print(f"  delisting returns applied for {len(dl):,} delisted securities")
    return crsp, mkt, names


def load_splits(db, permnos, start: int) -> pd.DataFrame:
    """CRSP price-adjustment events (splits, reverse splits, stock dividends): permno, exdt, f = 1 + facpr."""
    cols = ["permno", "exdt", "facpr"]
    if not db.columns("crsp", "dsedist"):
        print("  [warn] crsp.dsedist unavailable; guidance cannot be rebased for splits.")
        return pd.DataFrame(columns=["permno", "exdt", "f"])
    permnos = sorted({int(p) for p in permnos})
    d = db.cached_in(f"crsp_dsedist_v2_{start}", permnos, "permno", lambda: db.fetch_in(
        "crsp", "dsedist", cols, "permno", permnos, _range("exdt", f"{start - 2}-01-01", "2100-12-31")),
        norm=_nint)
    if d.empty:
        return pd.DataFrame(columns=["permno", "exdt", "f"])
    d["permno"] = d.permno.astype(int)
    d["exdt"] = pd.to_datetime(d.exdt).astype("datetime64[ns]")
    d["facpr"] = pd.to_numeric(d.facpr, errors="coerce")
    d = d[d.facpr.notna() & (d.facpr != 0) & (d.facpr > -1)]
    d["f"] = 1 + d.facpr
    return d[["permno", "exdt", "f"]].sort_values(["permno", "exdt"])


def split_factor_at(df: pd.DataFrame, splits: pd.DataFrame, date_col: str) -> pd.Series:
    """F(t) = product of (1 + facpr) over this firm's splits with ex-date after t.
    I/B/E/S guidance is stored on today's split-adjusted basis, whereas actuals and CRSP prices are as
    originally reported; value_as_reported = value_adjusted * F(t)."""
    F = pd.Series(1.0, index=df.index)
    if splits.empty:
        return F
    by_p = {p: s for p, s in splits.groupby("permno")}
    dates = pd.to_datetime(df[date_col]).astype("datetime64[ns]")
    for p, idx in df.groupby("permno").groups.items():
        s = by_p.get(int(p))
        if s is None:
            continue
        suffix = np.append(np.cumprod(s.f.values[::-1])[::-1], 1.0)      # suffix[i] = prod(f[i:])
        pos = np.searchsorted(s.exdt.values, dates.loc[idx].values, side="right")  # first ex-date > t
        F.loc[idx] = suffix[pos]
    return F


MAX_GAP = 1.0   # an EPS gap above 100% of the share price is treated as a data error, not a real miss


def load_guidance(db, permnos, start: int, end: int) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["permno", "anndats", "pdicity", "period_end", "mid", "consensus", "actual"])
    link_loc = db.find([("wrdsapps_link_crsp_ibes", "ibcrsphist"), ("wrdsapps", "ibcrsphist")])
    if link_loc is None or not db.columns("ibes", "det_guidance"):
        print("  [warn] I/B/E/S guidance or IBES-CRSP link unavailable; guidance features empty.")
        return empty
    link = db.cached("ibes_crsp_link", lambda: db.fetch(
        *link_loc, ["ticker", "permno", "sdate", "edate", "score"], [("score", "lte", 5)]))
    link["permno"] = pd.to_numeric(link.permno, errors="coerce")
    link = link[link.permno.isin(list(permnos))]
    tickers = link.ticker.unique()
    gcols = ["ticker", "measure", "pdicity", "prd_yr", "prd_mon", "anndats", "val_1", "val_2",
             "mean_at_date", "range_desc", "diff_code"]
    g = db.cached_in(f"ibes_guidance_v2_{start}_{end}", tickers, "ticker", lambda: db.fetch_in(
        "ibes", "det_guidance", gcols, "ticker", tickers,
        [("measure", "eq", "EPS"), *_range("anndats", f"{start - 1}-01-01", f"{end}-12-31")]))
    act = db.cached_in(f"ibes_actuals_v2_{start}_{end}", tickers, "ticker", lambda: db.fetch_in(
        "ibes", "actu_epsus", ["ticker", "pdicity", "pends", "value"], "ticker", tickers,
        [("measure", "eq", "EPS"), ("pends", "gte", f"{start - 1}-01-01")]))
    if g.empty:
        return empty
    act = act.rename(columns={"value": "actual"})
    for c in ["val_1", "val_2", "mean_at_date", "prd_yr", "prd_mon"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    act["actual"] = pd.to_numeric(act.actual, errors="coerce")
    g["anndats"] = pd.to_datetime(g.anndats)
    g = g.dropna(subset=["prd_yr", "prd_mon"])
    g["period_end"] = (pd.to_datetime(dict(year=g.prd_yr.astype(int), month=g.prd_mon.astype(int), day=1),
                                      errors="coerce") + pd.offsets.MonthEnd(0))
    g["mid"] = np.where(g.val_2.notna(), (g.val_1 + g.val_2) / 2, g.val_1)
    g = g.rename(columns={"mean_at_date": "consensus"})
    act["pends"] = pd.to_datetime(act.pends) + pd.offsets.MonthEnd(0)
    g = g.merge(act.rename(columns={"pends": "period_end"}).drop_duplicates(["ticker", "pdicity", "period_end"]),
                on=["ticker", "pdicity", "period_end"], how="left")
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    g = g.merge(link[["ticker", "permno", "sdate", "edate"]], on="ticker")
    g = g[(g.anndats >= g.sdate) & (g.anndats <= g.edate)]
    g["permno"] = g.permno.astype(int)
    g = g.reset_index(drop=True)
    # I/B/E/S guidance + consensus are stored on today's split-adjusted basis; actuals (and CRSP prices) are
    # as originally reported. Put everything on the as-reported basis at the GUIDANCE date, which is the
    # basis of the price used to scale gaps:
    #   guidance, consensus: x_adj * F(guidance date)
    #   actual (reported after the period end): x_rep / F(period end) * F(guidance date)
    # A split between the guidance date and the period end therefore cancels (the actual is already post-split).
    splits = load_splits(db, permnos, start)
    Fg = split_factor_at(g, splits, "anndats")
    Fa = split_factor_at(g, splits, "period_end")
    g["mid"] = g["mid"] * Fg
    g["consensus"] = g["consensus"] * Fg
    g["actual"] = g["actual"] / Fa * Fg
    print(f"  guidance rebased for splits: {int((Fg != 1).sum()):,} of {len(g):,} rows adjusted "
          f"({int((Fg != Fa).sum()):,} had a split between guidance date and period end)")
    return g[["permno", "anndats", "pdicity", "period_end", "mid", "consensus", "actual"]].dropna(subset=["mid"])


def load_insiders(db, permnos, names: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    # Thomson Reuters insider filings live in schema `tfn` on current WRDS (older: tr_insiders)
    loc = db.find([("tfn", "table1"), ("tr_insiders", "table1")])
    if loc is None:
        print("  [warn] Thomson insider table (tfn.table1 / tr_insiders.table1) unavailable; "
              "insider features empty.")
        return pd.DataFrame(columns=["permno", "personid", "owner", "trandate", "acqdisp", "shares", "tprice"])
    nm = names[names.permno.isin(list(permnos))].dropna(subset=["ncusip"]).copy()
    nm["ncusip"] = nm.ncusip.str[:8]
    nm["cusip6"] = nm.ncusip.str[:6]
    ins_prefix = f"tr_insiders_{start}_{end}"
    if isinstance(db, WRDSApi):
        # The REST API ignores every filter on this 17M-row table (each query returns it all,
        # ~3.5h), so use direct Postgres for it when a WRDS login is available.
        if os.environ.get("WRDS_USERNAME") and os.environ.get("WRDS_PASSWORD"):
            db = WRDSPostgres(db.refresh)
        elif not db.has_cache_in(ins_prefix, nm.cusip6.unique()):
            print("  [warn] REST API cannot filter the insider table (17M rows). Add WRDS_USERNAME and "
                  "WRDS_PASSWORD to .env to pull it via Postgres; insider features left empty for now.")
            return pd.DataFrame(columns=["permno", "personid", "owner", "trandate", "acqdisp", "shares",
                                         "tprice"])
    ins = db.cached_in(ins_prefix, nm.cusip6.unique(), "cusip6", lambda: db.fetch_in(
        *loc,
        ["personid", "owner", "cusip6", "cusip2", "rolecode1", "trandate", "acqdisp", "trancode",
         "shares", "tprice"], "cusip6", nm.cusip6.unique(),
        [("trancode", "in", ["S", "P"]), ("cleanse", "in", ["R", "H"]),
         *_range("trandate", f"{start - 4}-01-01", f"{end}-12-31")]))
    if ins.empty:
        return pd.DataFrame(columns=["permno", "personid", "owner", "trandate", "acqdisp", "shares", "tprice"])
    ins["ncusip"] = ins.cusip6.astype(str) + ins.cusip2.astype(str)
    ins = ins.merge(nm[["ncusip", "permno"]].drop_duplicates(), on="ncusip")
    ins["trandate"] = pd.to_datetime(ins.trandate)
    ins["permno"] = ins.permno.astype(int)
    for c in ["shares", "tprice"]:
        ins[c] = pd.to_numeric(ins[c], errors="coerce")
    return ins[["permno", "personid", "owner", "trandate", "acqdisp", "shares", "tprice"]]


def load_keydev(db, gvkeys, start: int, end: int) -> pd.DataFrame:
    cols_out = ["gvkey", "keydevid", "announcedate", "eventtype", "headline", "situation"]
    loc = db.find([("ciq_keydev", "wrds_keydev"), ("ciq", "wrds_keydev")])
    if loc is None:
        print("  [warn] Capital IQ Key Developments unavailable; text features limited.")
        return pd.DataFrame(columns=cols_out)
    sit = db.find([(loc[0], "ciqkeydev"), ("ciq", "ciqkeydev")])

    def q():
        k = db.fetch_in(*loc, ["gvkey", "keydevid", "announcedate", "eventtype", "headline"], "gvkey",
                        gvkeys, _range("announcedate", f"{start - 1}-01-01", f"{end + 1}-12-31"))
        k = k[k.eventtype.str.contains("Executive|Guidance|Restatement|Lawsuit", case=False, na=False)]
        k = k.drop_duplicates("keydevid")
        if sit and len(k):
            s = db.fetch_in(*sit, ["keydevid", "situation"], "keydevid", k.keydevid.unique())
            k = k.merge(s.drop_duplicates("keydevid"), on="keydevid", how="left")
        else:
            k["situation"] = None
        return k

    k = db.cached_in(f"ciq_keydev_v2_{start}_{end}", gvkeys, "gvkey", q, norm=_ngv)
    k["announcedate"] = pd.to_datetime(k.announcedate)
    k["gvkey"] = k.gvkey.astype(str).str.zfill(6)
    return k.drop_duplicates("keydevid")[cols_out]


def load_boardex_destinations(db, cases: pd.DataFrame) -> pd.DataFrame:
    """For a (small) set of cases, find BoardEx roles started within [-90,+365]d of departure
    at a different organisation. Name-based matching; treat as indicative."""
    emp = db.find([("boardex", "na_dir_profile_emp"), ("boardex_na", "na_dir_profile_emp")])
    det = db.find([("boardex", "na_dir_profile_details"), ("boardex_na", "na_dir_profile_details")])
    if emp is None or det is None:
        print("  [warn] BoardEx unavailable; skipping destinations.")
        return pd.DataFrame(columns=["event_id", "dest_org", "dest_role"])
    rows = []
    for _, r in cases.iterrows():
        last, first = last_name(r.exec_name), first_name(r.exec_name)
        if not last:
            continue
        try:
            people = db.fetch(*det, ["directorid", "directorname"], [("directorname", "icontains", last)])
            people = people[people.directorname.str.upper().str.contains(first[:3], na=False)]
            if people.empty:
                continue
            d = db.fetch_in(*emp, ["directorid", "companyname", "rolename", "datestartrole"],
                            "directorid", people.directorid.unique())
        except Exception:
            continue
        if d.empty:
            continue
        d["datestartrole"] = pd.to_datetime(d.datestartrole, errors="coerce")
        firm_tok = str(r.conm).split(" ")[0].upper()
        at_firm = d.companyname.str.upper().str.contains(re.escape(firm_tok), na=False)
        d = d[d.directorid.isin(d[at_firm].directorid.unique())]
        at_firm = d.companyname.str.upper().str.contains(re.escape(firm_tok), na=False)
        win = d[(d.datestartrole >= r.dep_ann - pd.Timedelta(days=90)) &
                (d.datestartrole <= r.dep_ann + pd.Timedelta(days=365)) & ~at_firm]
        for _, w in win.iterrows():
            rows.append({"event_id": r.event_id, "dest_org": w.companyname, "dest_role": w.rolename})
    return pd.DataFrame(rows)


# =============================================================================
# 2. Synthetic data (for --demo dry runs without WRDS)
# =============================================================================
def synthetic(seed=7, n_firms=250, n_dep=450):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-01", "2025-12-31")
    mkt = pd.DataFrame({"date": dates, "mkt": rng.normal(0.0003, 0.01, len(dates))})
    firms = pd.DataFrame({"permno": np.arange(10001, 10001 + n_firms),
                          "gvkey": [f"{i:06d}" for i in range(1, n_firms + 1)],
                          "conm": [f"FIRM{i} CORP" for i in range(1, n_firms + 1)],
                          "beta": rng.uniform(0.6, 1.5, n_firms)})
    first = ["John", "Mary", "Albert", "Linda", "Rex", "Henry", "Susan", "Jeff", "Paula", "Mark"]
    last = ["Smith", "Lee", "Garcia", "Patel", "Nguyen", "Brown", "Kim", "Rossi", "Cohen", "Okafor"]
    dep = []
    for i in range(n_dep):
        f = firms.iloc[rng.integers(n_firms)]
        dep.append({"gvkey": f.gvkey, "permno": f.permno, "conm": f.conm, "execid": str(50000 + i),
                    "exec_name": f"{rng.choice(first)} {rng.choice(last)}{i}",
                    "role": rng.choice(["CEO", "CFO"], p=[0.6, 0.4]),
                    "leftofc": dates[rng.integers(400, len(dates) - 400)],
                    "reason_ec": rng.choice(["RESIGNED", "RETIRED", "UNKNOWN"], p=[.4, .4, .2]),
                    "gg": rng.random() < 0.15})
    dep = pd.DataFrame(dep)
    # returns with planted post-departure drift for Guidance Reversal cases
    crsp = []
    for _, f in firms.iterrows():
        r = f.beta * mkt.mkt.values + rng.normal(0, 0.018, len(dates))
        for _, d in dep[(dep.permno == f.permno) & dep.gg].iterrows():
            i0 = dates.searchsorted(d.leftofc)
            r[i0:i0 + 252] -= 0.0012
            r[i0 - 1:i0 + 2] -= 0.015
        prc = 50 * np.exp(np.cumsum(r))
        crsp.append(pd.DataFrame({"permno": f.permno, "date": dates, "ret": r, "prc": prc,
                                  "shrout": 1e5}))
    crsp = pd.concat(crsp, ignore_index=True)
    # guidance
    g = []
    for _, f in firms.iterrows():
        for d in dates[rng.integers(0, len(dates), 30)]:
            cons = rng.normal(2, 0.5)
            g.append({"permno": f.permno, "anndats": d, "pdicity": "ANN",
                      "period_end": d + pd.offsets.YearEnd(1), "consensus": cons,
                      "mid": cons + rng.normal(0, .05), "actual": cons + rng.normal(0, .1)})
    for _, d in dep[dep.gg].iterrows():
        cons = rng.normal(2, 0.5)
        ann = d.leftofc - pd.Timedelta(days=int(rng.integers(20, 150)))
        g.append({"permno": d.permno, "anndats": ann, "pdicity": "ANN",
                  "period_end": ann + pd.offsets.YearEnd(1), "consensus": cons,
                  "mid": cons + abs(rng.normal(.25, .1)), "actual": cons - abs(rng.normal(.3, .1))})
    guid = pd.DataFrame(g)
    # insiders
    ins = []
    for _, d in dep.iterrows():
        nm = d.exec_name.split()
        owner = f"{nm[1].upper()} {nm[0].upper()}"
        for y in range(1, 4):  # routine history
            if rng.random() < .5:
                ins.append({"permno": d.permno, "personid": d.execid, "owner": owner,
                            "trandate": d.leftofc - pd.DateOffset(years=y), "acqdisp": "D",
                            "shares": 2000, "tprice": 50})
        if d.gg or rng.random() < .2:
            ins.append({"permno": d.permno, "personid": d.execid, "owner": owner,
                        "trandate": d.leftofc - pd.Timedelta(days=int(rng.integers(5, 15))),
                        "acqdisp": "D", "shares": 40000 if d.gg else 3000, "tprice": 50})
    ins = pd.DataFrame(ins)
    # key developments text
    kd, kid = [], 1
    tmpl_gg = ["{c} announced that CEO {n} will step down effective immediately to accept a "
               "nomination as Secretary in the incoming administration.",
               "{c} said {n} resigned to pursue other opportunities; the board named an interim CEO.",
               "{c} disclosed that {n} has left the company following an internal review."]
    tmpl_norm = ["{c} announced {n} will retire at year end after a planned succession process.",
                 "{c} appointed a new CFO as {n} retires after 20 years with the company.",
                 "{n} will step down for health reasons, {c} said."]
    for _, d in dep.iterrows():
        t = rng.choice(tmpl_gg if d.gg else tmpl_norm)
        kd.append({"gvkey": d.gvkey, "keydevid": kid, "eventtype": "Executive Changes - CEO",
                   "announcedate": d.leftofc - pd.Timedelta(days=int(rng.integers(0, 10))),
                   "headline": f"{d.conm} Announces Departure of {d.exec_name}",
                   "situation": t.format(c=d.conm, n=d.exec_name)})
        kid += 1
    keydev = pd.DataFrame(kd)
    names = pd.DataFrame({"permno": firms.permno, "siccd": rng.choice([2834, 3674, 6021, 1311, 7372],
                                                                      n_firms),
                          "comnam": firms.conm, "ticker": [f"T{i}" for i in range(n_firms)]})
    return dep.drop(columns="gg"), crsp, mkt, guid, ins, keydev, names


# =============================================================================
# 3. Text + embeddings
# =============================================================================
SUFFIX = {"JR", "SR", "II", "III", "IV", "PHD", "MD", "CPA", "ESQ", "MBA", "DR", "MR", "MS", "MRS"}


def last_name(n):
    return GL.last_name(n)          # credentials after the comma (J.D., Ph.D., CFA ...) no longer break this


def first_name(n):
    return GL.first_name(n)


def name_match(exec_name, owner) -> bool:
    return GL.match_owner(exec_name, owner) >= 3


class Embedder:
    """Sentence-transformer embeddings with TF-IDF fallback (so the app always runs)."""

    def __init__(self, model="BAAI/bge-small-en-v1.5"):
        self.kind = "tfidf"
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
            self.kind = "st"
            self.name = model
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Bounded vocabulary + SVD keep the dense matrix small (a full bigram TF-IDF
            # .toarray() on tens of thousands of docs exhausts RAM).
            self.model = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                         sublinear_tf=True, stop_words="english",
                                         dtype=np.float32)
            self.name = "tfidf"
            self.svd = None
            self._fitted = False

    SVD_DIM = 256

    def fit(self, corpus):
        if self.kind == "tfidf":
            from sklearn.decomposition import TruncatedSVD
            corpus = list(corpus)
            try:
                Xs = self.model.fit_transform(corpus)
            except ValueError:  # min_df=2 emptied the vocabulary (tiny corpus)
                self.model.set_params(min_df=1)
                Xs = self.model.fit_transform(corpus)
            k = min(self.SVD_DIM, Xs.shape[0] - 1, Xs.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=k, random_state=0).fit(Xs) if k >= 2 else None
            self._fitted = True
        return self

    def encode(self, texts) -> np.ndarray:
        texts = [str(t) for t in texts]
        if self.kind == "st":
            return np.asarray(self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                                show_progress_bar=False), dtype=np.float32)
        if not self._fitted:
            self.fit(texts)
        Xs = self.model.transform(texts)
        X = (self.svd.transform(Xs) if self.svd is not None else Xs.toarray()).astype(np.float32)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n == 0, 1, n)


# Zero-shot label prototypes (embedding similarity) + high-precision regex overrides
LABELS = {
    "government": "executive resigned to accept nomination or appointment to government: secretary, "
                  "cabinet, white house, administration, ambassador, federal agency, treasury, commission",
    "poached": "executive resigned to become chief executive of another company or competitor",
    "forced": "board terminated or ousted the executive; departure effective immediately; interim ceo named",
    "scandal": "departure following internal investigation, misconduct, accounting irregularities, "
               "SEC inquiry, restatement or lawsuit",
    "retirement": "executive will retire after planned succession process at year end",
    "health_personal": "executive stepping down for health or personal or family reasons",
    "other_opportunities": "executive resigned to pursue other opportunities or interests",
}
REGEX = {
    "government": r"\b(white house|administration|secretary of|cabinet|ambassador|nominat|"
                  r"under secretary|federal reserve|treasury|department of|commissioner|senate)\b",
    "scandal": r"\b(investigation|misconduct|irregularit|restat|sec inquiry|subpoena|fraud|probe)\b",
    "retirement": r"\bretir",
    "health_personal": r"\b(health|medical|family reasons|personal reasons)\b",
}
ABRUPT_LABELS = {"government", "poached", "forced", "scandal", "other_opportunities"}


def classify_text(emb: Embedder, texts: pd.Series) -> pd.DataFrame:
    texts = texts.fillna("").astype(str)
    labs = list(LABELS)
    if emb.kind == "tfidf":
        emb.fit(list(texts) + list(LABELS.values()))
    E = emb.encode(texts.tolist())
    P = emb.encode(list(LABELS.values()))
    S = E @ P.T
    lab = np.array(labs)[S.argmax(1)]
    conf = S.max(1)
    low = texts.str.lower()
    # regex overrides, lowest priority first so government/scandal win
    for k in ["retirement", "health_personal", "scandal", "government"]:
        hit = low.str.contains(REGEX[k], regex=True).values
        lab = np.where(hit, k, lab)
    lab = np.where(texts.str.len() == 0, "unknown", lab)
    return pd.DataFrame({"label": lab, "label_sim": conf}, index=texts.index)


# =============================================================================
# 4. Feature engineering
# =============================================================================
def attach_announcement(dep: pd.DataFrame, keydev: pd.DataFrame) -> pd.DataFrame:
    """Departure announcement date = earliest exec-change keydev mentioning the exec within
    [leftofc-365, leftofc+30]; fallback leftofc. Also attaches departure text."""
    dep = dep.copy()
    dep["dep_ann"] = dep.leftofc
    dep["dep_text"] = ""
    dep["ann_found"] = 0
    dep["ann_keydevid"] = np.nan
    dep["dep_type_rule"] = "not_found"
    dep["abrupt_rule"] = np.nan          # 1 abrupt, 0 planned/other, NaN unknown
    dep["eff_days_rule"] = np.nan
    if keydev.empty:
        return dep
    kx = keydev[keydev.eventtype.str.contains("Executive", case=False, na=False)].copy()
    kx["txt"] = (kx.headline.fillna("") + ". " + kx.situation.fillna("")).str.strip()
    by_g = dict(tuple(kx.groupby("gvkey")))
    for i, r in dep.iterrows():
        g = by_g.get(r.gvkey)
        if g is None:
            continue
        p = GL.parse_name(r.exec_name)
        if not p["last_tokens"]:
            continue
        w = g[(g.announcedate >= r.leftofc - pd.Timedelta(days=365)) &
              (g.announcedate <= r.leftofc + pd.Timedelta(days=30))]
        if w.empty:
            continue
        named = w.txt.map(lambda t: GL.mentions_name(t, p["last_tokens"])).astype(bool)
        w = w[named].sort_values("announcedate")
        if w.empty:
            continue
        chosen = None
        # the EARLIEST item in which this executive is the one leaving (not an appointee)
        for _, k in w.iterrows():
            c = GL.classify_item(k.txt, p["last_tokens"], k.announcedate)
            if c["departing"]:
                chosen = (k, c)
                break
        if chosen is not None:
            k, c = chosen
            dep.at[i, "dep_ann"] = k.announcedate
            dep.at[i, "dep_text"] = GL.name_excerpt(k.txt, p["last_tokens"], 3000)
            dep.at[i, "ann_found"] = 1
            dep.at[i, "ann_keydevid"] = k.keydevid
            dep.at[i, "dep_type_rule"] = c["dep_type"]
            dep.at[i, "abrupt_rule"] = np.nan if c["abrupt"] is None else float(bool(c["abrupt"]))
            dep.at[i, "eff_days_rule"] = np.nan if c["eff_days"] is None else c["eff_days"]
        else:
            dep.at[i, "dep_text"] = " ".join(GL.name_excerpt(t, p["last_tokens"], 1000) for t in w.txt.head(3))[:3000]
    return dep


def price_at(crsp: pd.DataFrame, df: pd.DataFrame, date_col: str) -> pd.Series:
    px = crsp[["permno", "date", "prc"]].dropna().sort_values("date")
    left = df[["permno", date_col]].reset_index().sort_values(date_col)
    # merge_asof needs identical datetime resolution (s vs us differs by data source / pandas version)
    px["date"] = px["date"].astype("datetime64[ns]")
    left[date_col] = pd.to_datetime(left[date_col]).astype("datetime64[ns]")
    px["permno"] = px.permno.astype(int)
    left["permno"] = left.permno.astype(int)
    m = pd.merge_asof(left, px, left_on=date_col, right_on="date", by="permno", direction="backward")
    return m.set_index("index").prc.reindex(df.index)


def guidance_features(ev: pd.DataFrame, guid: pd.DataFrame, crsp) -> pd.DataFrame:
    cols = ["n_guid_12m", "days_since_guid", "opt_max", "opt_last", "n_unfulfilled",
            "miss_unfulfilled", "first_guid_date", "guid_chg_last", "guid_raise_last"]
    if guid.empty:
        return pd.DataFrame(np.nan, index=ev.index, columns=cols)
    g = guid.copy()
    g["px"] = price_at(crsp, g, "anndats")
    g["opt"] = (g["mid"] - g["consensus"]) / g.px        # optimism vs street, scaled by price
    g["miss"] = (g["actual"] - g["mid"]) / g.px           # realized minus guided
    bad_o, bad_m = g.opt.abs() > MAX_GAP, g.miss.abs() > MAX_GAP
    if bad_o.any() or bad_m.any():
        print(f"  [warn] dropped implausible guidance gaps (> {MAX_GAP:.0%} of price): "
              f"{int(bad_o.sum())} optimism, {int(bad_m.sum())} miss values of {len(g):,} rows")
    g.loc[bad_o, "opt"] = np.nan
    g.loc[bad_m, "miss"] = np.nan
    by_p = dict(tuple(g.groupby("permno")))
    rows = {}
    for i, r in ev.iterrows():
        gg = by_p.get(r.permno)
        if gg is None:
            rows[i] = dict.fromkeys(cols, np.nan) | {"n_guid_12m": 0}
            continue
        w = gg[(gg.anndats < r.dep_ann) & (gg.anndats >= r.dep_ann - pd.Timedelta(days=365))]
        if w.empty:
            rows[i] = dict.fromkeys(cols, np.nan) | {"n_guid_12m": 0}
            continue
        w = w.sort_values("anndats")
        unf = w[w.period_end > r.dep_ann]                  # guidance the exec won't be around for
        # was the latest guidance a RAISE on the firm's own previous guidance for the same period? (change scaled by price)
        lr = w.iloc[-1]
        prev = gg[(gg.period_end == lr.period_end) & (gg.pdicity == lr.pdicity) & (gg.anndats < lr.anndats)].sort_values("anndats")
        chg = ((lr.mid - prev.mid.iloc[-1]) / lr.px) if len(prev) and pd.notna(lr.px) and lr.px > 0 else np.nan
        rows[i] = {"guid_chg_last": chg, "guid_raise_last": (np.nan if pd.isna(chg) else float(chg > 0)),
                   "n_guid_12m": len(w),
                   "days_since_guid": (r.dep_ann - w.anndats.iloc[-1]).days,
                   "opt_max": w.opt.max(), "opt_last": w.opt.iloc[-1],
                   "n_unfulfilled": len(unf),
                   "miss_unfulfilled": unf.miss.mean() if len(unf) else np.nan,
                   "first_guid_date": w.anndats.iloc[0]}
    return pd.DataFrame.from_dict(rows, orient="index")[cols]


def insider_features(ev: pd.DataFrame, ins: pd.DataFrame, crsp) -> pd.DataFrame:
    """Insider selling by the departing executive. Owners are matched to the executive with
    gg_labels.match_parsed(): score 3 = full first name (or nickname) agrees, 2 = initial only.
    Primary features use score 3 only; *_len ('lenient') also accept score 2.
    sell_val_post_lastguid = sales between the MOST RECENT guidance before the announcement and the
    announcement (the quantity the flag needs); sell_val_post_guid keeps the older 'since first guidance in
    the prior year' definition for comparison."""
    cols = ["sell_val_12m", "sell_val_post_guid", "opp_sell_val", "n_sells_12m", "mcap",
            "sell_val_post_lastguid", "sell_val_post_lastguid_len", "n_owner_matches"]
    out = pd.DataFrame(np.nan, index=ev.index, columns=cols)
    ev_px = price_at(crsp, ev, "dep_ann")
    sh = crsp[["permno", "date", "shrout"]].sort_values("date")
    sh["date"] = sh["date"].astype("datetime64[ns]")
    sh["permno"] = sh.permno.astype(int)
    evl = ev[["permno", "dep_ann"]].reset_index().sort_values("dep_ann")
    evl["dep_ann"] = pd.to_datetime(evl.dep_ann).astype("datetime64[ns]")
    evl["permno"] = evl.permno.astype(int)
    m = pd.merge_asof(evl, sh, left_on="dep_ann", right_on="date", by="permno").set_index("index")
    out["mcap"] = ev_px * m.shrout.reindex(ev.index) * 1000
    new_cols = [c for c in cols if c != "mcap"]
    if ins.empty:
        out[new_cols] = 0
        return out
    s = ins[ins.acqdisp == "D"].copy()
    s["tprice"] = s.tprice.where(s.tprice > 0)
    s["tprice"] = s.tprice.fillna(price_at(crsp, s, "trandate"))
    s["val"] = s.shares.abs() * s.tprice
    s["ym"] = s.trandate.dt.month
    s["yr"] = s.trandate.dt.year
    by_p = dict(tuple(s.groupby("permno")))
    for i, r in ev.iterrows():
        ss = by_p.get(r.permno)
        if ss is None:
            out.loc[i, new_cols] = 0
            continue
        p = GL.parse_name(r.exec_name)
        scores = {o: GL.match_parsed(p, o) for o in ss.owner.dropna().unique()}
        sc = ss.owner.map(scores).fillna(0)
        mine = ss[sc >= 3]                      # strict: full first name / nickname agrees
        mine_len = ss[sc >= 2]                  # lenient: initial-only matches allowed
        w = mine[(mine.trandate <= r.dep_ann) & (mine.trandate >= r.dep_ann - pd.Timedelta(days=365))]
        fg = r.get("first_guid_date")
        wg = w[w.trandate >= fg] if pd.notna(fg) else w.iloc[0:0]
        dsg = r.get("days_since_guid")
        if pd.notna(dsg):
            lg = r.dep_ann - pd.Timedelta(days=int(dsg))
            post = mine[(mine.trandate >= lg) & (mine.trandate <= r.dep_ann)].val.sum()
            post_len = mine_len[(mine_len.trandate >= lg) & (mine_len.trandate <= r.dep_ann)].val.sum()
        else:
            post = post_len = 0.0
        # Cohen-Malloy-Pomorski: routine if the insider traded in the same calendar month
        # in each of the three prior years
        hist = set(zip(mine.yr, mine.ym))
        routine = w.apply(lambda x: all((x.yr - k, x.ym) in hist for k in (1, 2, 3)), axis=1) \
            if len(w) else pd.Series(dtype=bool)
        out.loc[i, ["sell_val_12m", "sell_val_post_guid", "opp_sell_val", "n_sells_12m",
                    "sell_val_post_lastguid", "sell_val_post_lastguid_len", "n_owner_matches"]] = [
            w.val.sum(), wg.val.sum(), w.loc[~routine.astype(bool), "val"].sum() if len(w) else 0,
            len(w), post, post_len, sum(1 for v in scores.values() if v >= 3)]
    return out


def event_study(ev: pd.DataFrame, crsp: pd.DataFrame, mkt: pd.DataFrame, path_len=252):
    r = crsp[["permno", "date", "ret"]].merge(mkt, on="date").sort_values(["permno", "date"])
    r["ret"] = r.ret.fillna(0)
    res, paths = {}, {}
    ev_by_p = dict(tuple(ev.groupby("permno")))
    for permno, g in r.groupby("permno"):
        if permno not in ev_by_p:
            continue
        d, x, m = g.date.values, g.ret.values, g.mkt.values
        for i, e in ev_by_p[permno].iterrows():
            i0 = d.searchsorted(np.datetime64(e.dep_ann))
            if i0 < 260 or i0 >= len(d) - 2:
                continue
            est = slice(i0 - 250, i0 - 30)
            b, a = np.polyfit(m[est], x[est], 1)
            ar = x - (a + b * m)
            out = {"beta": b, "car_m1p1": ar[i0 - 1:i0 + 2].sum(),
                   "pre_ret": np.prod(1 + x[i0 - 252:i0 - 1]) - np.prod(1 + m[i0 - 252:i0 - 1])}
            for h in (126, 252):
                seg = slice(i0, min(i0 + h, len(d)))
                out[f"bhar_{h}"] = (np.prod(1 + x[seg]) - np.prod(1 + m[seg])
                                    if (seg.stop - seg.start) >= 0.8 * h else np.nan)
            seg = slice(i0 - 20, min(i0 + path_len, len(d)))
            p = np.cumsum(x[seg] - m[seg])
            paths[i] = np.pad(p, (0, path_len + 20 - len(p)), constant_values=np.nan)
            res[i] = out
    return pd.DataFrame.from_dict(res, orient="index"), paths


def winsor(s, p=0.01):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def z(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


# =============================================================================
# 5. Build
# =============================================================================
@dataclass
class Cfg:
    start: int
    end: int
    roles: set
    demo: bool
    refresh: bool
    boardex: bool
    max_events: int | None
    embed_model: str
    guid_window: int = 180          # "recent guidance" = within N days before departure
    big_loss: float = -0.20         # BHAR_252 threshold for a bad outcome


def build(cfg: Cfg):
    make_output_dirs()
    print("[1/7] Loading data", "(synthetic demo)" if cfg.demo else "from WRDS")
    if cfg.demo:
        dep, crsp, mkt, guid, ins, keydev, names = synthetic()
    else:
        db = WRDS(refresh=cfg.refresh)
        dep = load_departures(db, cfg.start, cfg.end, cfg.roles)
        dep = link_gvkey_permno(db, dep)
        dep["permno"] = dep.permno.astype(int)
        if cfg.max_events:
            dep = dep.sample(min(cfg.max_events, len(dep)), random_state=0)
        print(f"  departures linked to CRSP: {len(dep):,}")
        crsp, mkt, names = load_crsp(db, dep.permno.unique(), cfg.start, cfg.end)
        keydev = load_keydev(db, dep.gvkey.unique(), cfg.start, cfg.end)
        guid = load_guidance(db, dep.permno.unique(), cfg.start, cfg.end)
        ins = load_insiders(db, dep.permno.unique(), names, cfg.start, cfg.end)

    print("[2/7] Departure announcement dates + text")
    ev = attach_announcement(dep.reset_index(drop=True), keydev)
    ev["event_id"] = ev.gvkey + "_" + ev.execid + "_" + ev.leftofc.dt.strftime("%Y%m%d")

    print("[3/7] Guidance + insider features")
    ev = ev.join(guidance_features(ev, guid, crsp))
    ev = ev.join(insider_features(ev, ins, crsp))

    print("[4/7] Event study")
    es, paths = event_study(ev, crsp, mkt)
    ev = ev.join(es)
    make_output_dirs()
    with open(DATA / "car_paths.pkl", "wb") as fh:   # used by the analysis notebook
        pickle.dump({ev.event_id[i]: p for i, p in paths.items()}, fh)

    print("[5/7] Departure type (validated sentence-level rules; see validation/)")
    emb = Embedder(cfg.embed_model)             # embeddings are now used only for the RAG search index
    print(f"  embedder (RAG index only): {emb.name}")
    ev["label"] = ev.dep_type_rule
    ev["label_sim"] = np.nan
    if cfg.demo:
        cls = classify_text(emb, ev.dep_text)   # the synthetic demo has no real announcement structure
        ev["label"] = cls["label"].values
        ev["label_sim"] = cls["label_sim"].values
        ev["abrupt_rule"] = ev.label.isin(ABRUPT_LABELS).astype(float)
    print(f"  announcement found for {int(ev.ann_found.sum()):,} of {len(ev):,} departures; "
          f"types: {ev.label.value_counts().head(6).to_dict()}")
    if cfg.boardex and not cfg.demo:
        top = ev.sort_values("opt_max", ascending=False).head(400)
        dest = load_boardex_destinations(db, top)
        if len(dest):
            dest["dest_gov"] = dest.dest_org.str.lower().str.contains(REGEX["government"], regex=True)
            g = dest.groupby("event_id").agg(dest_org=("dest_org", "first"), dest_gov=("dest_gov", "max"))
            ev = ev.merge(g, on="event_id", how="left")
    dest_gov = ev["dest_gov"].fillna(False).astype(bool) if "dest_gov" in ev else False
    ev["gov_dest"] = (ev.label.eq("government") | dest_gov).astype(int)
    # abrupt = the rule classifier found a departure announcement AND judged it sudden. Missing or unreadable
    # announcements are NOT abrupt (the old rule counted missing text as abrupt, inflating the flag ~3x).
    ev["abrupt"] = (ev.abrupt_rule == 1).astype(int)

    # ----------------------------------------------------------- derived vars
    ev["guided_recent"] = (ev.days_since_guid <= cfg.guid_window).fillna(False).astype(int)
    ev["sell_pct_mcap"] = (ev.sell_val_post_guid / ev.mcap).replace([np.inf, -np.inf], np.nan).fillna(0)
    ev["opp_sell"] = (ev.opp_sell_val > 0).astype(int)
    # --- the Guidance Reversal flag, built ONLY from information known on the announcement date -----------
    #   1. EPS guidance issued within cfg.guid_window days before the announcement       (guided_recent)
    #   2. the executive sold shares between that guidance and the announcement          (sold_recent)
    #   3. the departure was abrupt                                                       (abrupt)
    # (1) and (2) do not depend on text, so only 'gr_candidate' events need the abrupt judgement read by hand.
    ev["sold_recent"] = ((ev.sell_val_post_lastguid > 0) & (ev.guided_recent == 1)).astype(int)
    ev["sold_recent_len"] = ((ev.sell_val_post_lastguid_len > 0) & (ev.guided_recent == 1)).astype(int)
    ev["gr_candidate"] = ev.sold_recent
    ev["gr_rule"] = ((ev.gr_candidate == 1) & (ev.abrupt == 1)).astype(int)
    ev["log_mcap"] = np.log(ev.mcap.where(ev.mcap > 0))
    ev["year"] = ev.dep_ann.dt.year
    if "siccd" in names:
        sic = names.dropna(subset=["siccd"]).drop_duplicates("permno", keep="last")[["permno", "siccd"]]
        ev = ev.merge(sic, on="permno", how="left")
        ev["sector"] = (pd.to_numeric(ev.siccd, errors="coerce") // 100).map(sic2_sector).fillna("Other")
    for c in ["opt_max", "miss_unfulfilled", "sell_pct_mcap", "car_m1p1", "bhar_126", "bhar_252",
              "pre_ret"]:
        ev[c + "_w"] = winsor(ev[c]) if ev[c].notna().sum() > 20 else ev[c]

    # ex-ante score: info known on announcement day only
    ev["score_exante"] = (z(ev.guided_recent) + z(ev.opt_max_w.fillna(0)) +
                          z(np.log1p(ev.sell_pct_mcap_w * 1e4)) + z(ev.opp_sell) +
                          z(ev.abrupt) + 0.5 * z(ev.gov_dest)) / 5.5
    # ex-post damage: guidance missed after exit + abnormal returns
    ev["score_expost"] = (z(-ev.miss_unfulfilled_w.fillna(0)) + z(-ev.car_m1p1_w.fillna(0)) +
                          z(-ev.bhar_252_w.fillna(0))) / 3
    ev["score"] = ev.score_exante + ev.score_expost
    # LEGACY / descriptive only: the old outcome-conditioned flag (the ex-ante flag plus a later bad outcome).
    # Comparing its returns with other firms is circular; use gr_rule / the calendar-time test for inference.
    ev["guide_and_go"] = ((ev.gr_rule == 1) &
                          ((ev.miss_unfulfilled < 0) | (ev.bhar_252 < cfg.big_loss))).astype(int)

    print("[6/7] Statistics + models")
    stats = analyze(ev, paths, cfg)

    print("[7/7] Screen, report, RAG index")
    keep = ["event_id", "conm", "exec_name", "role", "dep_ann", "leftofc", "label", "gov_dest",
            "abrupt", "n_guid_12m", "days_since_guid", "opt_max", "n_unfulfilled", "miss_unfulfilled",
            "sell_val_post_guid", "sell_pct_mcap", "opp_sell", "car_m1p1", "bhar_126", "bhar_252",
            "score_exante", "score_expost", "score", "guide_and_go", "dep_text"]
    keep = [c for c in keep if c in ev]
    screen = ev.sort_values("score", ascending=False)[keep]
    screen.to_csv(DATA / "screen.csv", index=False)
    ev.drop(columns=["dep_text"]).to_csv(DATA / "events.csv", index=False)
    build_rag(ev, emb, keydev)
    write_report(ev, screen, stats, cfg, emb)
    print(f"\nDone. {int(ev.guide_and_go.sum())} Guidance Reversal cases out of {len(ev):,} departures.")
    print(f"Outputs in {OUT}")


def sic2_sector(s2):
    if pd.isna(s2):
        return np.nan
    s2 = int(s2)
    if s2 == 28:
        return "Chemicals/Pharma"
    if 35 <= s2 <= 36 or s2 == 73:
        return "Tech"
    if 60 <= s2 <= 67:
        return "Financials"
    if 10 <= s2 <= 14 or s2 == 29:
        return "Energy/Mining"
    if s2 == 80:
        return "Health Services"
    if 20 <= s2 <= 39:
        return "Manufacturing"
    if 40 <= s2 <= 49:
        return "Transport/Utilities"
    if 50 <= s2 <= 59:
        return "Retail/Wholesale"
    return "Other"


# =============================================================================
# 6. Statistics + ML
# =============================================================================
def analyze(ev: pd.DataFrame, paths: dict, cfg: Cfg) -> dict:
    from scipy import stats as st
    import statsmodels.formula.api as smf
    import patsy
    out = {}

    # --- group comparison
    def grp_tbl(flag):
        rows = []
        for v, g in ev.groupby(flag):
            rows.append({flag: v, "n": len(g), "CAR[-1,+1]": g.car_m1p1.mean(),
                         "BHAR126": g.bhar_126.mean(), "BHAR252": g.bhar_252.mean(),
                         "median BHAR252": g.bhar_252.median()})
        a, b = ev.loc[ev[flag] == 1, "bhar_252"].dropna(), ev.loc[ev[flag] == 0, "bhar_252"].dropna()
        t, p = st.ttest_ind(a, b, equal_var=False) if len(a) > 2 and len(b) > 2 else (np.nan, np.nan)
        return pd.DataFrame(rows), t, p

    for f in ["guide_and_go", "guided_recent", "abrupt", "opp_sell", "gov_dest"]:
        tbl, t, p = grp_tbl(f)
        out[f"grp_{f}"] = (tbl, t, p)

    # bootstrap CI for Guidance Reversal mean BHAR252
    x = ev.loc[ev.guide_and_go == 1, "bhar_252"].dropna().values
    if len(x) > 5:
        rng = np.random.default_rng(0)
        bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        out["boot_ci"] = (x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))

    # --- OLS: does ex-ante information predict post-departure returns / misses?
    d = ev.copy()
    d["lsell"] = np.log1p(d.sell_pct_mcap_w.fillna(0) * 1e4)
    d["opt"] = d.opt_max_w.fillna(0)
    regs = {}
    for y in ["bhar_252_w", "car_m1p1_w", "miss_unfulfilled_w"]:
        dd = d.dropna(subset=[y, "log_mcap", "pre_ret_w"])
        if len(dd) < 50:
            continue
        # Drop constant / collinear regressors so the design matrix is full rank
        terms = ["guided_recent", "opt", "lsell", "opp_sell", "abrupt", "gov_dest",
                 "guided_recent:lsell", "log_mcap", "pre_ret_w"]
        if dd.role.nunique() > 1:
            terms.append("C(role)")
        kept = []
        for t in terms:
            cand = kept + [t]
            X = patsy.dmatrix(" + ".join(cand), dd, return_type="dataframe")
            if np.linalg.matrix_rank(X.values) == X.shape[1]:
                kept = cand
            else:
                print(f"  [{y}] dropping collinear/constant term: {t}")
        f = f"{y} ~ " + " + ".join(kept)
        try:
            m = smf.ols(f, dd).fit(cov_type="cluster", cov_kwds={"groups": dd.year})
            regs[y] = m
        except Exception as e:
            print("  regression failed:", y, e)
    out["regs"] = regs
    out["corr_exante_expost"] = ev[["score_exante", "score_expost"]].corr(method="spearman").iloc[0, 1]

    # --- out-of-time ML: predict a bad outcome from ex-ante features
    feats = ["guided_recent", "n_guid_12m", "days_since_guid", "opt_max_w", "opt_last", "lsell",
             "opp_sell", "abrupt", "gov_dest", "log_mcap", "pre_ret_w", "label_sim"]
    d["y"] = (d.bhar_252 < cfg.big_loss).astype(int)
    dd = d.dropna(subset=["bhar_252"]).copy()
    for lab in sorted(dd.label.dropna().unique()):
        dd[f"lab_{lab}"] = (dd.label == lab).astype(int)
        feats.append(f"lab_{lab}")
    feats = [f for f in feats if f in dd]
    out["ml"] = None
    if len(dd) > 150 and dd.y.nunique() == 2:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        from sklearn.inspection import permutation_importance
        cut = int(dd.year.quantile(0.7))
        tr, te = dd[dd.year <= cut], dd[dd.year > cut]
        feats = [f for f in feats if tr[f].astype(float).nunique(dropna=True) > 1]   # all-NaN / constant columns break binning
        if te.y.nunique() == 2 and tr.y.nunique() == 2 and feats:
            X_tr, X_te = tr[feats].astype(float), te[feats].astype(float)
            gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                l2_regularization=1.0, random_state=0).fit(X_tr, tr.y)
            lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=2000, C=0.5)).fit(X_tr, tr.y)
            auc_gb = roc_auc_score(te.y, gb.predict_proba(X_te)[:, 1])
            auc_lr = roc_auc_score(te.y, lr.predict_proba(X_te)[:, 1])
            auc_score = roc_auc_score(te.y, te.score_exante)
            pi = permutation_importance(gb, X_te, te.y, scoring="roc_auc", n_repeats=10, random_state=0)
            imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
            out["ml"] = {"cut": cut, "n_tr": len(tr), "n_te": len(te), "base_rate": te.y.mean(),
                         "auc_gb": auc_gb, "auc_lr": auc_lr, "auc_score": auc_score, "imp": imp}

    # --- counts
    gg = ev[ev.guide_and_go == 1]
    out["by_year"] = ev.groupby("year").agg(departures=("event_id", "size"),
                                            guide_and_go=("guide_and_go", "sum"))
    if "sector" in ev:
        out["by_sector"] = ev.groupby("sector").agg(departures=("event_id", "size"),
                                                    guide_and_go=("guide_and_go", "sum"),
                                                    mean_bhar252=("bhar_252", "mean")
                                                    ).sort_values("guide_and_go", ascending=False)
    out["by_label"] = gg.label.value_counts()

    # --- figure: CAR paths
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = np.arange(-20, 252)
        for flag, lbl in [(1, "Guidance Reversal"), (0, "Other departures")]:
            ids = [i for i in ev.index[ev.guide_and_go == flag] if i in paths]
            if ids:
                ax.plot(xs, np.nanmean(np.vstack([paths[i] for i in ids]), 0) * 100,
                        label=f"{lbl} (n={len(ids)})")
        ax.axvline(0, color="k", lw=.8, ls="--")
        ax.axhline(0, color="grey", lw=.5)
        ax.set(xlabel="Trading days relative to departure announcement",
               ylabel="Cumulative market-adjusted return (%)",
               title="Post-departure abnormal returns")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "car_paths.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print("  plot failed:", e)
    return out


# =============================================================================
# 7. RAG index + search
# =============================================================================
SEED_CASES = [  # well-known reference narratives (public record) -- used as retrieval anchors
    "Enron 2001: CEO Jeffrey Skilling resigned abruptly in August 2001 citing personal reasons, months "
    "after bullish outlook statements and substantial stock sales; Enron collapsed into bankruptcy in "
    "December 2001 and Skilling was later convicted of fraud and insider trading.",
    "ExxonMobil 2016: CEO Rex Tillerson departed after being nominated U.S. Secretary of State; "
    "a pre-arranged retirement package placed his deferred stock in a trust.",
    "Goldman Sachs 2006: CEO Henry Paulson left to become U.S. Treasury Secretary; government "
    "ethics rules required divesting his Goldman shares (tax-deferred).",
    "Halliburton 2000: CEO Dick Cheney left to run for Vice President, exercising options and selling shares; "
    "the company later faced asbestos liabilities and an SEC accounting inquiry for that era.",
    "Pfizer 2020: CEO Albert Bourla sold about $5.6M of shares on 9 Nov 2020, the day positive COVID "
    "vaccine data were announced, under a 10b5-1 plan adopted in August 2020; he did not depart.",
    "Countrywide 2006-2007: CEO Angelo Mozilo sold large amounts of stock under amended 10b5-1 plans "
    "while publicly upbeat; later settled SEC insider-trading and disclosure charges.",
]


def case_card(r) -> str:
    def f(x, fmt):
        return format(x, fmt) if pd.notna(x) else "n/a"
    return (f"{r.conm} {r.role} {r.exec_name} departure announced {r.dep_ann:%Y-%m-%d} "
            f"(reason class: {r.label}; abrupt={int(r.abrupt)}; government destination={int(r.gov_dest)}). "
            f"Guidance: {int(r.n_guid_12m) if pd.notna(r.n_guid_12m) else 0} EPS guidance releases in prior 12m, "
            f"last {f(r.days_since_guid, '.0f')} days before exit; max optimism vs consensus "
            f"{f(r.opt_max * 100 if pd.notna(r.opt_max) else np.nan, '.2f')}% of price; "
            f"{int(r.n_unfulfilled) if pd.notna(r.n_unfulfilled) else 0} guidance periods ended after exit, "
            f"realized minus guided {f(r.miss_unfulfilled * 100 if pd.notna(r.miss_unfulfilled) else np.nan, '.2f')}% of price. "
            f"Insider sales after guidance ${f(r.sell_val_post_guid / 1e6 if pd.notna(r.sell_val_post_guid) else np.nan, ',.2f')}M "
            f"({f(r.sell_pct_mcap * 1e4, '.1f')} bp of market cap; opportunistic={int(r.opp_sell)}). "
            f"Market reaction CAR[-1,+1] {f(r.car_m1p1 * 100 if pd.notna(r.car_m1p1) else np.nan, '.1f')}%, "
            f"12m BHAR {f(r.bhar_252 * 100 if pd.notna(r.bhar_252) else np.nan, '.1f')}%. "
            f"Guidance Reversal flag={int(r.guide_and_go)}. Text: {str(r.dep_text)[:600]}")


def build_rag(ev: pd.DataFrame, emb: Embedder, keydev: pd.DataFrame):
    docs = [{"type": "case", "id": r.event_id, "text": case_card(r)} for _, r in ev.iterrows()]
    docs += [{"type": "seed", "id": f"seed{i}", "text": t} for i, t in enumerate(SEED_CASES)]
    if not keydev.empty:
        kg = keydev[keydev.eventtype.str.contains("Guidance|Restatement|Lawsuit", case=False, na=False)]
        kg = kg[kg.gvkey.isin(ev.gvkey)].head(50000)
        docs += [{"type": "keydev", "id": str(r.keydevid),
                  "text": f"{r.announcedate:%Y-%m-%d} {r.eventtype}: {r.headline}. {str(r.situation)[:800]}"}
                 for _, r in kg.iterrows()]
    D = pd.DataFrame(docs)
    if emb.kind == "tfidf":
        emb.fit(D.text)
    X = emb.encode(D.text.tolist())
    np.save(RAG / "rag_vectors.npy", X)
    D.to_pickle(RAG / "rag_docs.pkl")
    with open(RAG / "rag_embedder.pkl", "wb") as fh:
        pickle.dump({"kind": emb.kind, "name": emb.name,
                     "tfidf": emb.model if emb.kind == "tfidf" else None,
                     "svd": getattr(emb, "svd", None) if emb.kind == "tfidf" else None}, fh)
    print(f"  RAG index: {len(D):,} documents ({emb.name})")


def search(query: str, k: int, use_llm: bool):
    D = pd.read_pickle(RAG / "rag_docs.pkl")
    X = np.load(RAG / "rag_vectors.npy")
    meta = pickle.load(open(RAG / "rag_embedder.pkl", "rb"))
    if meta["kind"] == "st":
        emb = Embedder(meta["name"])
    else:
        emb = Embedder.__new__(Embedder)
        emb.kind, emb.name, emb.model, emb._fitted = "tfidf", "tfidf", meta["tfidf"], True
        emb.svd = meta.get("svd")
    q = emb.encode([query])[0]
    sims = X @ q
    top = np.argsort(-sims)[:k]
    hits = D.iloc[top].assign(sim=sims[top])
    for _, h in hits.iterrows():
        print(f"\n[{h.sim:.3f}] ({h.type}) {h.id}\n" + textwrap.fill(h.text, 110))
    if use_llm:
        try:
            import anthropic
        except ImportError:
            sys.exit("pip install anthropic")
        ctx = "\n\n".join(f"<doc id='{h.id}' type='{h.type}'>{h.text}</doc>" for _, h in hits.iterrows())
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("GG_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=1200,
            system=("You are a forensic equity research assistant. Answer ONLY from the provided "
                    "documents; cite doc ids; distinguish correlation from evidence of wrongdoing; "
                    "flag where data (e.g., name matching, 10b5-1 status) is uncertain."),
            messages=[{"role": "user", "content": f"Question: {query}\n\nDocuments:\n{ctx}"}])
        print("\n" + "=" * 80 + "\nLLM synthesis\n" + "=" * 80)
        print(msg.content[0].text)


# =============================================================================
# 8. Report
# =============================================================================
def write_report(ev, screen, s, cfg, emb):
    L = [f"# Guidance Reversal screen ({'DEMO / synthetic' if cfg.demo else 'WRDS'} data)\n",
         f"Departures analysed: **{len(ev):,}** ({', '.join(sorted(cfg.roles))}), "
         f"{ev.dep_ann.min():%Y} to {ev.dep_ann.max():%Y}. Embedder: `{emb.name}`.\n",
         f"Guidance Reversal definition: EPS guidance within {cfg.guid_window}d before announcement, "
         f"exec sold shares after that guidance, abrupt / non-retirement exit, and either the "
         f"post-exit guidance was missed or 12m BHAR < {cfg.big_loss:.0%}.\n",
         f"**Cases flagged: {int(ev.guide_and_go.sum())}** "
         f"({ev.guide_and_go.mean():.1%} of departures). "
         f"Departures with recent guidance: {int(ev.guided_recent.sum())}. "
         f"Government destinations (text/BoardEx): {int(ev.gov_dest.sum())}.\n"]
    if "boot_ci" in s:
        m, lo, hi = s["boot_ci"]
        L.append(f"Mean 12m BHAR of flagged cases: **{m:.1%}** (bootstrap 95% CI {lo:.1%} to {hi:.1%}).\n")
    L.append(f"Spearman corr(ex-ante score, ex-post damage): **{s['corr_exante_expost']:.3f}**\n")
    L.append("## Group comparisons (Welch t-test on BHAR252)\n")
    for k, (tbl, t, p) in {k: v for k, v in s.items() if k.startswith("grp_")}.items():
        L.append(f"### {k[4:]}  (t={t:.2f}, p={p:.3g})\n")
        L.append(tbl.to_markdown(index=False, floatfmt=".4f") + "\n")
    L.append("## Regressions (SE clustered by year)\n")
    for y, m in s["regs"].items():
        L.append(f"### {y}  (N={int(m.nobs)}, R2={m.rsquared:.3f})\n```\n"
                 + m.summary2().tables[1].round(4).to_string() + "\n```\n")
    if s.get("ml"):
        ml = s["ml"]
        L.append("## Out-of-time prediction of 12m BHAR < "
                 f"{cfg.big_loss:.0%}\nTrain <= {ml['cut']} (n={ml['n_tr']}), test > {ml['cut']} "
                 f"(n={ml['n_te']}, base rate {ml['base_rate']:.1%}).\n\n"
                 f"| model | AUC |\n|---|---|\n| Gradient boosting | {ml['auc_gb']:.3f} |\n"
                 f"| Logistic | {ml['auc_lr']:.3f} |\n| Ex-ante score (no fitting) | {ml['auc_score']:.3f} |\n\n"
                 "Permutation importance (AUC drop):\n\n" + ml["imp"].round(4).to_frame("imp").to_markdown() + "\n")
    L.append("## Counts by year\n" + s["by_year"].to_markdown() + "\n")
    if "by_sector" in s:
        L.append("## Counts by sector\n" + s["by_sector"].to_markdown(floatfmt=".3f") + "\n")
    L.append("## Departure reason of flagged cases\n" + s["by_label"].to_frame("n").to_markdown() + "\n")
    cols = ["conm", "exec_name", "role", "dep_ann", "label", "opt_max", "miss_unfulfilled",
            "sell_val_post_guid", "car_m1p1", "bhar_252", "score"]
    L.append("## Top 30 screen\n" + screen.head(30)[[c for c in cols if c in screen]]
             .to_markdown(index=False, floatfmt=".4f") + "\n")
    L.append("![CAR paths](car_paths.png)\n")
    L.append("## Caveats\n- Execucomp covers S&P 1500; small caps are missing.\n"
             "- Insider-to-exec matching is name-based; 10b5-1 plan status is not identified in "
             "TR table1 (it is flagged on Form 4s from 2023, after the SEC rule change).\n"
             "- BHARs ignore delisting returns; add crsp.dsedelist for survivorship-clean results.\n"
             "- Flags show patterns, not proof of misconduct.\n")
    (RESULTS / "report.md").write_text("\n".join(L), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="pull data, compute features, run stats, build screen + RAG")
    b.add_argument("--start", type=int, default=2000)
    b.add_argument("--end", type=int, default=2025)
    b.add_argument("--roles", default="CEO,CFO")
    b.add_argument("--demo", action="store_true", help="synthetic data, no WRDS needed")
    b.add_argument("--refresh", action="store_true", help="ignore cache and re-pull")
    b.add_argument("--boardex", action="store_true", help="look up destinations in BoardEx")
    b.add_argument("--max-events", type=int, default=None)
    b.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    b.add_argument("--guid-window", type=int, default=180)
    sub.add_parser("test-api", help="check WRDS_API_KEY with one tiny request")
    s = sub.add_parser("search", help="semantic search over cases (RAG)")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--llm", action="store_true", help="synthesize answer with Claude")
    if len(sys.argv) == 1:          # run with no command (IDE "Run" button / double-click): show a menu
        menu = {"1": ["test-api"],
                "2": ["build", "--start", "2010", "--max-events", "300"],
                "3": ["build", "--start", "2000", "--end", "2025"],
                "4": ["build", "--demo"],
                "5": ["search"]}
        print("guide_and_go -- choose what to run:\n"
              "  1) test-api      check your WRDS API key (do this first)\n"
              "  2) quick build   2010+, 300 departures (test run)\n"
              "  3) full build    2000-2025\n"
              "  4) demo build    synthetic data, no WRDS needed\n"
              "  5) search        semantic search over built cases")
        choice = input("Enter 1-5: ").strip()
        if choice not in menu:
            sys.exit("No valid choice. You can also run e.g.:  python src/guide_and_go.py test-api")
        args = menu[choice]
        if choice == "5":
            args = args + [input("Search query: ").strip() or "CEO left after guidance and sold stock"]
        sys.argv += args
    a = ap.parse_args()
    if a.cmd == "build":
        build(Cfg(a.start, a.end, set(a.roles.upper().split(",")), a.demo, a.refresh, a.boardex,
                  a.max_events, a.embed_model, a.guid_window))
    elif a.cmd == "test-api":
        test_api()
    else:
        search(a.query, a.k, a.llm)


if __name__ == "__main__":
    main()
