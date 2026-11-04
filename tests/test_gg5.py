"""
Tests for Stage 5 (segment-reporting transparency): gg_filings, gg_transparency, gg_stage5.
Run:  uv run python tests/test_gg5.py    (from the repo root; no pytest needed)
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # modules live in src/

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


# ---------------------------------------------------------------- task 1: GG_ROOT override
def test_root_override():
    import guide_and_go as G
    repo_root = Path(__file__).resolve().parents[1]
    old = os.environ.pop("GG_ROOT", None)
    try:
        G = importlib.reload(G)
        check("root defaults to repo root", G.ROOT == repo_root, str(G.ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["GG_ROOT"] = tmp
            G = importlib.reload(G)
            check("GG_ROOT sets ROOT", G.ROOT == Path(tmp), str(G.ROOT))
            check("GG_ROOT moves CACHE", G.CACHE == Path(tmp) / "gg_cache", str(G.CACHE))
            check("GG_ROOT moves OUT", G.OUT == Path(tmp) / "outputs", str(G.OUT))
    finally:
        os.environ.pop("GG_ROOT", None)
        if old is not None:
            os.environ["GG_ROOT"] = old
        importlib.reload(G)


# ---------------------------------------------------------------- task 2: filing index and timing rule
def _index(rows):
    return pd.DataFrame(rows, columns=["gvkey", "form", "filing_date", "accession"]).assign(
        filing_date=lambda d: pd.to_datetime(d["filing_date"]), cik="1", sic="3600", url="")


def test_filings():
    import gg_filings as F
    idx = _index([("000001", "10-K", "2019-03-01", "a19"), ("000001", "10-K", "2020-03-01", "a20"),
                  ("000001", "10-K", "2021-03-01", "a21"), ("000002", "10-K", "2020-03-01", "b20")])
    d = pd.Timestamp("2020-03-01")
    check("strict excludes same-day filing", F.last_10k_before(idx, "000001", d)["accession"] == "a19")
    check("non-strict includes same-day filing", F.last_10k_before(idx, "000001", d, strict=False)["accession"] == "a20")
    check("other gvkeys are not mixed in", F.last_10k_before(idx, "000002", pd.Timestamp("2020-06-01"))["accession"] == "b20")

    am = _index([("000001", "10-K", "2020-03-01", "orig"), ("000001", "10-K/A", "2020-06-15", "amend")])
    check("amendment ignored when the original exists",
          F.last_10k_before(am, "000001", pd.Timestamp("2020-12-01"))["accession"] == "orig")
    only_am = _index([("000001", "10-K/A", "2020-06-15", "amend")])
    check("amendment used when no original exists",
          F.last_10k_before(only_am, "000001", pd.Timestamp("2020-12-01"))["accession"] == "amend")
    old = _index([("000001", "10-K405", "2001-03-01", "k405")])
    check("10-K405 counts as a 10-K", F.last_10k_before(old, "000001", pd.Timestamp("2002-01-01"))["accession"] == "k405")
    other = _index([("000001", "8-K", "2020-03-01", "x")])
    check("non-10-K forms are ignored", F.last_10k_before(other, "000001", pd.Timestamp("2021-01-01")) is None)
    check("None when no filing before the date", F.last_10k_before(idx, "000001", pd.Timestamp("2018-01-01")) is None)
    check("None for an unknown gvkey", F.last_10k_before(idx, "999999", pd.Timestamp("2022-01-01")) is None)


def test_links():
    import gg_filings as F
    idx = _index([("000001", "10-K", "2005-03-01", "pre"), ("000001", "10-K", "2012-03-01", "in"),
                  ("000001", "10-K", "2022-03-01", "post")]).assign(
        link_start=pd.Timestamp("2010-01-01"), link_end=pd.Timestamp("2020-12-31"))
    kept = F.apply_link_window(idx)
    check("filings outside the link window are not attributed", list(kept["accession"]) == ["in"], str(list(kept["accession"])))
    open_end = idx.assign(link_end=pd.NaT)
    check("open-ended link keeps later filings", set(F.apply_link_window(open_end)["accession"]) == {"in", "post"})

    two = _index([("000001", "10-K", "2010-03-01", "a"), ("000001", "10-K", "2015-03-01", "b"),
                  ("000002", "10-K", "2012-03-01", "c")]).assign(cik=["111", "222", "333"])
    shared = _index([("000003", "10-K", "2010-03-01", "d"), ("000004", "10-K", "2011-03-01", "e")]).assign(cik="444")
    rep = F.validate_links(pd.concat([two, shared], ignore_index=True), db=None)
    check("gvkey with two CIKs is flagged", "000001" in rep["gvkey_multi_cik"], str(rep["gvkey_multi_cik"]))
    check("CIK shared by two gvkeys is flagged", "444" in rep["cik_multi_gvkey"], str(rep["cik_multi_gvkey"]))
    check("clean gvkey is not flagged", "000002" not in rep["gvkey_multi_cik"])


class _FakeDB:
    """Just enough of the guide_and_go WRDS interface for build_filing_index."""
    def __init__(self, tables):
        self.t = tables

    def columns(self, schema, table):
        df = self.t.get((schema, table))
        return list(df.columns) if df is not None else []

    def fetch(self, schema, table, cols, filters=()):
        return self.t[(schema, table)][[c for c in cols if c in self.t[(schema, table)].columns]].copy()

    def fetch_in(self, schema, table, cols, col, values, filters=()):
        df = self.t[(schema, table)]
        df = df[df[col].isin(set(values))]
        for c, op, v in filters:
            if op == "in":
                df = df[df[c].isin(set(v))]
        return df[[c for c in cols if c in df.columns]].copy()


def test_build_index():
    import gg_filings as F
    forms = pd.DataFrame({"cik": ["0000000111", "0000000111", "0000000111", "0000000222"],
                          "form": ["10-K", "10-K", "8-K", "10-K405"],
                          "fdate": pd.to_datetime(["2005-03-01", "2012-03-01", "2012-04-01", "2001-03-01"]),
                          "accession": ["a1", "a2", "a3", "b1"],
                          "fname": ["edgar/data/111/a1.txt", "edgar/data/111/a2.txt", "x", "edgar/data/222/b1.txt"]})
    # real link tables repeat a gvkey-cik pair: an undated row beside dated ones must not reopen the window
    link = pd.DataFrame({"gvkey": ["000001", "000001", "000001", "000002"],
                         "cik": ["0000000111", "0000000111", "0000000111", "0000000222"],
                         "link_start_date": pd.to_datetime(["2010-01-01", None, "2008-01-01", "1990-01-01"]),
                         "link_end_date": pd.to_datetime(["2020-12-31", None, "2015-12-31", None])})
    comp = pd.DataFrame({"gvkey": ["000001", "000002"], "sic": ["3600", "2800"], "cik": ["111", "222"]})
    db = _FakeDB({("wrdssec", "wrds_forms"): forms, ("wrdssec", "wciklink_gvkey"): link, ("comp", "company"): comp})
    idx = F.build_filing_index(db)
    check("index has the documented columns", list(idx.columns) == ["gvkey", "cik", "sic", "form", "filing_date", "accession",
                                                                    "url", "link_start", "link_end"], str(list(idx.columns)))
    check("gvkey is zero-padded to 6", set(idx["gvkey"]) == {"000001", "000002"}, str(set(idx["gvkey"])))
    check("filing before the link start is dropped", "a1" not in set(idx["accession"]))
    check("only 10-K forms are kept", "a3" not in set(idx["accession"]))
    check("undated duplicate link row does not reopen the window", "a1" not in set(idx["accession"]))
    check("one row per filing despite duplicate link rows", idx["accession"].is_unique)
    check("cik normalised to 10 digits", set(idx["cik"]) == {"0000000111", "0000000222"}, str(set(idx["cik"])))
    check("10-K405 kept and joined to SIC", idx.loc[idx["accession"] == "b1", "sic"].iloc[0] == "2800")
    check("url built from the SEC path", idx.loc[idx["accession"] == "a2", "url"].iloc[0].endswith("edgar/data/111/a2.txt"))
    sub = F.build_filing_index(db, gvkeys=["000002"])
    check("gvkeys restriction applies", set(sub["gvkey"]) == {"000002"})
    try:
        F.build_filing_index(_FakeDB({}))
        check("missing source raises", False)
    except RuntimeError:
        check("missing source raises", True)


# ---------------------------------------------------------------- task 3: extraction, statuses, identity, cache
_FILLER = "The company designs and sells products to customers worldwide and competes on quality and price. " * 40


def _html(item1=True, seg_note="", mdna="", filler=_FILLER):
    parts = ["<html><body>", "<p>TABLE OF CONTENTS Item 1. Business 3 Item 1A. Risk Factors 9 Item 7. MD&A 20</p>"]
    if item1:
        parts += ["<h2>Item 1. Business</h2>", f"<p>Acme Corp ('ACME') overview. {filler}</p>"]
    parts += ["<h2>Item 1A. Risk Factors</h2><p>" + filler + "</p>"]
    if mdna:
        parts += ["<h2>Item 7. Management's Discussion and Analysis</h2>", f"<p>{mdna}</p>"]
    if seg_note:
        parts += ["<h2>Note 12. Segment Information</h2>", f"<p>{seg_note}</p>"]
    parts += ["</body></html>"]
    return "".join(parts)


_SEG = ("We have three reportable segments: Devices, Consumables and Services. Devices revenue was $400 million "
        "on 20,000 units. Consumables revenue was $900 million. Segment operating income was $310 million. ") * 3


def test_extraction():
    import gg_transparency as T
    text, st = T.extract_sections(_html(seg_note=_SEG, mdna="Segment results: Consumables grew 12% on volume."))
    check("segment note found -> ok", st == "ok", st)
    check("segment note text is in the output", "Consumables revenue was $900 million" in text)
    check("Item 1 text is in the output", "overview" in text)
    big = _html(seg_note=_SEG * 400, filler="Lorem ipsum dolor sit amet. " * 5000)
    text, st = T.extract_sections(big)
    check("cap respected", len(text) <= 40_000, str(len(text)))
    text, st = T.extract_sections(_html(seg_note="We operate in one reportable segment. All revenue is from product sales."))
    check("single reportable segment -> limited_disclosure, scored", st == "limited_disclosure" and len(text) > 0, st)
    text, st = T.extract_sections(_html())
    check("Item 1 but no segment note -> limited_disclosure", st == "limited_disclosure" and "overview" in text, st)
    text, st = T.extract_sections("<html><body><p>hello</p></body></html>")
    check("unparseable body -> extraction_failed with empty text", st == "extraction_failed" and text == "", st)
    text, st = T.extract_sections("")
    check("empty input -> extraction_failed", st == "extraction_failed" and text == "")
    text, st = T.extract_sections("<html><body><p>" + "The quick brown fox jumps. " * 600 + "</p></body></html>")
    check("long body with no structure at all -> extraction_failed", st == "extraction_failed", st)


def test_strip_identity():
    import gg_transparency as T
    out = T.strip_identity("Acme Corp (ticker ACME) sells units. acme corp also licenses; Acmeology is unrelated.",
                           ["Acme Corp", "ACME"])
    check("name and ticker removed", "acme corp" not in out.lower() and "(ticker [COMPANY])" in out, out)
    check("placeholder present", "[COMPANY]" in out)
    check("unrelated words untouched", "Acmeology" in out and "sells units" in out, out)
    check("empty and short names are ignored", T.strip_identity("A b c", ["", " ", "a"]) == "A b c")


def test_load_sections_cache():
    import gg_filings as F
    import gg_transparency as T
    row = pd.Series({"accession": "0000000111-20-000001", "url": "https://x/y.htm", "cik": "0000000111"})
    calls = []

    def fake_fetch(url, user_agent):
        calls.append(url)
        return _html(seg_note=_SEG)

    with tempfile.TemporaryDirectory() as tmp:
        text, st = F.load_sections(row, ["Acme Corp", "ACME"], Path(tmp), "test agent a@b.c", min_interval=0, fetch=fake_fetch)
        check("load_sections ok and identity stripped", st == "ok" and "Acme Corp" not in text and "[COMPANY]" in text, st)
        text2, st2 = F.load_sections(row, ["Acme Corp", "ACME"], Path(tmp), "test agent a@b.c", min_interval=0, fetch=fake_fetch)
        check("second call served from cache, no refetch", len(calls) == 1 and text2 == text and st2 == "ok", str(calls))
        import gzip, json
        rec = json.loads(gzip.open(next(Path(tmp).glob("*.json.gz")), "rt", encoding="utf-8").read())
        import hashlib
        check("cache records version and text hash", rec["extraction_version"] == T.EXTRACTION_VERSION
              and rec["sha256"] == hashlib.sha256(text.encode()).hexdigest())
        check("raw HTML is never cached", "<html" not in json.dumps(rec).lower())

    def boom(url, user_agent):
        raise OSError("network down")

    with tempfile.TemporaryDirectory() as tmp:
        text, st = F.load_sections(row, [], Path(tmp), "ua", min_interval=0, fetch=boom)
        check("fetch error -> fetch_failed, nothing cached", st == "fetch_failed" and text == "" and not list(Path(tmp).glob("*")))

    def junk(url, user_agent):
        return "<html><body>x</body></html>"

    with tempfile.TemporaryDirectory() as tmp:
        text, st = F.load_sections(row, [], Path(tmp), "ua", min_interval=0, fetch=junk)
        check("junk page -> extraction_failed", st == "extraction_failed" and text == "")


# ---------------------------------------------------------------- task 4: rubric, parsing, config hash, reads
def test_parse_score():
    import gg_transparency as T
    for raw, exp in [("3", 3), (" 3\n", 3), (3, 3), ("0", 0), ("5", 5), ("3.5", None), ("N/A", None), ("7", None),
                     ("three", None), ("", None), (None, None), (True, None), (-1, None), ("3 4", None)]:
        check(f"parse_score({raw!r}) -> {exp}", T.parse_score(raw) == exp, str(T.parse_score(raw)))


_GOOD = ('{"engines": [{"line": "Consumables", "profit_weight": "high", "price": true, "quantity": true}, '
         '{"line": "Devices", "profit_weight": "low", "price": true, "quantity": true}], "T": 5}')


def test_parse_reply():
    import gg_transparency as T
    r = T.parse_reply(_GOOD)
    check("valid reply parses", r is not None and r["T"] == 5 and len(r["engines"]) == 2)
    check("code fences and prose around JSON tolerated", T.parse_reply("Here:\n```json\n" + _GOOD + "\n```") is not None)
    check("prose only is rejected", T.parse_reply("I think it is a 3") is None)
    check("missing engines key is rejected", T.parse_reply('{"T": 3}') is None)
    check("unknown profit_weight is rejected",
          T.parse_reply('{"engines": [{"line": "x", "profit_weight": "huge", "price": true, "quantity": true}], "T": 3}') is None)
    check("non-integer T is rejected", T.parse_reply(_GOOD.replace('"T": 5', '"T": 3.5')) is None)
    check("T out of range is rejected", T.parse_reply(_GOOD.replace('"T": 5', '"T": 7')) is None)
    check("empty engines allowed only with T <= 2 (engine not identifiable)",
          T.parse_reply('{"engines": [], "T": 1}') is not None and T.parse_reply('{"engines": [], "T": 3}') is None)


def test_rule_check():
    import gg_transparency as T
    bad = {"engines": [{"line": "Injectables", "profit_weight": "high", "price": False, "quantity": False},
                       {"line": "Equipment", "profit_weight": "low", "price": True, "quantity": True}], "T": 4}
    check("razor-blade contradiction flagged (T>=4 but high-profit engine lacks P and Q)", len(T.rule_check(bad)) == 1)
    ok = {"engines": [{"line": "Injectables", "profit_weight": "high", "price": True, "quantity": True}], "T": 4}
    check("consistent reply not flagged", T.rule_check(ok) == [])
    nohigh = {"engines": [{"line": "x", "profit_weight": "low", "price": True, "quantity": True}], "T": 4}
    check("no high-profit engine named but T>2 flagged", len(T.rule_check(nohigh)) == 1)
    low = {"engines": [{"line": "Injectables", "profit_weight": "high", "price": False, "quantity": False}], "T": 2}
    check("low T with an undisclosed engine is consistent", T.rule_check(low) == [])


def test_rubric():
    import gg_transparency as T
    r = T.RUBRIC.lower()
    check("rubric is count-neutral", "number of segments" in r and "neutral" in r)
    check("rubric is profit-weighted", "profit engine" in r and "at most 2" in r)
    check("rubric asks for the structured reply", '"engines"' in T.RUBRIC and '"T"' in T.RUBRIC)
    check("rubric version set", bool(T.RUBRIC_VERSION))


def test_config_hash():
    import gg_transparency as T
    a = {"model": "m1", "prompts": ["p1", "p2"], "extraction_version": "ex1"}
    b = {"extraction_version": "ex1", "prompts": ["p1", "p2"], "model": "m1"}
    check("config hash ignores key order", T.config_hash(a) == T.config_hash(b))
    check("config hash changes with the model", T.config_hash(a) != T.config_hash({**a, "model": "m2"}))
    check("config hash changes with one prompt character", T.config_hash(a) != T.config_hash({**a, "prompts": ["p1", "p3"]}))


def _reads(pairs):
    return pd.DataFrame({"unit_id": [f"u{i}" for i in range(len(pairs))], "score": np.array(pairs, dtype=float)})


def test_combine_reads():
    import gg_transparency as T
    c = T.combine_reads(_reads([2, 4, 1, 3]), _reads([3, 5, 3, 3]))
    check("reads 2 and 3 -> 3 (half up)", c.loc[0, "t_final"] == 3)
    check("reads 4 and 5 -> 5 (half up)", c.loc[1, "t_final"] == 5)
    check("gap of 2 needs a third read", bool(c.loc[2, "needs_third"]) and np.isnan(c.loc[2, "t_final"]))
    check("agreeing reads need no third", not bool(c.loc[3, "needs_third"]) and c.loc[3, "t_final"] == 3)
    c3 = T.combine_reads(_reads([1]), _reads([3]), _reads([2]))
    check("third read -> median 2", c3.loc[0, "t_final"] == 2 and c3.loc[0, "t3"] == 2)
    c4 = T.combine_reads(_reads([3]), _reads([3]), _reads([0]))
    check("third read is ignored when not needed", c4.loc[0, "t_final"] == 3)
    c5 = T.combine_reads(_reads([np.nan]), _reads([3]))
    check("an unparseable read gives no final score", np.isnan(c5.loc[0, "t_final"]) and not bool(c5.loc[0, "needs_third"]))


def test_make_tasks():
    import json
    import gg_transparency as T
    units = pd.DataFrame({"unit_id": [f"u{i}" for i in range(95)], "text": ["t"] * 95,
                          "bhar_252": 0.1, "gvkey": "000001", "dep_ann": "2020-01-01"})
    with tempfile.TemporaryDirectory() as tmp:
        n = T.make_tasks(units, "pilot", Path(tmp), per_file=40)
        files = sorted(Path(tmp).glob("pilot_*.jsonl"))
        rows = [json.loads(l) for f in files for l in f.read_text(encoding="utf-8").splitlines()]
        check("95 units -> 3 files of at most 40", n == 3 and len(files) == 3 and len(rows) == 95)
        check("rows carry only id, text and rubric version", all(set(r) == {"id", "text", "rubric_version"} for r in rows))


# ---------------------------------------------------------------- task 5: reliability statistics, group rule, gate
def test_kappa():
    import gg_transparency as T
    a = [0, 1, 2, 3, 4, 5, 2, 2]
    check("kappa of identical ratings is 1", abs(T.weighted_kappa(a, a) - 1.0) < 1e-12)
    # hand computed (k=3): a=[0,1,2,2], b=[0,2,2,1] -> po=0.875, pe=0.65625, kappa=7/11
    check("kappa known value 7/11", abs(T.weighted_kappa([0, 1, 2, 2], [0, 2, 2, 1], k=3) - 7 / 11) < 1e-9,
          str(T.weighted_kappa([0, 1, 2, 2], [0, 2, 2, 1], k=3)))
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 6, 20000), rng.integers(0, 6, 20000)
    check("kappa of independent draws is near 0", abs(T.weighted_kappa(x, y)) < 0.03, str(T.weighted_kappa(x, y)))
    check("kappa ignores NaN pairs", abs(T.weighted_kappa([1, 2, np.nan, 3], [1, 2, 4, 3]) - 1.0) < 1e-12)
    check("within_one share", T.within_one([0, 1, 2, 5], [1, 3, 2, 3]) == 0.5)


def test_bunching_and_groups():
    import gg_transparency as T
    t = pd.Series([2] * 45 + [3] * 40 + [0] * 5 + [5] * 10)
    check("bunching: levels 2 and 3 hold 85%", abs(T.bunching(t) - 0.85) < 1e-12, str(T.bunching(t)))
    spread = pd.Series(np.repeat(np.arange(6), 10))
    lo, hi = T.cut_points(spread)
    check("cut points on a uniform 0-5 score", (lo, hi) == (1, 4), str((lo, hi)))
    g = T.assign_group(spread, (lo, hi))
    check("low/mid/high by the cut points", set(g[spread <= 1]) == {"low"} and set(g[spread >= 4]) == {"high"}
          and set(g[(spread > 1) & (spread < 4)]) == {"mid"})
    # 65% at T=3: both percentiles land on 3, the tie rule still gives a low and a high group
    heavy = pd.Series([3] * 65 + [0] * 10 + [1] * 5 + [2] * 5 + [4] * 10 + [5] * 5)
    cuts = T.cut_points(heavy)
    check("tied cuts coincide", cuts[0] == cuts[1] == 3, str(cuts))
    gh = T.assign_group(heavy, cuts)
    check("tied cuts: low = T<3, high = T>3, mid = T==3 (no exception)",
          set(gh[heavy < 3]) == {"low"} and set(gh[heavy > 3]) == {"high"} and set(gh[heavy == 3]) == {"mid"})
    check("equal T values land in one group", gh[heavy == 3].nunique() == 1)
    withnan = pd.Series([0, 3, np.nan, 5, 3, 3, 3, 3])
    check("NaN stays NaN", T.assign_group(withnan, (3, 3)).isna().sum() == 1)
    allsame = pd.Series([2] * 30)
    check("all-equal T gives only the mid group", set(T.assign_group(allsame, T.cut_points(allsame))) == {"mid"})


def test_viability():
    import gg_transparency as T
    t = pd.Series([0] * 100 + [3] * 100 + [5] * 100)
    cuts = T.cut_points(t)

    def flags(n_low, n_mid, n_high):
        f = np.zeros(300, dtype=bool)
        f[:n_low] = True
        f[100:100 + n_mid] = True
        f[200:200 + n_high] = True
        return pd.Series(f)

    f = flags(9, 3, 9)
    ok = T.group_viability(t, f, ~f, cuts, min_flagged=8, pool_ratio=5.0)
    check("viable: 9 flagged in low and high, pools of 91", ok["ok"] and not ok["reasons"], str(ok))
    f = flags(7, 3, 9)
    small = T.group_viability(t, f, ~f, cuts, min_flagged=8, pool_ratio=5.0)
    check("not viable: only 7 flagged in low", not small["ok"] and any("low" in r and "flagged" in r for r in small["reasons"]), str(small))
    f = flags(20, 3, 9)
    thin = T.group_viability(t, f, ~f, cuts, min_flagged=8, pool_ratio=5.0)
    check("not viable: low pool of 80 is under 5x 20 flagged", not thin["ok"] and any("low" in r and "pool" in r for r in thin["reasons"]), str(thin))
    same = pd.Series([2] * 300)
    f = flags(9, 3, 9)
    deg = T.group_viability(same, f, ~f, T.cut_points(same), min_flagged=8, pool_ratio=5.0)
    check("not viable: everyone has the same T (no low or high group)", not deg["ok"], str(deg))


def _gate_inputs(kappa_ok=True, within_ok=True, audit_ok=True, bunch_ok=True):
    rng = np.random.default_rng(1)
    n = 400
    base = rng.integers(0, 6, n)
    t1 = base.copy()
    t2 = base.copy()
    if not kappa_ok:
        t2 = rng.integers(0, 6, n)
    elif not within_ok:
        t2 = np.clip(base + rng.choice([-2, 2], n), 0, 5)
        t2[:300] = base[:300]            # 25% off by two: kappa stays high, within-one falls to 75%
    reads = pd.DataFrame({"t1": t1, "t2": t2})
    th = base[:100].copy()
    tm = th.copy() if audit_ok else rng.integers(0, 6, 100)
    audit = pd.DataFrame({"t_model": tm, "t_hand": th})
    t_all = pd.Series(base if bunch_ok else rng.choice([2, 3], n))
    return reads, audit, t_all


_VIAB_OK = {"ok": True, "reasons": []}
_VIAB_BAD = {"ok": False, "reasons": ["low: 7 flagged events (< 8)"]}


def test_gate():
    import gg_transparency as T
    r, a, t = _gate_inputs()
    g = T.evaluate_gate(r, a, t, _VIAB_OK, "abc123")
    check("gate passes on clean inputs", g["passed"] and g["tests_enabled"] == ["T1", "T2", "T3", "T4"] and not g["failures"], str(g))
    check("gate records the config hash", g["config_hash"] == "abc123")
    r, a, t = _gate_inputs(kappa_ok=False)
    g = T.evaluate_gate(r, a, t, _VIAB_OK, "x")
    check("gate fails on kappa between reads", not g["passed"] and "kappa_reads" in g["failures"] and g["tests_enabled"] == [], str(g["failures"]))
    r, a, t = _gate_inputs(within_ok=False)
    g = T.evaluate_gate(r, a, t, _VIAB_OK, "x")
    check("gate fails on within-one agreement", "within_one" in g["failures"], str(g["failures"]))
    r, a, t = _gate_inputs(audit_ok=False)
    g = T.evaluate_gate(r, a, t, _VIAB_OK, "x")
    check("gate fails on kappa vs hand audit", "kappa_audit" in g["failures"], str(g["failures"]))
    r, a, t = _gate_inputs(bunch_ok=False)
    g = T.evaluate_gate(r, a, t, _VIAB_OK, "x")
    check("gate fails on bunching", "max_adjacent_share" in g["failures"], str(g["failures"]))
    r, a, t = _gate_inputs()
    g = T.evaluate_gate(r, a, t, _VIAB_BAD, "x")
    check("viability-only failure keeps T2 and T3", g["passed"] and g["tests_enabled"] == ["T2", "T3"]
          and g["viability_failures"] == _VIAB_BAD["reasons"], str(g))


# ---------------------------------------------------------------- task 6: freeze, samples, audit sheets, leakage
_CFG = {"model": "model-x", "prompts": ["prompt A", "prompt B"], "extraction_version": "ex1",
        "rubric_version": "r1", "rubric_sha256": "deadbeef", "temperature": "default"}


def test_freeze():
    import gg_transparency as T
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "freeze.json"
        try:
            T.assert_frozen(path, _CFG)
            check("missing freeze file raises NotFrozen", False)
        except T.NotFrozen:
            check("missing freeze file raises NotFrozen", True)
        h = T.freeze(path, _CFG)
        check("freeze returns the config hash", h == T.config_hash(_CFG))
        T.assert_frozen(path, _CFG)
        check("unchanged config passes assert_frozen", True)
        for label, cfg in [("model", {**_CFG, "model": "model-y"}), ("prompt", {**_CFG, "prompts": ["prompt A", "prompt C"]}),
                           ("extraction version", {**_CFG, "extraction_version": "ex2"}), ("rubric hash", {**_CFG, "rubric_sha256": "x"})]:
            try:
                T.assert_frozen(path, cfg)
                check(f"changed {label} raises NotFrozen", False)
            except T.NotFrozen:
                check(f"changed {label} raises NotFrozen", True)
        import json
        rec = json.loads(path.read_text(encoding="utf-8"))
        check("freeze file records the cut rule and model", rec["cut_rule"] == T.CUT_RULE and rec["model"] == "model-x")


def test_motivating_env():
    import gg_transparency as T
    old = os.environ.pop("GG_MOTIVATING_GVKEYS", None)
    try:
        check("unset -> empty set", T.motivating_gvkeys() == set())
        os.environ["GG_MOTIVATING_GVKEYS"] = "1234, 56789"
        check("comma list -> zero-padded set", T.motivating_gvkeys() == {"001234", "056789"}, str(T.motivating_gvkeys()))
    finally:
        os.environ.pop("GG_MOTIVATING_GVKEYS", None)
        if old is not None:
            os.environ["GG_MOTIVATING_GVKEYS"] = old


def _world(seed=0, n_events=900):
    rng = np.random.default_rng(seed)
    sics = ["0100", "1300", "2000", "3600", "4800", "5200", "6000", "7370", "8000"]
    firms = [f"{i:06d}" for i in range(1, 61)]
    rows = []
    for g in firms:
        for y in range(2003, 2026):
            rows.append((g, "10-K", f"{y}-03-01", f"{g}-{y}"))
    idx = _index(rows).assign(sic=lambda d: d["gvkey"].map(lambda g: sics[int(g) % len(sics)]))
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(n_events)], "gvkey": rng.choice(firms, n_events),
                       "dep_ann": pd.to_datetime("2004-06-01") + pd.to_timedelta(rng.integers(0, 365 * 21, n_events), unit="D"),
                       "bhar_252": rng.normal(0, 1, n_events)})
    return ev, idx


def test_select_units():
    import gg_transparency as T
    ev, idx = _world()
    flagged = set(ev["event_id"][:50])
    mot = {"000007"}
    first = T.select_units(ev, idx, 60, 1, exclude_gvkeys=set(), exclude_event_ids=set(), exclude_accessions=set())
    taken = set(first["accession"][:10])
    s = T.select_units(ev, idx, 120, 2, exclude_gvkeys=mot, exclude_event_ids=flagged, exclude_accessions=taken)
    check("returns n units", len(s) == 120, str(len(s)))
    check("flagged events never appear", not (set(s["event_id"]) & flagged))
    check("motivating gvkey never appears", "000007" not in set(s["gvkey"]))
    check("excluded accessions never appear", not (set(s["accession"]) & taken))
    check("unique by accession", s["accession"].is_unique)
    check("every era stratum represented", set(s["era"]) == {"le2009", "2010s", "ge2020"}, str(set(s["era"])))
    check("several SIC divisions represented", s["sic_div"].nunique() >= 5, str(s["sic_div"].nunique()))
    check("no return column in the output", not any("bhar" in c or "car" in c or "ret" in c for c in s.columns), str(list(s.columns)))
    check("filing is strictly before the announcement", (s["filing_date"] < s["dep_ann"]).all())
    again = T.select_units(ev, idx, 120, 2, exclude_gvkeys=mot, exclude_event_ids=flagged, exclude_accessions=taken)
    check("same seed gives the same sample", list(again["accession"]) == list(s["accession"]))


def test_disjoint():
    import gg_transparency as T
    T.assert_disjoint(calib={"a", "b"}, pilot={"c"}, audit={"d"})
    check("disjoint sets pass", True)
    try:
        T.assert_disjoint(calib={"a", "b"}, pilot={"b", "c"})
        check("overlap by accession raises OverlapError naming the pair", False)
    except T.OverlapError as e:
        check("overlap by accession raises OverlapError naming the pair", "calib" in str(e) and "pilot" in str(e) and "b" in str(e), str(e))


def _scored(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.choice(6, n, p=[0.1, 0.1, 0.1, 0.2, 0.2, 0.3])
    return pd.DataFrame({"unit_id": [f"u{i}" for i in range(n)], "text": ["segment text " * 1000] * n, "t_final": t,
                         "t_model": t, "sic_div": rng.choice(list("DEHI"), n)})


def test_audit_sheet():
    import gg_transparency as T
    df = _scored()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.csv"
        T.make_audit_sheet(df, path, n=60, seed=0, design="representative")
        sheet = pd.read_csv(path)
        check("audit sheet hides model scores (columns unit_id, excerpt, t_hand only)",
              list(sheet.columns) == ["unit_id", "excerpt", "t_hand"], str(list(sheet.columns)))
        check("t_hand is blank", sheet["t_hand"].isna().all())
        check("excerpt capped at 8000 characters", sheet["excerpt"].str.len().max() <= 8000)
        check("60 distinct units", sheet["unit_id"].nunique() == 60)
        check("full text kept in a separate file", (Path(tmp) / "audit_full.jsonl").exists())
        T.make_audit_sheet(df, Path(tmp) / "diag.csv", n=30, seed=0, design="diagnostic")
        diag = pd.read_csv(Path(tmp) / "diag.csv").merge(df[["unit_id", "t_final"]], on="unit_id")
        counts = diag["t_final"].value_counts()
        check("diagnostic sample covers every T level about equally", len(counts) == 6 and counts.max() - counts.min() <= 1, str(counts.to_dict()))
    # representative design is not stratified: inclusion probability tracks the population share of each level
    share0 = []
    for seed in range(150):
        with tempfile.TemporaryDirectory() as tmp:
            T.make_audit_sheet(df, Path(tmp) / "a.csv", n=60, seed=seed, design="representative")
            ids = set(pd.read_csv(Path(tmp) / "a.csv")["unit_id"])
        share0.append(df[df["unit_id"].isin(ids)]["t_final"].eq(5).mean())
    pop = df["t_final"].eq(5).mean()
    check("representative audit keeps the population mix", abs(np.mean(share0) - pop) < 0.02, f"{np.mean(share0):.3f} vs {pop:.3f}")


def test_leakage_and_tokens():
    import gg_transparency as T
    g = pd.DataFrame({"guess": ["Acme Corp", "acme corp ", "Beta Inc", ""], "truth": ["ACME CORP", "Acme Corp", "Gamma", "Delta"]})
    check("leakage rate is the share of correct guesses (case-insensitive, trimmed)", T.leakage_rate(g) == 0.5)
    check("estimate_tokens is characters // 4", T.estimate_tokens(["a" * 400, "b" * 8]) == 102)


# ---------------------------------------------------------------- task 7: production scorer (fake runner)
_REPLY = '{"engines": [{"line": "Consumables", "profit_weight": "high", "price": true, "quantity": true}], "T": 4}'


class _FakeRunner:
    def __init__(self, replies=None):
        self.prompts, self.replies = [], list(replies or [])

    def __call__(self, prompt):
        self.prompts.append(prompt)
        r = self.replies.pop(0) if self.replies else _REPLY
        if isinstance(r, Exception):
            raise r
        return r


def _units(n=3):
    return pd.DataFrame({"unit_id": [f"u{i}" for i in range(n)], "text": [f"excerpt number {i} about segments" for i in range(n)]})


def test_measurement_config():
    import gg_transparency as T
    c = T.measurement_config("model-x")
    check("config has two distinct prompt templates", len(c["prompts"]) == 2 and c["prompts"][0] != c["prompts"][1])
    check("both prompts carry the rubric and an excerpt slot", all("PROFIT ENGINES" in p and "<<EXCERPT>>" in p for p in c["prompts"]))
    check("config records versions and the rubric hash", c["extraction_version"] == T.EXTRACTION_VERSION
          and c["rubric_version"] == T.RUBRIC_VERSION and len(c["rubric_sha256"]) == 64 and c["temperature"] == "default")
    check("config hash changes with the model", T.config_hash(c) != T.config_hash(T.measurement_config("model-y")))


def test_score_reads():
    import gg_transparency as T
    cfg = T.measurement_config("model-x")
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "scores.jsonl"
        run = _FakeRunner()
        out = T.score_reads(_units(3), cfg=cfg, cache_path=cache, runner=run)
        check("two reads per unit", len(out) == 6 and set(out["read"]) == {1, 2} and (out["score"] == 4).all())
        check("two reads use different prompts", len({p.split("<<EXCERPT>>")[0] for p in run.prompts[:2]}) == 2
              and "excerpt number 0" in run.prompts[0] and "excerpt number 0" in run.prompts[1])
        import hashlib
        row = out[(out["unit_id"] == "u0") & (out["read"] == 1)].iloc[0]
        check("rows carry config hash and input hash", row["config_hash"] == T.config_hash(cfg)
              and row["input_sha256"] == hashlib.sha256("excerpt number 0 about segments".encode()).hexdigest())
        check("engines are stored for audit", "Consumables" in row["engines"])
        run2 = _FakeRunner()
        again = T.score_reads(_units(3), cfg=cfg, cache_path=cache, runner=run2)
        check("resume skips units already done", len(run2.prompts) == 0 and len(again) == 6)
        run3 = _FakeRunner()
        T.score_reads(_units(3), cfg=T.measurement_config("model-y"), cache_path=cache, runner=run3)
        check("cache entries from another config are not reused", len(run3.prompts) == 6, str(len(run3.prompts)))
        run4 = _FakeRunner()
        edited = _units(3).assign(text=lambda d: d["text"].where(d["unit_id"] != "u1", "different text"))
        T.score_reads(edited, cfg=cfg, cache_path=cache, runner=run4)
        check("changed input text is re-scored", len(run4.prompts) == 2, str(len(run4.prompts)))
        third = T.score_reads(_units(1), cfg=cfg, cache_path=cache, runner=_FakeRunner(), reads=(3,))
        check("a third read can be requested", list(third["read"]) == [3])

    with tempfile.TemporaryDirectory() as tmp:
        run = _FakeRunner(["I think it is a 3", "still prose"])
        out = T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "s.jsonl", runner=run, reads=(1,), max_retries=1)
        check("unparseable reply retried once then NaN", len(run.prompts) == 2 and np.isnan(out.iloc[0]["score"]))
        run = _FakeRunner(["prose", _REPLY])
        out = T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "s2.jsonl", runner=run, reads=(1,), max_retries=1)
        check("a valid retry is accepted", out.iloc[0]["score"] == 4)
        bad = '{"engines": [{"line": "Inj", "profit_weight": "high", "price": false, "quantity": false}], "T": 4}'
        out = T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "s3.jsonl", runner=_FakeRunner([bad]), reads=(1,))
        check("rule_check flags are recorded on the row", "lacks price or quantity" in out.iloc[0]["flags"], str(out.iloc[0]["flags"]))

    with tempfile.TemporaryDirectory() as tmp:
        waits = []
        run = _FakeRunner([T.UsageLimit("usage limit reached"), _REPLY])
        out = T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "l.jsonl", runner=run, reads=(1,), sleeper=waits.append)
        check("usage limit: sleeps then resumes and succeeds", len(waits) == 1 and out.iloc[0]["score"] == 4, str(waits))

    with tempfile.TemporaryDirectory() as tmp:
        freeze_path = Path(tmp) / "freeze.json"
        try:
            T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "f.jsonl", runner=_FakeRunner(), require_frozen=freeze_path)
            check("refuses when a freeze is required and missing", False)
        except T.NotFrozen:
            check("refuses when a freeze is required and missing", True)
        T.freeze(freeze_path, cfg)
        out = T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "f.jsonl", runner=_FakeRunner(), require_frozen=freeze_path)
        check("runs when the config matches the freeze", len(out) == 2)
        try:
            T.score_reads(_units(1), cfg=T.measurement_config("model-z"), cache_path=Path(tmp) / "g.jsonl",
                          runner=_FakeRunner(), require_frozen=freeze_path)
            check("refuses a config that differs from the freeze", False)
        except T.NotFrozen:
            check("refuses a config that differs from the freeze", True)


# ---------------------------------------------------------------- task 9: attach T, standardise once, validate
_META = {"mu": 3.0, "sd": 1.5}


def _scores(rows):
    return pd.DataFrame(rows, columns=["accession", "t_final", "status"])


def test_attach():
    import gg_stage5 as S
    idx = _index([("000001", "10-K", "2019-03-01", "a19"), ("000001", "10-K", "2020-03-01", "a20"),
                  ("000002", "10-K", "2020-03-01", "b20"), ("000003", "10-K", "2020-03-01", "c20"),
                  ("000004", "10-K", "2020-03-01", "d20"), ("000005", "10-K", "2020-03-01", "e20")])
    sc = _scores([("a19", 1.0, "ok"), ("a20", 5.0, "ok"), ("b20", 3.0, "limited_disclosure"),
                  ("c20", np.nan, "extraction_failed")])           # d20 and e20 were never scored
    ev = pd.DataFrame({"gvkey": ["000001", "000001", "000002", "000003", "000004", "000009", "000005"],
                       "dep_ann": pd.to_datetime(["2020-03-01", "2020-06-01", "2020-06-01", "2020-06-01", "2020-06-01",
                                                  "2020-06-01", "2019-01-01"])})
    out = S.attach_T(ev, sc, idx, _META, date_col="dep_ann", strict=True)
    check("event on the filing day uses the earlier filing's score", out.loc[0, "T"] == 1.0 and out.loc[0, "accession"] == "a19")
    check("later event uses the latest filing", out.loc[1, "T"] == 5.0)
    check("limited_disclosure filings are scored", out.loc[2, "T"] == 3.0 and out.loc[2, "T_status"] == "limited_disclosure")
    check("failed extraction -> NaN with reason", np.isnan(out.loc[3, "T"]) and out.loc[3, "T_missing_reason"] == "extraction_failed")
    check("never-scored filing -> not_scored", np.isnan(out.loc[4, "T"]) and out.loc[4, "T_missing_reason"] == "not_scored")
    check("unknown firm -> no_filing, NaN not zero", np.isnan(out.loc[5, "T"]) and out.loc[5, "T_missing_reason"] == "no_filing")
    check("no 10-K before an early event -> no_filing", np.isnan(out.loc[6, "T"]) and out.loc[6, "T_missing_reason"] == "no_filing")
    check("scored rows have an empty reason", out.loc[1, "T_missing_reason"] == "")
    check("T_z uses the stored mean and sd", abs(out.loc[1, "T_z"] - (5.0 - 3.0) / 1.5) < 1e-12)
    check("row order and index preserved", list(out.index) == list(ev.index) and list(out["gvkey"]) == list(ev["gvkey"]))
    q = S.attach_T(ev.iloc[[0]], sc, idx, _META, date_col="dep_ann", strict=False)
    check("non-strict (quarter start) includes the same-day filing", q.iloc[0]["T"] == 5.0)


def test_attach_matches_lookup():
    import gg_filings as F
    import gg_stage5 as S
    ev, idx = _world(seed=3, n_events=300)
    amend = _index([(g, "10-K/A", f"{y}-09-01", f"{g}-{y}-A") for g in ("000001", "000002") for y in range(2008, 2012)]).assign(sic="3600")
    idx = pd.concat([idx, amend], ignore_index=True)
    sc = _scores([(a, float(i % 6), "ok") for i, a in enumerate(idx["accession"])])
    out = S.attach_T(ev, sc, idx, _META, date_col="dep_ann", strict=True)
    looked = [F.last_10k_before(idx, g, d) for g, d in zip(ev["gvkey"], ev["dep_ann"])]
    exp = [None if r is None else r["accession"] for r in looked]
    got = [a if isinstance(a, str) else None for a in out["accession"]]
    check("vectorised attach agrees with last_10k_before (incl. amendments) on 300 events", got == exp,
          f"{sum(g != e for g, e in zip(got, exp))} differ")


def test_standardisation_and_coverage():
    import gg_stage5 as S
    idx = _index([("000001", "10-K", "2019-03-01", "a19"), ("000001", "10-K", "2020-03-01", "a20")])
    sc = _scores([("a19", 4.0, "ok"), ("a20", 5.0, "ok")])
    ev1 = pd.DataFrame({"gvkey": ["000001"], "dep_ann": pd.to_datetime(["2020-01-01"])})
    q = pd.DataFrame({"gvkey": ["000001"] * 4, "qs": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"])})
    a = S.attach_T(ev1, sc, idx, _META, date_col="dep_ann")
    b = S.attach_T(q, sc, idx, _META, date_col="qs", strict=False)
    check("same filing gets the same T_z in events and quarters", b["T_z"].iloc[0] == a["T_z"].iloc[0]
          and b["T_z"].iloc[1] == b["T_z"].iloc[3] and b["T_z"].iloc[1] != b["T_z"].iloc[0])
    cov = S.coverage_report(pd.concat([a, b[["gvkey", "T", "T_z", "T_missing_reason", "T_status"]].assign(
        dep_ann=pd.to_datetime("2021-01-01"))], ignore_index=True), "dep_ann")
    check("coverage report counts reasons and scored share by year", cov["reasons"].get("", 0) == 5
          and cov["scored_share_by_year"][2020] == 1.0, str(cov))
    meta = S.score_meta(pd.DataFrame({"accession": ["a", "b", "c", "c"], "t_final": [1.0, 3.0, 5.0, 5.0]}))
    check("meta is computed once per unique accession (quarterly repeats not double counted)",
          abs(meta["mu"] - 3.0) < 1e-12 and meta["n"] == 3, str(meta))


def test_validation_report():
    import gg_stage5 as S
    rng = np.random.default_rng(0)
    n = 600
    div = rng.choice(list("DEHI"), n)
    year = rng.integers(2005, 2025, n)
    nseg = rng.integers(1, 9, n)
    t = np.clip(np.round(np.where(div == "D", 4, 1) + rng.normal(0, 0.4, n)), 0, 5)
    scores = pd.DataFrame({"accession": [f"a{i}" for i in range(n)], "t_final": t, "status": "ok"})
    meta = pd.DataFrame({"accession": scores["accession"], "n_segments": nseg, "log_assets": rng.normal(8, 1, n),
                         "sic_div": div, "year": year})
    rep = S.validation_report(scores, meta)
    check("variance of T explained by industry and year is reported and high here", rep["r2_industry_year"] > 0.8, str(rep["r2_industry_year"]))
    check("T is nearly uncorrelated with segment count when constructed independent", abs(rep["spearman_segments"]) < 0.15, str(rep["spearman_segments"]))
    check("mean T by division reported", set(rep["mean_T_by_division"]) == set("DEHI"))
    check("limited-disclosure share by division reported", "limited_share_by_division" in rep and "mean_T_by_year" in rep)


# ---------------------------------------------------------------- task 10: T1 (centred placebo contrast by T group)
def _sim_t1(seed, n=300, days=800, eff_low=0.0, eff_high=0.0, t_gradient=None, constant_T=None):
    """Synthetic event windows and factors. Flagged events: 15 with T <= 1 and 15 with T >= 4 (cut points 1 and 4).
    Planted annual alpha eff_low / eff_high in the flagged events of each group; t_gradient plants alpha linear in T_z."""
    import gg_tests as GT
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2010-01-01", periods=days)
    fac = pd.DataFrame(rng.normal(0, 0.008, (days, 4)), index=cal, columns=GT.FACTORS_FF4)
    fac["rf"] = 0.0001
    T_ = rng.integers(0, 6, n) if constant_T is None else np.full(n, constant_T)
    start = rng.integers(0, days - 252, n)
    low_idx = np.where(T_ <= 1)[0]
    high_idx = np.where(T_ >= 4)[0]
    flagged = np.zeros(n, dtype=bool)
    if constant_T is None:
        flagged[rng.choice(low_idx, 15, replace=False)] = True
        flagged[rng.choice(high_idx, 15, replace=False)] = True
    else:
        flagged[:30] = True
    betas = np.array([1.0, 0.3, 0.2, -0.1])
    windows = {}
    for i in range(n):
        idx = np.arange(start[i], start[i] + 252)
        alpha = 0.0
        if flagged[i]:
            alpha = eff_low if T_[i] <= 1 else (eff_high if T_[i] >= 4 else 0.0)
            if t_gradient is not None:
                alpha = t_gradient * (T_[i] - 2.5) / 1.7
        r = 0.0001 + fac[GT.FACTORS_FF4].values[idx] @ betas + alpha / 252 + rng.normal(0, 0.012, 252)
        windows[f"e{i}"] = (idx, r)
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "T": T_.astype(float), "year": 2010 + start // 250,
                       "ann_quarter": start // 63, "flag": flagged})
    ev["T_z"] = (ev["T"] - 2.5) / 1.7
    return windows, ev, fac, days


def _run_sim(seed, **kw):
    import gg_stage5 as S
    import gg_transparency as T
    windows, ev, fac, days = _sim_t1(seed, **{k: v for k, v in kw.items() if k in ("eff_low", "eff_high", "constant_T")})
    cuts = T.cut_points(ev["T"])
    return S.run_t1(windows, ev, fac, days, set(ev.loc[ev["flag"], "event_id"]), cuts=cuts, reps=kw.get("reps", 100),
                    seed=seed, n_boot=0, with_spread=False, with_sens=False)


def test_t1_core():
    import gg_stage5 as S
    import gg_tests as GT
    import gg_transparency as T
    windows, ev, fac, days = _sim_t1(1, eff_low=-0.25)
    cuts = T.cut_points(ev["T"])
    check("sim cut points are 1 and 4", cuts == (1.0, 4.0), str(cuts))
    flagged = set(ev.loc[ev["flag"], "event_id"])
    r = S.run_t1(windows, ev, fac, days, flagged, cuts=cuts, reps=300, seed=1, n_boot=200, with_sens=False)
    check("planted -25%/yr in low-T flagged events: did < 0", r["did"] < -0.10, str(r.get("did")))
    check("planted low-T effect: perm_p < 0.05", r["perm_p"] < 0.05, str(r["perm_p"]))
    check("low group alpha is strongly negative, high near 0", r["groups"]["low"]["alpha_ann"] < -0.12
          and abs(r["groups"]["high"]["alpha_ann"]) < 0.12, str(r["groups"]))
    check("group sizes reported", r["groups"]["low"]["n"] == 15 and r["groups"]["high"]["n"] == 15)
    check("bootstrap CI brackets the estimate and is below 0", r["ci"][0] < r["did"] < r["ci"][1] and r["ci"][1] < 0.05, str(r["ci"]))
    check("spread alpha and mde reported", np.isfinite(r["spread_alpha"]) and r["mde"] > 0)
    d = GT.placebo_difference.__name__
    check("placebo_difference exposed in gg_tests", d == "placebo_difference")
    flagged_df = ev[ev["flag"]]
    pool = ev[~ev["flag"]]
    lo_f, hi_f = flagged_df[flagged_df["T"] <= 1], flagged_df[flagged_df["T"] >= 4]
    lo_p, hi_p = pool[pool["T"] <= 1], pool[pool["T"] >= 4]
    null = GT.placebo_difference(windows, lo_f, hi_f, lo_p, hi_p, fac, days, reps=200, seed=3)
    check("placebo contrast null is centred at zero", abs(np.nanmean(null)) < 1e-9 and np.isfinite(null).all(), str(np.nanmean(null)))


def test_event_T_uses_window_start_date():
    import gg_stage5 as S
    idx = _index([("000001", "10-K", "2014-03-01", "a14"), ("000001", "10-K", "2015-02-25", "a15"), ("000002", "10-K", "2014-03-01", "b14")])
    sc = _scores([("a14", 1.0, "ok"), ("a15", 5.0, "ok"), ("b14", 3.0, "ok")])
    events = pd.DataFrame({"event_id": ["e1", "e2"], "year": [2015, 2015]})
    ev_final = pd.DataFrame({"event_id": ["e1", "e2"], "gvkey": ["1", "2"], "dep_ann": pd.to_datetime(["2015-03-05", "2015-03-05"]),
                             "dep_ann_final": pd.to_datetime(["2015-02-10", "2015-03-05"])})
    out = S.attach_event_T(events, ev_final, sc, idx, _META)
    d = out.set_index("event_id")
    check("an annotator-dated event takes the 10-K before the real announcement, not the one filed in between",
          d.loc["e1", "accession"] == "a14" and d.loc["e1", "T"] == 1.0, str(d.loc["e1", ["accession", "T"]].tolist()))
    check("an event without an annotator date uses its rule date as before", d.loc["e2", "accession"] == "b14")
    check("the announcement quarter follows the same date as the windows", d.loc["e1", "ann_quarter"] == "2015Q1" and d.loc["e1", "dep_ann_final"] == pd.Timestamp("2015-02-10"))
    check("rows and event ids are preserved", list(out["event_id"]) == ["e1", "e2"])


def test_t1_pool_requires_announcement():
    import gg_stage5 as S
    import gg_transparency as T
    windows, ev, fac, days = _sim_t1(3, eff_low=-0.1)
    cuts = T.cut_points(ev["T"])
    flagged = set(ev.loc[ev["flag"], "event_id"])
    base = S.run_t1(windows, ev, fac, days, flagged, cuts=cuts, reps=20, seed=0, n_boot=0, with_spread=False, with_sens=False)
    lo_pool = ev[(~ev["flag"]) & (ev["T"] <= cuts[0])]
    ev2 = ev.assign(ann_found=1)
    ev2.loc[lo_pool.index[:10], "ann_found"] = 0                  # ten low-T placebo events have no identified announcement (pool stays above 5x flagged)
    r = S.run_t1(windows, ev2, fac, days, flagged, cuts=cuts, reps=20, seed=0, n_boot=0, with_spread=False, with_sens=False)
    want = int(((~ev2["flag"]) & (ev2["T"] <= cuts[0]) & (ev2["ann_found"] == 1)).sum())
    check("placebo pool counts only events with an identified announcement", r["groups"]["low"]["pool"] == want and want < base["groups"]["low"]["pool"], f"{r['groups']['low']['pool']} vs {want} vs {base['groups']['low']['pool']}")
    check("flagged events are not dropped by the pool rule", r["groups"]["low"]["n"] == base["groups"]["low"]["n"] and r["groups"]["high"]["n"] == base["groups"]["high"]["n"])
    ev3 = ev.assign(ann_found=0)
    ev3.loc[ev3["flag"], "ann_found"] = 0
    r3 = S.run_t1(windows, ev3, fac, days, flagged, cuts=cuts, reps=20, seed=0, n_boot=0, with_spread=False, with_sens=False)
    check("with no identified announcements there is no pool, so T1 refuses", "nan_reason" in r3, str(list(r3)))


def test_t1_reasons():
    import gg_stage5 as S
    import gg_transparency as T
    windows, ev, fac, days = _sim_t1(2, constant_T=3)
    flagged = set(ev.loc[ev["flag"], "event_id"])
    r = S.run_t1(windows, ev, fac, days, flagged, cuts=T.cut_points(ev["T"]), reps=20, seed=0, n_boot=0, with_spread=False, with_sens=False)
    check("all-equal T returns a reason instead of crashing", "nan_reason" in r and "flagged" in r["nan_reason"], str(r))
    windows, ev, fac, days = _sim_t1(3)
    low_all = set(ev.loc[ev["T"] <= 1, "event_id"])        # every low-T event flagged: no placebo pool left
    r = S.run_t1(windows, ev, fac, days, low_all | set(ev.loc[ev["flag"], "event_id"]), cuts=(1.0, 4.0), reps=20, seed=0,
                 n_boot=0, with_spread=False, with_sens=False)
    check("empty placebo pool returns a reason", "nan_reason" in r and "pool" in r["nan_reason"], str(r))


def test_t1_rejection_rates():
    eq = [_run_sim(s, eff_low=-0.10, eff_high=-0.10)["perm_p"] for s in range(200)]
    rate = float(np.mean(np.array(eq) < 0.05))
    check("equal -10%/yr effects in both groups are NOT rejected (rate between 1.5% and 10%)", 0.015 <= rate <= 0.10, f"{rate:.3f}")
    nul = [_run_sim(1000 + s)["perm_p"] for s in range(200)]
    rate0 = float(np.mean(np.array(nul) < 0.05))
    check("no effect anywhere: false positive rate near 5% (1.5% to 10%)", 0.015 <= rate0 <= 0.10, f"{rate0:.3f}")


def test_t1_continuous():
    import gg_stage5 as S
    rng = np.random.default_rng(0)
    n = 400
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "T_z": rng.normal(0, 1, n), "flag": rng.random(n) < 0.3,
                       "ann_quarter": rng.integers(0, 40, n)})
    y = pd.Series(rng.normal(0, 0.3, n) + np.where(ev["flag"], -0.2 + 0.15 * ev["T_z"], 0.0), index=ev["event_id"])
    r = S.run_t1_continuous(ev, y)
    check("continuous gradient recovered: flag x T_z positive and significant", r["coef"] > 0.05 and r["p_one"] < 0.05, str(r))
    y0 = pd.Series(rng.normal(0, 0.3, n), index=ev["event_id"])
    r0 = S.run_t1_continuous(ev, y0)
    check("no gradient: p_one not small", r0["p_one"] > 0.02, str(r0))


# ---------------------------------------------------------------- task 11: T2 and T3 regressions with T
def _sim_p1(seed=0, planted=-3.0, n_firms=120, n_q=24):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        tz_by_year = rng.normal(0, 1, n_q // 4 + 1)
        for q in range(n_q):
            rows.append((f"{f:06d}", f"{2015 + q // 4}Q{q % 4 + 1}", 2015 + q // 4, tz_by_year[q // 4]))
    d = pd.DataFrame(rows, columns=["gvkey", "tq", "fy", "T_z"])
    n = len(d)
    d["opt_pos"] = rng.integers(0, 2, n).astype(float)
    d["salepost"] = (rng.random(n) < 0.25).astype(float)
    d["T"] = d["T_z"] * 1.5 + 3
    d["opt"] = d["opt_pos"]
    for c in ["sale12", "lmcap", "ret12", "ltenure", "age"]:
        d[c] = rng.normal(0, 1, n)
    d["sale12"] = (d["sale12"] > 0.5).astype(float)
    d["log_assets"], d["n_segments"] = rng.normal(8, 1, n), rng.integers(1, 8, n).astype(float)
    y = 2 + planted * d["T_z"] * d["opt_pos"] * d["salepost"] + rng.normal(0, 5, n)
    d["exit"] = y / 100.0
    return d


def _sim_p2(seed=0, planted=3.0, n_firms=100, n_q=24):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        tz_by_year = rng.normal(0, 1, n_q // 4 + 1)
        for q in range(n_q):
            if rng.random() < 0.45:
                rows.append((f"{f:06d}", f"{2015 + q // 4}Q{q % 4 + 1}", pd.Timestamp(2015 + q // 4, q % 4 * 3 + 1, 15), tz_by_year[q // 4]))
    d = pd.DataFrame(rows, columns=["gvkey", "tq", "t", "T_z"])
    n = len(d)
    d["opt_pos"] = rng.integers(0, 2, n).astype(float)
    d["opt"] = d["opt_pos"]
    d["sale"] = (rng.random(n) < 0.35).astype(float)
    d["exit_in"] = (rng.random(n) < 0.35).astype(float)
    d["days_to_p"] = rng.integers(10, 200, n)
    d["pdicity"] = rng.choice(["QTR", "ANN"], n)
    for c in ["lmcap", "ret12", "ltenure", "age"]:
        d[c] = rng.normal(0, 1, n)
    d["log_assets"], d["n_segments"] = rng.normal(8, 1, n), rng.integers(1, 8, n).astype(float)
    d["miss"] = (planted * d["T_z"] * d["opt_pos"] * d["sale"] * d["exit_in"] + rng.normal(0, 5, n)) / 100.0
    return d


def test_p1_T():
    import gg_stage5 as S
    d = _sim_p1(0, planted=-3.0)
    r = S.run_p1_T(d)
    check("P1: planted negative T x O x S recovered (p_one < 0.05)", r["TOS"]["coef"] < -1.5 and r["p_one"] < 0.05, str(r.get("p_one")))
    check("P1: lower-order terms T, TO, TS, TOS all present", all(k in r for k in ["T", "TO", "TS", "TOS", "OS", "O", "S"]))
    flip = S.run_p1_T(_sim_p1(0, planted=+3.0))
    check("P1: sign orientation (positive planted -> p_one > 0.5)", flip["p_one"] > 0.5, str(flip["p_one"]))
    dry = S.run_p1_T(d, dry=True, rng=np.random.default_rng(1))
    check("P1: dry run randomises the outcome (planted effect gone, not significant)", dry["p_one"] > 0.01 and dry["TOS"]["coef"] > r["TOS"]["coef"] + 1.0, str((dry["TOS"]["coef"], dry["p_one"])))
    cx = S.run_p1_T(d, complexity=True)
    check("P1: complexity sensitivity adds log assets and segment count terms", "LA" in cx and "NS" in cx and "TOS" in cx, str(list(cx)))
    check("P1: mde reported", r["mde"] > 0)


def test_p2_T():
    import gg_stage5 as S
    d = _sim_p2(0, planted=4.0)
    r = S.run_p2_T(d)
    check("P2: planted positive T x O x S x X recovered (p_one < 0.05)", r["TOSX"]["coef"] > 1.5 and r["p_one"] < 0.05, str(r.get("p_one")))
    need = ["T", "TO", "TS", "TX", "TOS", "TOX", "TSX", "TOSX", "O", "S", "X", "OS", "OX", "SX", "OSX"]
    check("P2: all lower-order terms present", all(k in r for k in need), str([k for k in need if k not in r]))
    flip = S.run_p2_T(_sim_p2(0, planted=-4.0))
    check("P2: sign orientation (negative planted -> p_one > 0.5)", flip["p_one"] > 0.5, str(flip["p_one"]))
    dry = S.run_p2_T(d, dry=True, rng=np.random.default_rng(2))
    check("P2: dry run randomises the outcome (planted effect gone, not significant)", dry["p_one"] > 0.01 and dry["TOSX"]["coef"] < r["TOSX"]["coef"] - 1.0, str((dry["TOSX"]["coef"], dry["p_one"])))
    cx = S.run_p2_T(d, complexity=True)
    check("P2: complexity sensitivity adds terms", "LA" in cx and "NS" in cx and "TOSX" in cx)


def test_permute_T_blocks():
    import gg_stage5 as S
    d = _sim_p1(0)
    out = S.permute_T_blocks(d, block=("gvkey", "fy"), within="fy", rng=np.random.default_rng(0))
    check("T stays constant within a firm-year block", out.groupby(["gvkey", "fy"])["T_z"].nunique().max() == 1)
    same_year = all(sorted(out.loc[out["fy"] == y].drop_duplicates(["gvkey", "fy"])["T_z"].round(9))
                    == sorted(d.loc[d["fy"] == y].drop_duplicates(["gvkey", "fy"])["T_z"].round(9)) for y in d["fy"].unique())
    check("the multiset of T within each calendar year is preserved", same_year)
    check("T is actually shuffled across firms", (out["T_z"].values != d["T_z"].values).mean() > 0.5)
    check("other columns untouched", (out["exit"].values == d["exit"].values).all() and (out["gvkey"].values == d["gvkey"].values).all())


# ---------------------------------------------------------------- task 12: T4 and the runner
def _t4_inputs(seed, **kw):
    import gg_stage5 as S
    windows, ev, fac, days = _sim_t1(seed, **kw)
    ev = ev.assign(leaver=ev["flag"])
    return windows, ev, fac, days


def _run_t4(seed, reps=199, **kw):
    import gg_stage5 as S
    import gg_transparency as T
    windows, ev, fac, days = _t4_inputs(seed, **kw)
    return S.run_t4(ev, windows, fac, days, cuts=T.cut_points(ev["T"]), reps=reps, seed=seed, n_boot=0)


def test_t4_core():
    import gg_stage5 as S
    import gg_transparency as T
    windows, ev, fac, days = _t4_inputs(1, eff_low=-0.30)
    r = S.run_t4(ev, windows, fac, days, cuts=T.cut_points(ev["T"]), reps=499, seed=1, n_boot=200)
    p = r["primary"]
    check("T4: leaver effect planted only at low T -> interaction d > 0", p["coef"] > 0.10, str(p))
    check("T4: planted effect p_one < 0.05 (wild cluster bootstrap)", p["p_one"] < 0.05, str(p["p_one"]))
    check("T4: group sizes and effects reported", r["groups"]["low"]["leavers"] == 15 and r["groups"]["high"]["leavers"] == 15
          and r["groups"]["low"]["effect"] < -0.15, str(r["groups"]))
    check("T4: continuous corroboration and secondary reported", "continuous" in r and "secondary" in r)
    sec = r["secondary"]
    check("T4: recentred bootstrap p reported and CI brackets the contrast", 0 < sec["p_lower"] <= 1 and sec["ci"][0] < sec["contrast"] < sec["ci"][1], str(sec))
    centred = S._recentred_contrast(ev, windows, fac, days, T.cut_points(ev["T"]), n_boot=300, seed=2)
    check("T4: recentred bootstrap distribution is centred at zero", abs(np.mean(centred)) < 0.03, str(np.mean(centred)))
    r0 = S.run_t4(ev.assign(T=3.0, T_z=0.0), windows, fac, days, cuts=(3.0, 3.0), reps=99, seed=0, n_boot=0)
    check("T4: degenerate T returns a reason", "nan_reason" in r0, str(r0))


def test_t4_rejection_rates():
    eq = np.array([_run_t4(s, eff_low=-0.10, eff_high=-0.10)["primary"]["p_one"] for s in range(200)])
    rate = float((eq < 0.05).mean())
    check("T4: equal -10%/yr leaver effects in both groups are NOT rejected (1.5% to 10%)", 0.015 <= rate <= 0.10, f"{rate:.3f}")
    nul = np.array([_run_t4(2000 + s)["primary"]["p_one"] for s in range(200)])
    rate0 = float((nul < 0.05).mean())
    check("T4: no leaver effect anywhere: false positive rate near 5% (1.5% to 10%)", 0.015 <= rate0 <= 0.10, f"{rate0:.3f}")


def _runner_inputs(seed=7):
    """Synthetic loader output for the runner mechanics."""
    windows, ev, fac, days = _t4_inputs(seed, eff_low=-0.25)
    ev = ev.assign(sic_div="D")
    return {"windows": windows, "events": ev, "fac": fac, "H": days, "flagged_ids": set(ev.loc[ev["flag"], "event_id"]),
            "broad_ids": None, "Q": _sim_p1(seed, planted=-3.0), "B": _sim_p2(seed, planted=4.0), "ev_all": ev}


def _setup_runner(tmp, *, gate=None, freeze=True, scores_hash=None):
    import json
    import gg_transparency as T
    cfg = T.measurement_config("model-x")
    paths = {k: Path(tmp) / f for k, f in [("freeze", "freeze.json"), ("gate", "gate.json"), ("results", "stage5_results.json"),
                                             ("dryrun", "stage5_results_DRYRUN.json")]}
    if freeze:
        T.freeze(paths["freeze"], cfg)
    if gate is not None:
        g = {"passed": True, "tests_enabled": ["T1", "T2", "T3", "T4"], "config_hash": T.config_hash(cfg), "failures": []}
        g.update(gate)
        paths["gate"].write_text(json.dumps(g), encoding="utf-8")
    paths["scores_config_hashes"] = [scores_hash or T.config_hash(cfg)]
    return paths


def test_runner():
    import json
    import gg_stage5 as S

    def run(paths, argv):
        return S.main(argv, loader=lambda dry: _runner_inputs(), paths=paths)

    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate=None)
        rc = run(p, ["--reps", "40"])
        check("runner refuses without a gate file", rc != 0 and not p["results"].exists() and not p["dryrun"].exists())
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={"passed": False, "tests_enabled": []})
        rc = run(p, ["--reps", "40"])
        check("runner refuses when the gate did not pass", rc != 0 and not p["results"].exists())
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={}, freeze=False)
        rc = run(p, ["--reps", "40"])
        check("runner refuses when the measurement is not frozen", rc != 0 and not p["results"].exists())
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={"config_hash": "someotherhash"})
        rc = run(p, ["--reps", "40"])
        check("runner refuses when the gate belongs to another config", rc != 0 and not p["results"].exists())
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={}, scores_hash="someotherhash")
        rc = run(p, ["--reps", "40"])
        check("runner refuses when the scores were made under another config", rc != 0 and not p["results"].exists())
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={})
        rc = run(p, ["--reps", "40", "--dry-run"])
        check("dry run writes only the DRYRUN file", rc == 0 and p["dryrun"].exists() and not p["results"].exists())
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={"tests_enabled": ["T2", "T3"]})
        rc = run(p, ["--reps", "40"])
        res = json.loads(p["results"].read_text(encoding="utf-8")) if p["results"].exists() else {}
        check("viability-limited gate runs only T2 and T3", rc == 0 and set(res.get("tests", {})) == {"T2", "T3"}, str(list(res.get("tests", {}))))
        check("Holm family follows the enabled tests", set(res["holm"]) <= {"T2", "T3"} and res["family_size"] == 2, str(res.get("family_size")))
        check("results record the config hash and project-wide test count",
              res["config_hash"] and res["project_wide_tests"] == res["prior_stage_tests"] + 2)
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={})
        rc = run(p, ["--reps", "60"])
        res = json.loads(p["results"].read_text(encoding="utf-8")) if p["results"].exists() else {}
        check("full run writes T1-T4 with a Holm family of four",
              rc == 0 and set(res.get("tests", {})) == {"T1", "T2", "T3", "T4"} and res["family_size"] == 4, str(list(res.get("tests", {}))))
        check("each test reports its primary p-value", all("p_primary" in t for t in res["tests"].values()))


def test_p3_refactor_static():
    """run_p3 and build_p3_events must not use a name that the extraction left behind (needs no database)."""
    import ast
    import builtins
    src = (Path(__file__).resolve().parents[1] / "src" / "gg_stage3.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    glob = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                glob |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            glob |= {a.asname or a.name.split(".")[0] for a in n.names}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("run_p3", "build_p3_events")]:
        assigned = {a.arg for a in fn.args.args}
        loads = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                (assigned.add(n.id) if isinstance(n.ctx, ast.Store) else loads.setdefault(n.id, n.lineno))
        missing = sorted(k for k in loads if k not in assigned and k not in glob and not hasattr(builtins, k))
        check(f"{fn.name} uses no undefined names", not missing, str(missing))
    check("run_p3 reads the stayer-executive count from the frame", 'ev_all.attrs["n_stayer_execs"]' in src)


# ---------------------------------------------------------------- final review fixes (C1, C2, I1-I7)
def test_review_c1_viability_deferred():
    import json
    import gg_stage5 as S
    import gg_transparency as T
    r, a, t = _gate_inputs()
    g = T.evaluate_gate(r, a, t, None, "h")
    check("C1: gate with viability deferred enables all four tests", g["passed"] and g["tests_enabled"] == ["T1", "T2", "T3", "T4"]
          and g["viability_deferred"] is True, str(g))
    g2 = T.evaluate_gate(r, a, t, _VIAB_OK, "h")
    check("C1: an explicit viability result is still honoured", g2["viability_deferred"] is False)

    def flat_loader(dry):
        inp = _runner_inputs()
        ev = inp["events"].assign(T=3.0, T_z=0.0)
        return {**inp, "events": ev, "ev_all": ev}

    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={})
        rc = S.main(["--reps", "40"], loader=flat_loader, paths=p)
        res = json.loads(p["results"].read_text(encoding="utf-8"))
        check("C1: tests found not viable at run time are skipped with a reason, not silently dropped",
              rc == 0 and set(res["skipped"]) == {"T1", "T4"} and "flagged" in res["skipped"]["T1"], str(res.get("skipped")))
        check("C1: Holm family excludes the skipped tests and says so", res["family_size"] == 2 and set(res["holm"]) <= {"T2", "T3"})


def test_review_c2_audit_excerpt():
    import gg_transparency as T
    html = _html(seg_note=_SEG, mdna="Segment results: Consumables grew 12% on volume.", filler="Business prose about the company. " * 600)
    text, st = T.extract_sections(html)
    check("C2: extraction separates Item 1 from the segment text with a marker", st == "ok" and T.SEGMENT_MARKER in text)
    df = pd.DataFrame({"unit_id": ["u1"], "text": [text], "t_final": [3], "sic_div": ["D"]})
    with tempfile.TemporaryDirectory() as tmp:
        T.make_audit_sheet(df, Path(tmp) / "a.csv", n=1, seed=0, design="representative")
        ex = pd.read_csv(Path(tmp) / "a.csv").loc[0, "excerpt"]
    check("C2: the audit excerpt is the segment and KPI text, not the start of Item 1",
          "Consumables revenue was $900 million" in ex and len(ex) <= 8000, ex[:80])
    short, st2 = T.extract_sections(_html())
    df2 = pd.DataFrame({"unit_id": ["u2"], "text": [short], "t_final": [1], "sic_div": ["D"]})
    with tempfile.TemporaryDirectory() as tmp:
        T.make_audit_sheet(df2, Path(tmp) / "b.csv", n=1, seed=0, design="representative")
        ex2 = pd.read_csv(Path(tmp) / "b.csv").loc[0, "excerpt"]
    check("C2: with no segment text the excerpt falls back to the start of the text", "overview" in ex2)


def test_review_i1_tickers():
    import gg_transparency as T
    txt = "It reports low-margin results. LOW sold more. Our key drivers: Acme Corp grew."
    out = T.strip_identity(txt, ["Acme Corp"], tickers=["IT", "LOW", "KEY"])
    check("I1: ordinary words are not mangled by tickers (case-sensitive standalone tokens only)",
          "It reports" in out and "low-margin" in out and "key drivers" in out, out)
    check("I1: the upper-case ticker token is replaced", "[COMPANY] sold more" in out, out)
    check("I1: company names still match case-insensitively", "Acme Corp" not in out and "[COMPANY] grew" in out, out)


def test_review_i2_item1_crossref():
    import gg_transparency as T
    body = ("We make products. See Item 1A, Risk Factors, for more detail. " + "Plain business description without the s-word. " * 300)
    html = ("<html><body><h2>Item 1. Business</h2><p>" + body + "</p><h2>Item 1A. Risk Factors</h2><p>" + "Risks. " * 400 + "</p></body></html>")
    text, st = T.extract_sections(html)
    check("I2: an in-text cross-reference to Item 1A does not truncate Item 1 or fail the extraction",
          st == "limited_disclosure" and len(text) > 5000, f"{st} {len(text)}")


def test_review_i3_runner_secondary():
    import json
    import gg_stage5 as S

    def loader(dry):
        inp = _runner_inputs()
        ev = inp["events"].assign(size_tercile=lambda d: (d["event_id"].str[1:].astype(int) % 3))
        return {**inp, "events": ev, "ev_all": ev, "t_sd": 1.7}

    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={})
        rc = S.main(["--reps", "60"], loader=loader, paths=p)
        res = json.loads(p["results"].read_text(encoding="utf-8"))
        sec = res.get("secondary", {})
        check("I3: runner reports the pre-registered secondary analyses",
              rc == 0 and {"T1_continuous", "T2_complexity", "T3_complexity"} <= set(sec), str(list(sec)))
        t1 = res["tests"]["T1"]["result"]
        check("I3: T1 within-size-tercile sensitivity reported", "strata_year_size_tercile" in t1.get("sens", {}), str(list(t1.get("sens", {}))))
        check("I3: per-point effect reported when the T sd is supplied", "TOS_per_point" in res["tests"]["T2"]["result"])


def test_review_i4_cli_version():
    import gg_transparency as T
    a = T.measurement_config("m", cli_version="2.1.1")
    b = T.measurement_config("m", cli_version="2.1.2")
    check("I4: the claude CLI version is part of the config hash", a["cli_version"] == "2.1.1" and T.config_hash(a) != T.config_hash(b))
    check("I4: a default config records some CLI version string", isinstance(T.measurement_config("m")["cli_version"], str))


def test_review_i5_stale_filing():
    import gg_stage5 as S
    idx = _index([("000001", "10-K", "2012-03-01", "old"), ("000002", "10-K", "2019-12-01", "new")])
    sc = _scores([("old", 4.0, "ok"), ("new", 2.0, "ok")])
    ev = pd.DataFrame({"gvkey": ["000001", "000002"], "dep_ann": pd.to_datetime(["2020-06-01", "2020-06-01"])})
    out = S.attach_T(ev, sc, idx, _META, date_col="dep_ann")
    check("I5: a filing older than the tolerance is stale: NaN with its own reason",
          np.isnan(out.loc[0, "T"]) and out.loc[0, "T_missing_reason"] == "stale_filing", str(out.loc[0, "T_missing_reason"]))
    check("I5: a recent filing is used and its age reported", out.loc[1, "T"] == 2.0 and 150 < out.loc[1, "T_filing_age_days"] < 220)
    loose = S.attach_T(ev, sc, idx, _META, date_col="dep_ann", max_age_days=None)
    check("I5: tolerance can be switched off", loose.loc[0, "T"] == 4.0)
    cov = S.coverage_report(out, "dep_ann")
    check("I5: coverage report counts stale filings and gives the age distribution", cov["reasons"].get("stale_filing") == 1 and "filing_age_days" in cov)


def test_review_i6_audit_disjoint():
    import gg_transparency as T
    df = _scored()
    with tempfile.TemporaryDirectory() as tmp:
        T.make_audit_sheet(df, Path(tmp) / "rep.csv", n=60, seed=0, design="representative")
        rep = set(pd.read_csv(Path(tmp) / "rep.csv")["unit_id"])
        T.make_audit_sheet(df, Path(tmp) / "diag.csv", n=30, seed=0, design="diagnostic", exclude_unit_ids=rep)
        diag = pd.read_csv(Path(tmp) / "diag.csv").merge(df[["unit_id", "sic_div", "t_final"]], on="unit_id")
    check("I6: the diagnostic sample shares no unit with the representative one", not (set(diag["unit_id"]) & rep))
    check("I6: the diagnostic sample spreads across sectors as well as T levels", diag["sic_div"].nunique() >= 3 and diag["t_final"].nunique() >= 5,
          str((diag["sic_div"].nunique(), diag["t_final"].nunique())))


def test_review_i7_input_hashes():
    import gg_stage5 as S
    import gg_transparency as T
    scores = pd.DataFrame({"unit_id": ["a", "b"], "read": [1, 1], "input_sha256": ["h1", "h2"], "config_hash": "c"})
    check("I7: matching input hashes verify", S.verify_score_inputs(scores, {"a": "h1", "b": "h2"}) == [])
    check("I7: a re-extracted text that no longer matches its scored hash is reported", S.verify_score_inputs(scores, {"a": "h1", "b": "zz"}) == ["b"])
    with tempfile.TemporaryDirectory() as tmp:
        p = _setup_runner(tmp, gate={})
        p["current_input_hashes"] = {"a": "changed"}
        p["scores_frame"] = scores
        try:
            S.check_preconditions(p)
            check("I7: the runner refuses scores whose input text changed", False)
        except S.RunRefused as e:
            check("I7: the runner refuses scores whose input text changed", "input" in str(e))


def test_sec_user_agent():
    import gg_filings as F
    old = os.environ.pop("SEC_CONTACT_EMAIL", None)
    try:
        try:
            F.sec_user_agent()
            check("no contact configured raises (SEC requires one)", False)
        except RuntimeError:
            check("no contact configured raises (SEC requires one)", True)
        os.environ["SEC_CONTACT_EMAIL"] = "someone@example.com"
        ua = F.sec_user_agent()
        check("user agent names the project and carries the contact", "someone@example.com" in ua and "guidance-reversal" in ua, ua)
    finally:
        os.environ.pop("SEC_CONTACT_EMAIL", None)
        if old is not None:
            os.environ["SEC_CONTACT_EMAIL"] = old


def test_review_runaway_item1():
    import gg_transparency as T
    # no heading line ends Item 1 (no Item 1A heading): the span runs to the end of the document. Segment text far
    # beyond the 12,000 characters taken from Item 1 must still be found, not skipped as "inside Item 1".
    filler = "Plain business description sentence number one. " * 800
    note = ("We have three reportable segments: Devices, Consumables and Services. Devices revenue was $400 million "
            "on 20,000 units. Segment operating income was $310 million. ") * 3
    html = "<html><body><h2>Item 1. Business</h2><p>" + filler + "</p><p>" + filler + "</p><p>" + note + "</p></body></html>"
    text, st = T.extract_sections(html)
    check("runaway Item 1 span: segment text beyond the first 12,000 characters is kept and the status is ok",
          st == "ok" and "Devices revenue was $400 million" in text, st)
    check("runaway Item 1 span: the segment text sits after the marker", T.SEGMENT_MARKER in text
          and "Devices revenue" in text.split(T.SEGMENT_MARKER)[-1])


def test_review_single_segment_context():
    import gg_transparency as T
    merge = ("In 2013 we combined the European Water and the Asia-Pacific Water reportable segments into one reportable segment "
             "titled International Water. We report four reportable segments: Water, Energy, Applied and International. "
             "Segment operating income was $310 million. ") * 3
    text, st = T.extract_sections(_html(seg_note=merge))
    check("a merger of two segments into 'one reportable segment' does not make a multi-segment firm limited", st == "ok", st)
    own = "We have one operating segment and one reporting unit. Segment information is not presented separately. "
    check("a company statement of one operating segment still gives limited_disclosure",
          T.extract_sections(_html(seg_note=own))[1] == "limited_disclosure")
    named = "Our Retail segment grew 4% on volume. The Wholesale segment shrank on price. Retail prices rose 2% in the segment. " * 3
    text, st = T.extract_sections(_html(seg_note=named))
    check("segments named without an adjective (Retail segment) are found", st == "ok" and "Wholesale segment shrank" in text, st)


def test_rationale():
    import gg_transparency as T
    good = ('{"engines": [{"line": "Consumables", "profit_weight": "high", "price": true, "quantity": true}], "T": 4, '
            '"rationale": "Consumables earn most profit; units and price are disclosed."}')
    r = T.parse_reply(good)
    check("reply with a rationale parses and keeps it", r is not None and r["rationale"].startswith("Consumables earn"))
    check("rationale is optional (older replies still parse)", T.parse_reply(_GOOD) is not None and T.parse_reply(_GOOD)["rationale"] == "")
    check("a non-string rationale is rejected", T.parse_reply(good.replace('"Consumables earn most profit; units and price are disclosed."', "42")) is None)
    long_ = good.replace("Consumables earn most profit; units and price are disclosed.", "x" * 2000)
    check("rationale is truncated to 600 characters", len(T.parse_reply(long_)["rationale"]) == 600)
    check("the rubric asks for a rationale", '"rationale"' in T.RUBRIC)
    cfg = T.measurement_config("model-x", cli_version="v")
    with tempfile.TemporaryDirectory() as tmp:
        out = T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "r.jsonl", runner=_FakeRunner([good, good]))
        check("score_reads keeps the rationale on every row", list(out["rationale"]) == [r["rationale"]] * 2, str(list(out["rationale"])))


def test_review_kpi_windows():
    import gg_transparency as T
    note = ("We have three reportable segments: Devices, Consumables and Services. Segment operating income was $310 million. "
            "Devices revenue was $400 million. ") * 60                                 # a long segment note that would fill the budget
    kpi = ("Operating statistics. Available seat miles (millions) 255,000. Revenue passenger miles (millions) 210,000. Load factor 82.4%. "
           "Passenger mile yield 15.9 cents. Average revenue per car rose 3% while carloads fell 2%. " * 4)
    mdna = "Item 7. MD&A. " + kpi + "General discussion of the year and the economy. " * 700
    html = _html(seg_note=note, mdna=mdna, filler="Business prose about the company. " * 600)
    text, st = T.extract_sections(html)
    check("KPI text far from any 'segment' word is captured (seat miles, load factor, yield)",
          "Available seat miles" in text and "Load factor" in text and "Passenger mile yield" in text, st)
    check("operating statistics beat a long segment note for budget (carloads, revenue per car kept)", "carloads fell" in text)
    check("the cap still holds with both families", len(text) <= 40_000, str(len(text)))
    check("Item 1 body is capped at 8,000 characters", len(text.split(T.SEGMENT_MARKER)[0]) <= 8_100)
    plain, st2 = T.extract_sections(_html(seg_note=_SEG))
    check("a filing with no KPI terms is unaffected (status ok, segment text kept)", st2 == "ok" and "Consumables revenue was $900 million" in plain)
    ps = ("Procedures grew 18%. We placed 1,019 systems. Instruments and accessories revenue per procedure was about $1,800. "
          "Installed base of systems grew 14% to 5,582. " * 3)
    t3, st3 = T.extract_sections(_html(mdna="Item 7. " + "Discussion. " * 2000 + ps))
    check("medical KPIs (procedures, systems placed, per procedure, installed base) are captured", "per procedure" in t3 and "Installed base" in t3, st3)


def test_rubric_derivable():
    import gg_transparency as T
    r = T.RUBRIC
    check("rubric: a derivable price or quantity counts as disclosed", "derive" in r.lower() and "same line and period" in r.lower())
    check("rubric names the materials the excerpt may draw on (10-K, 10-Qs, earnings releases, presentations)",
          "10-q" in r.lower() and "presentation" in r.lower() and "earnings release" in r.lower())
    check("both measurement prompts carry the derivable rule", all("same line and period" in pr.lower() for pr in T.measurement_config("m", cli_version="v")["prompts"]))


# ---------------------------------------------------------------- all company materials: bundle of 10-K + 10-Qs + 8-K exhibits
def _sgml(*docs):
    """A fake EDGAR full-submission text: docs are (type, body) pairs."""
    return "<SEC-DOCUMENT>" + "".join(f"<DOCUMENT><TYPE>{t}<SEQUENCE>1<TEXT>{b}</TEXT></DOCUMENT>" for t, b in docs) + "</SEC-DOCUMENT>"


_KPI_TXT = ("Operating statistics. Available seat miles (millions) 255,000. Load factor 82.4%. Passenger mile yield 15.9 cents. "
            "Average revenue per car rose 3% while carloads fell 2%. " * 3)


def test_split_documents():
    import gg_transparency as T
    raw = _sgml(("10-K", "<html>main</html>"), ("EX-13", "<html>annual report</html>"), ("EX-101.INS", "<xbrl>huge</xbrl>"), ("EX-99.1", "<html>release</html>"))
    docs = T.split_documents(raw)
    check("submission is split into typed documents", [t for t, _ in docs] == ["10-K", "EX-13", "EX-101.INS", "EX-99.1"], str([t for t, _ in docs]))
    check("select_docs keeps only the wanted types (XBRL dropped)", "huge" not in T.select_docs(raw, ("10-K", "EX-13")) and "annual report" in T.select_docs(raw, ("10-K", "EX-13")))
    check("select_docs by prefix picks every EX-99 exhibit", "release" in T.select_docs(raw, ("EX-99",)))
    plain = "<html><body>no sgml wrapper</body></html>"
    check("a plain HTML document without SGML tags is used as is", T.select_docs(plain, ("10-K",)) == plain)
    check("no matching type returns empty", T.select_docs(raw, ("10-Q",)) == "")


def test_extract_supplement():
    import gg_transparency as T
    html = "<html><body><p>" + "Intro text about nothing. " * 400 + _KPI_TXT + "Closing remarks. " * 400 + "</p></body></html>"
    out = T.extract_supplement(html, 6000)
    check("supplement keeps operating statistics from a release", "Available seat miles" in out and "carloads fell" in out)
    check("supplement respects its cap", len(out) <= 6000, str(len(out)))
    check("a document with no operating statistics gives nothing", T.extract_supplement("<html><body>" + "Thanks for coming. " * 300 + "</body></html>", 6000) == "")


def _docidx(rows):
    return pd.DataFrame(rows, columns=["gvkey", "form", "filing_date", "accession"]).assign(
        filing_date=lambda d: pd.to_datetime(d["filing_date"]), cik="1", sic="3600", url=lambda d: "u/" + d["accession"])


def test_bundle_documents():
    import gg_filings as F
    tenk = pd.Series({"gvkey": "000001", "filing_date": pd.Timestamp("2020-02-20"), "accession": "K20", "cik": "1", "url": "u/K20"})
    rows = [("000001", "10-Q", "2019-04-25", "Q1"), ("000001", "10-Q", "2019-07-25", "Q2"), ("000001", "10-Q", "2019-10-24", "Q3"),
            ("000001", "10-Q", "2018-10-25", "Qold"),                                              # older than 365 days: out
            ("000001", "8-K", "2019-10-17", "E3"), ("000001", "8-K", "2020-02-12", "E4"), ("000001", "8-K", "2019-07-18", "E2"),
            ("000001", "8-K", "2019-09-02", "RANDOM"),                                              # not near any filing: out
            ("000001", "8-K", "2020-03-01", "AFTER"), ("000001", "10-Q", "2020-04-24", "QAFTER"),   # filed after the 10-K: never used
            ("000002", "10-Q", "2019-10-24", "OTHERFIRM")]
    idx = _docidx(rows)
    qs, earn = F.bundle_documents(idx, tenk, max_10q=2, max_8k=4)
    check("bundle takes the latest 10-Qs within 12 months, newest last, capped", list(qs["accession"]) == ["Q2", "Q3"], str(list(qs["accession"])))
    check("bundle takes 8-Ks filed shortly before a 10-Q or the 10-K", list(earn["accession"]) == ["E2", "E3", "E4"], str(list(earn["accession"])))
    check("NO document filed after the 10-K is ever included (no look-ahead)",
          (qs["filing_date"] <= tenk["filing_date"]).all() and (earn["filing_date"] <= tenk["filing_date"]).all()
          and not ({"AFTER", "QAFTER"} & set(qs["accession"]) | {"AFTER", "QAFTER"} & set(earn["accession"])))
    check("other firms' documents are never used", "OTHERFIRM" not in set(qs["accession"]))
    qs2, earn2 = F.bundle_documents(idx, tenk, max_10q=3, max_8k=2)
    check("8-K cap keeps the most recent", list(earn2["accession"]) == ["E3", "E4"], str(list(earn2["accession"])))


def test_load_bundle():
    import gg_filings as F
    import gg_transparency as T
    tenk = pd.Series({"gvkey": "000001", "filing_date": pd.Timestamp("2020-02-20"), "accession": "K20", "cik": "0000000001", "url": "u/K20"})
    idx = _docidx([("000001", "10-Q", "2019-10-24", "Q3"), ("000001", "8-K", "2019-10-17", "E3")])
    raws = {"u/K20": _sgml(("10-K", _html(seg_note=_SEG)), ("EX-101.INS", "<x>junk</x>")),
            "u/Q3": _sgml(("10-Q", "<html><body><p>" + "Acme Corp quarterly text. " * 200 + _KPI_TXT + "</p></body></html>"), ("EX-101.INS", "<x>junk</x>")),
            "u/E3": _sgml(("8-K", "<html>cover</html>"), ("EX-99.1", "<html><body><p>Acme Corp reports results. " + _KPI_TXT + "</p></body></html>"))}
    calls = []

    def fake(url, ua):
        calls.append(url)
        return raws[url]

    with tempfile.TemporaryDirectory() as tmp:
        text, st, used = F.load_bundle(tenk, idx, ["Acme Corp"], Path(tmp), "ua", min_interval=0, fetch=fake)
        check("bundle status comes from the 10-K and lists the documents used", st == "ok" and used == ["Q3", "E3"], f"{st} {used}")
        check("bundle holds the 10-K text, then the other materials after a marker", T.BUNDLE_MARKER in text and "Consumables revenue" in text.split(T.BUNDLE_MARKER)[0])
        other = text.split(T.BUNDLE_MARKER)[-1]
        check("other materials are labelled by type and date", "10-Q filed 2019-10-24" in other and "8-K exhibit filed 2019-10-17" in other, other[:200])
        q_part, _, e_part = other.partition("=== 8-K exhibit")
        check("operating statistics from BOTH the 10-Q and the release are in the bundle", "Available seat miles" in q_part and "Available seat miles" in e_part)
        check("identity is stripped from the other materials too", "Acme Corp" not in other and "[COMPANY]" in other)
        check("XBRL documents are never parsed into the bundle", "junk" not in text)
        n = len(calls)
        F.load_bundle(tenk, idx, ["Acme Corp"], Path(tmp), "ua", min_interval=0, fetch=fake)
        check("second call is served from the cache (no refetch)", len(calls) == n, f"{n} -> {len(calls)}")

    def flaky(url, ua):
        if url == "u/E3":
            raise OSError("down")
        return raws[url]

    with tempfile.TemporaryDirectory() as tmp:
        text, st, used = F.load_bundle(tenk, idx, ["Acme Corp"], Path(tmp), "ua", min_interval=0, fetch=flaky)
        check("a supplementary document that cannot be fetched is skipped, the bundle still builds", st == "ok" and used == ["Q3"], f"{st} {used}")

    def no10k(url, ua):
        if url == "u/K20":
            raise OSError("down")
        return raws[url]

    with tempfile.TemporaryDirectory() as tmp:
        text, st, used = F.load_bundle(tenk, idx, ["Acme Corp"], Path(tmp), "ua", min_interval=0, fetch=no10k)
        check("if the 10-K itself cannot be fetched the bundle is fetch_failed", st == "fetch_failed" and text == "")


def test_document_index_forms():
    import gg_filings as F
    forms = pd.DataFrame({"cik": ["0000000111"] * 4, "form": ["10-Q", "8-K", "10-K", "4"],
                          "fdate": pd.to_datetime(["2019-10-24", "2019-10-17", "2020-02-20", "2020-01-01"]),
                          "accession": ["q", "e", "k", "f4"], "fname": ["edgar/data/111/q.txt", "edgar/data/111/e.txt", "edgar/data/111/k.txt", "x"]})
    link = pd.DataFrame({"gvkey": ["000001"], "cik": ["0000000111"], "link_start_date": pd.to_datetime(["2000-01-01"]), "link_end_date": pd.to_datetime([None])})
    comp = pd.DataFrame({"gvkey": ["000001"], "sic": ["3600"], "cik": ["111"]})
    db = _FakeDB({("wrdssec", "wrds_forms"): forms, ("wrdssec", "wciklink_gvkey"): link, ("comp", "company"): comp})
    d = F.build_filing_index(db, forms=("10-Q", "8-K"))
    check("document index keeps exactly the requested forms", set(d["accession"]) == {"q", "e"}, str(set(d["accession"])))
    k = F.build_filing_index(db)
    check("default index is unchanged (10-K only)", set(k["accession"]) == {"k"})


def test_audit_excerpt_with_bundle():
    import gg_transparency as T
    text = "ITEM1 body" + T.SEGMENT_MARKER + "tenk segment text " * 800 + T.BUNDLE_MARKER + "=== 10-Q ===\nquarterly KPI text " * 300
    df = pd.DataFrame({"unit_id": ["u"], "text": [text], "t_final": [3], "sic_div": ["D"]})
    with tempfile.TemporaryDirectory() as tmp:
        T.make_audit_sheet(df, Path(tmp) / "a.csv", n=1, seed=0, design="representative")
        ex = pd.read_csv(Path(tmp) / "a.csv").loc[0, "excerpt"]
    check("audit excerpt shows both the 10-K segment text and the other materials", "tenk segment text" in ex and "quarterly KPI text" in ex, ex[:60])


def test_review_multi_segment_count():
    import gg_transparency as T
    multi = ("We have four reportable segments: Electric, Gas, Renewables and Other. Segment operating income was $310 million. "
             "Duke Energy Ohio operates as one reportable segment. Duke Energy Indiana has one reportable segment. ") * 3
    text, st = T.extract_sections(_html(seg_note=multi))
    check("a stated count of several segments overrides a subsidiary's 'one reportable segment'", st == "ok", st)
    major = "The Firm is organised into four major reportable business segments. Segment results are shown below. A subsidiary has one reportable segment. " * 3
    check("'four major reportable business segments' counts as several segments", T.extract_sections(_html(seg_note=major))[1] == "ok")
    one = "We operate as one reportable segment. Segment information is not presented separately. " * 3
    check("a single-segment firm with no plural count is still limited_disclosure", T.extract_sections(_html(seg_note=one))[1] == "limited_disclosure")


def test_review_item1_heading_variants():
    import gg_transparency as T
    body = "The company makes products for customers and describes its operations here. " * 60
    zw = "<html><body><h2>Item 1.​ Business</h2><p>" + body + "</p><h2>Item 1A. Risk Factors</h2><p>Risks.</p></body></html>"
    text, st = T.extract_sections(zw)
    check("Item 1 heading with a zero-width character is found", "describes its operations" in text, st)
    split = "<html><body><h2>Item 1.</h2><h2>Busines</h2><h2>s.</h2><p>" + body + "</p><h2>Item 1A. Risk Factors</h2><p>Risks.</p></body></html>"
    text2, st2 = T.extract_sections(split)
    check("Item 1 heading whose word 'Business' is split across cells is found", "describes its operations" in text2, st2)
    toc = ("<html><body><p>Item 1. Business 6 Item 1A. Risk Factors 11</p><h2>PART I</h2><h2>ITEM 1​</h2><h2>Business</h2><p>" + body +
           "</p><h2>ITEM 1A</h2><p>Risks.</p></body></html>")
    text3, st3 = T.extract_sections(toc)
    check("the table-of-contents line is still skipped in favour of the real heading", "describes its operations" in text3, st3)


def test_runner_timeout():
    import gg_transparency as T
    cfg = T.measurement_config("model-x", cli_version="v")
    waits = []
    with tempfile.TemporaryDirectory() as tmp:
        run = _FakeRunner([T.RunnerTimeout("slow"), T.RunnerTimeout("slow"), _REPLY])
        out = T.score_reads(_units(1), cfg=cfg, cache_path=Path(tmp) / "t.jsonl", runner=run, reads=(1,), sleeper=waits.append)
        check("a timed-out call is retried and the read succeeds", out.iloc[0]["score"] == 4 and len(run.prompts) == 3, str(len(run.prompts)))
        check("each timeout backs off before retrying", len(waits) == 2, str(waits))
    with tempfile.TemporaryDirectory() as tmp:
        cp = Path(tmp) / "t2.jsonl"
        always = _FakeRunner([T.RunnerTimeout("slow")] * 10)
        out = T.score_reads(_units(1), cfg=cfg, cache_path=cp, runner=always, reads=(1,), sleeper=lambda s: None, max_timeouts=2)
        check("a read that keeps timing out is NaN and flagged, and the run does not crash", np.isnan(out.iloc[0]["score"]) and "timeout" in out.iloc[0]["flags"])
        check("a timed-out read is NOT cached, so a resume retries it", not cp.exists() or cp.read_text(encoding="utf-8").strip() == "")
        ok = T.score_reads(_units(1), cfg=cfg, cache_path=cp, runner=_FakeRunner(), reads=(1,))
        check("the resume then scores it", ok.iloc[0]["score"] == 4)


def test_bundle_documents_empty_cases():
    import gg_filings as F
    tenk = pd.Series({"gvkey": "000001", "filing_date": pd.Timestamp("2020-02-20"), "accession": "K20", "cik": "1", "url": "u/K20"})
    only_q = _docidx([("000001", "10-Q", "2019-10-24", "Q3")])
    qs, earn = F.bundle_documents(only_q, tenk)
    check("no 8-K in the window: empty result, no crash, columns kept", list(qs["accession"]) == ["Q3"] and len(earn) == 0 and "filing_date" in earn.columns)
    only_k = _docidx([("000001", "8-K", "2020-02-12", "E3"), ("000001", "8-K", "2019-10-17", "TOOEARLY")])
    qs, earn = F.bundle_documents(only_k, tenk)
    check("no 10-Q in the window: the 8-K near the 10-K is still found", len(qs) == 0 and list(earn["accession"]) == ["E3"])
    none = _docidx([("000009", "10-Q", "2019-10-24", "OTHER")])
    qs, earn = F.bundle_documents(none, tenk)
    check("no documents at all for the firm: both empty", len(qs) == 0 and len(earn) == 0)
    qs, earn = F.bundle_documents(_docidx([("000001", "10-Q", "2019-10-24", "Q3")]).iloc[0:0], tenk)
    check("an entirely empty document index: both empty", len(qs) == 0 and len(earn) == 0)


# ---------------------------------------------------------------- phases: has the revenue split changed? (computational check + small triage)
def _rev_html(labels, year="2019", note=""):
    rows = "".join(f"<tr><td>{l}</td><td>$ {1000 + 37 * i:,}</td><td>$ {900 + 29 * i:,}</td></tr>" for i, l in enumerate(labels))
    return ("<html><body><h2>Note 12. Segment Information</h2><p>" + note + "Net sales by segment for the years ended December 31.</p>"
            f"<table><tr><td>Segment</td><td>{year}</td><td>{int(year) - 1}</td></tr>" + rows +
            "<tr><td>Total</td><td>$ 9,999</td><td>$ 8,888</td></tr><tr><td>Net income</td><td>$ 111</td><td>$ 99</td></tr></table></body></html>")


def test_revenue_labels():
    import gg_phases as P
    labs = P.revenue_labels(_rev_html(["Consumer Products", "Industrial", "Health Care", "Safety and Graphics"]))
    check("table row labels under a revenue-by-segment heading are extracted", {"consumer products", "industrial", "health care", "safety and graphics"} <= set(labs), str(labs))
    check("totals, net income and header rows are ignored", not any(x in labs for x in ("total", "net income", "segment")), str(labs))
    sent = P.revenue_labels("<html><body><p>We have three reportable segments: Devices, Consumables and Services. Revenue by segment is shown below.</p></body></html>")
    check("a sentence listing the reportable segments is parsed", {"devices", "consumables", "services"} <= set(sent), str(sent))
    check("a filing with no revenue breakdown gives an empty list", P.revenue_labels("<html><body><p>Nothing about the business here.</p></body></html>") == [])


def test_compare_labels():
    import gg_phases as P
    base = ["consumer products", "industrial", "health care", "safety and graphics"]
    same = P.compare_labels(base, list(base))
    check("identical splits: level NONE, similarity 1", same["level"] == "NONE" and same["similarity"] == 1.0, str(same))
    renamed = P.compare_labels(base, ["consumer", "industrial", "health care", "safety and graphics"])
    check("one segment renamed (fuzzy match): not MAJOR", renamed["level"] in ("NONE", "MINOR"), str(renamed))
    added = P.compare_labels(base, base + ["energy"])
    check("one segment added to four: MINOR, not MAJOR", added["level"] == "MINOR", str(added))
    swapped = P.compare_labels(base, ["devices", "consumables", "services", "retail"])
    check("a completely different split is MAJOR", swapped["level"] == "MAJOR" and swapped["similarity"] < 0.3, str(swapped))
    collapsed = P.compare_labels(base, ["total company"])
    check("four segments collapsing to one is MAJOR", collapsed["level"] == "MAJOR", str(collapsed))
    merged = P.compare_labels(base + ["international"], ["consumer products", "industrial", "health care"])
    check("two of five segments dropped is at least MINOR", merged["level"] in ("MINOR", "MAJOR"), str(merged))
    unk = P.compare_labels([], base)
    check("no labels on one side: UNKNOWN (never an automatic break)", unk["level"] == "UNKNOWN", str(unk))


def test_triage_change():
    import gg_phases as P
    prompts = []

    def runner(prompt):
        prompts.append(prompt)
        return '{"major": true, "reason": "four product segments replaced by two regional ones"}'

    out = P.triage_change(["consumer", "industrial", "health care"], ["north america", "europe"], runner)
    check("triage returns the verdict and reason", out["major"] is True and "regional" in out["reason"], str(out))
    check("the triage prompt is small (just the two label lists)", len(prompts[0]) < 3000, str(len(prompts[0])))
    bad = P.triage_change(["a", "b"], ["c", "d"], lambda p: "not sure")
    check("an unparseable triage answer is 'unknown', not a break", bad["major"] is None, str(bad))
    calls = []
    lv = ["START", "NONE", "MINOR", "UNKNOWN", "MAJOR"]          # level of the change from the previous filing; the first filing has none
    phases = P.assign_phases(lv, triage=lambda i: calls.append(i) or True)
    check("triage is called only for MAJOR changes", calls == [4], str(calls))
    check("phases increase only at a confirmed MAJOR change", phases == [0, 0, 0, 0, 1], str(phases))
    phases2 = P.assign_phases(lv, triage=lambda i: False)
    check("a MAJOR change the triage rejects does not start a phase", phases2 == [0, 0, 0, 0, 0], str(phases2))
    phases3 = P.assign_phases(["START", "MAJOR", "NONE", "MAJOR", "NONE"], triage=lambda i: True)
    check("two confirmed changes give three phases", phases3 == [0, 1, 1, 2, 2], str(phases3))


def _tbl(rows, header=None):
    h = "" if header is None else "<tr>" + "".join(f"<td>{c}</td>" for c in header) + "</tr>"
    return "<table>" + h + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


def test_revenue_table_labels():
    import time
    import gg_phases as P
    rev = _tbl([["Consumer Products", "$ 1,200", "$ 1,100"], ["Industrial", "$ 2,300", "$ 2,250"], ["Health Care", "$ 900", "$ 870"],
                ["Total", "$ 4,400", "$ 4,220"]], header=["", "2019", "2018"])
    page = "<html><body><p>Net sales by segment for the years ended December 31 (in millions).</p>" + rev + "</body></html>"
    labs = P.revenue_labels(page)
    check("rows of a table captioned 'net sales by segment' are the labels", set(labs) == {"consumer products", "industrial", "health care"}, str(labs))
    bs = "<html><body><p>Consolidated balance sheet</p>" + _tbl([["Cash and equivalents", "$ 500", "$ 400"], ["Accounts receivable", "$ 700", "$ 650"],
                                                                 ["Inventories", "$ 300", "$ 290"]], header=["", "2019", "2018"]) + "</body></html>"
    check("a balance-sheet table is ignored", P.revenue_labels(bs) == [], str(P.revenue_labels(bs)))
    execs = ("<html><body><p>Executive officers of the registrant. Segment leaders are listed below.</p>" +
             _tbl([["Franklin H. Yoho", "Chief Executive Officer", "58"], ["Lynn J. Good", "Chairman", "60"]]) + "</body></html>")
    check("a table of names (no numbers) is ignored even near the word segment", P.revenue_labels(execs) == [], str(P.revenue_labels(execs)))
    prose = "<html><body><p>Net sales by segment increased compared to the prior year. Franklin H Yoho said results were higher.</p></body></html>"
    check("prose near a revenue heading yields no labels", P.revenue_labels(prose) == [], str(P.revenue_labels(prose)))
    prod = ("<html><body><h3>Revenue by product line</h3>" + _tbl([["Instruments and accessories", "1,100", "950"], ["Systems", "900", "880"], ["Services", "400", "350"]],
                                                                 header=["", "2019", "2018"]) + "</body></html>")
    check("a revenue-by-product-line table gives its lines", set(P.revenue_labels(prod)) == {"instruments and accessories", "systems", "services"}, str(P.revenue_labels(prod)))
    cap = ("<html><body><table><caption>Net revenues by geographic region</caption><tr><td>United States</td><td>5,000</td><td>4,800</td></tr>"
           "<tr><td>Europe</td><td>2,000</td><td>1,900</td></tr></table></body></html>")
    check("a <caption> naming revenue by region is used", set(P.revenue_labels(cap)) == {"united states", "europe"}, str(P.revenue_labels(cap)))
    big = "<html><body>" + "".join(f"<p>Filler paragraph {i} about the business and its markets.</p>" + _tbl([[f"Row {i}", "1", "2"]]) for i in range(4000)) + page[12:]
    t0 = time.time()
    P.revenue_labels(big)
    check("a document with 4,000 tables is parsed in under 5 seconds", time.time() - t0 < 5, f"{time.time() - t0:.1f}s")


def test_compare_noise_and_stop():
    import gg_phases as P
    base = ["consumer products", "industrial", "health care", "safety and graphics", "electronics", "transportation", "energy", "other", "retail", "wholesale"]
    noisy_add = base + [f"balance item {i}" for i in range(16)]
    c = P.compare_labels(base, noisy_add)
    check("a table appearing (16 extra rows) while every old label remains is NOT MAJOR", c["level"] != "MAJOR", str(c))
    c = P.compare_labels(noisy_add, base)
    check("a table disappearing while every remaining label matches is NOT MAJOR", c["level"] != "MAJOR", str(c))
    c = P.compare_labels(base, [f"new line {i}" for i in range(10)])
    check("old labels gone AND new ones appear is MAJOR", c["level"] == "MAJOR", str(c))
    c = P.compare_labels(base[:5], base[:3] + ["x line", "y line"])
    check("a partial overlap (3 of 5 kept) is MINOR, not MAJOR", c["level"] == "MINOR", str(c))
    import gg_phases as G
    for junk in ("first quarter", "fourth quarter", "accounts receivable", "raw materials", "work in process", "allowance for doubtful accounts",
                 "unbilled current", "deferred income taxes", "intangible asset impairment charges", "restructuring charges", "property plant and equipment",
                 "research and development", "selling general and administrative"):
        assert not G._good(G._norm(junk)), junk
    check("quarters, balance-sheet and expense rows are not revenue lines", True)
    for ok in ("consumer products", "united states", "domestic streaming", "instruments and accessories", "clear aligner", "personal lines"):
        assert G._good(G._norm(ok)), ok
    check("genuine revenue lines are kept", True)


def test_audit_workbook():
    import gg_audit_book as AB
    import gg_transparency as T
    import openpyxl
    NL = chr(10)
    txt = lambda k: ("Business text for filing %d. We make products & services <b>." % k + NL + T.SEGMENT_MARKER +
                     "Available seat miles 255,000 and load factor 82%% in filing %d." % k + NL + T.BUNDLE_MARKER +
                     "=== 10-Q filed 2019-10-24 ===" + NL + "Subscribers grew 5%% in quarter %d." % k)
    rep = pd.DataFrame({"unit_id": [f"ACC-{i}" for i in range(5)], "text": [txt(i) for i in range(5)]})
    diag = pd.DataFrame({"unit_id": [f"DIA-{i}" for i in range(3)], "text": [txt(10 + i) for i in range(3)]})
    with tempfile.TemporaryDirectory() as tmp:
        path = AB.build_audit_workbook({"Representative": ("rep", rep), "Diagnostic": ("diag", diag)}, Path(tmp))
        wb = openpyxl.load_workbook(path)
        check("workbook has a guide tab and one tab per sheet", wb.sheetnames == ["Guide", "Representative", "Diagnostic"], str(wb.sheetnames))
        ws = wb["Representative"]
        hdr = {c.value: c.column for c in ws[4] if c.value}
        check("columns: row, open link, YOUR SCORE, note, done, hidden id", {"Row", "OPEN (click)", "YOUR SCORE 0-5", "Note (optional)", "Done"} <= set(hdr), str(list(hdr)))
        r1 = 5
        link = ws.cell(r1, hdr["OPEN (click)"]).hyperlink
        check("each row has one hyperlink to its own reader page", link is not None and link.target.endswith("rep_001.html") and (Path(tmp) / link.target).exists(), str(link and link.target))
        check("the score cell is empty and has a 0-5 whole-number validation",
              ws.cell(r1, hdr["YOUR SCORE 0-5"]).value is None and any("0" in str(dv.formula1) or dv.type == "whole" for dv in ws.data_validations.dataValidation))
        check("a hidden column keeps the filing id for mapping back", ws.cell(r1, hdr["id (hidden)"]).value == "ACC-0" and ws.column_dimensions[openpyxl.utils.get_column_letter(hdr["id (hidden)"])].hidden)
        check("the diagnostic tab links to diag pages", wb["Diagnostic"].cell(5, hdr["OPEN (click)"]).hyperlink.target.endswith("diag_001.html"))
        html = (Path(tmp) / "reader" / "rep_002.html").read_text(encoding="utf-8")
        check("reader page shows the row number and the model's text", "Row 2 of 5" in html and "filing 1" in html)
        check("reader page never shows the filing id (blind)", "ACC-1" not in html and "ACC-" not in html)
        check("reader page escapes text from the filing", "&lt;b&gt;" in html and "<b>." not in html)
        check("operating-statistics terms are highlighted", "<mark>Available seat miles</mark>" in html or "<mark>available seat miles</mark>" in html.lower())
        check("reader page separates the three sections", html.count("<h2") >= 3 and "10-Q filed 2019-10-24" in html)
        check("reader page carries the scoring cheat sheet", "Dollars only" in html and "razor" in html.lower())
        check("reader pages link to the previous and next row", "rep_001.html" in html and "rep_003.html" in html)
        first = (Path(tmp) / "reader" / "rep_001.html").read_text(encoding="utf-8")
        last = (Path(tmp) / "reader" / "rep_005.html").read_text(encoding="utf-8")
        check("the first page has no previous link and the last no next", "rep_000.html" not in first and "rep_006.html" not in last)


def test_phase_runner():
    import json
    import gg_phase_run as R
    table = ("<p>Net sales by segment</p><table><tr><td>Widgets</td><td>$ 100</td><td>90</td></tr>"
             "<tr><td>Gadgets</td><td>$ 50</td><td>40</td></tr><tr><td>Total</td><td>150</td><td>130</td></tr></table>")
    rows = pd.DataFrame({"gvkey": ["000001", "000001", "000002"], "accession": ["A1", "A2", "B1"], "filing_date": ["2016-03-01", "2017-03-01", "2016-03-05"],
                         "cik": [1, 1, 2], "url": ["u1", "u2", "u3"]})
    calls = []

    def fetch(url, ua):
        calls.append(url)
        if url == "u3":
            raise RuntimeError("503")
        return "<html><body>" + table + "</body></html>"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "labels.jsonl"
        s1 = R.run_labels(rows, out, fetch, "ua", workers=2, min_interval=0.0)
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        done = {r["accession"]: r for r in recs}
        check("successful filings are stored with their labels", set(done) == {"A1", "A2"} and "widgets" in done["A1"]["labels"], str(done))
        check("a failed download is not stored as done", "B1" not in done and s1["failed"] == 1 and s1["done"] == 2)
        check("no raw HTML is kept in the output", "<table" not in out.read_text(encoding="utf-8"))
        n = len(calls)
        s2 = R.run_labels(rows, out, fetch, "ua", workers=2, min_interval=0.0)
        check("a rerun skips finished filings and retries only the failed one", len(calls) == n + 1 and calls[-1] == "u3" and s2["skipped"] == 2)
        lv = R.change_levels(done_df := pd.DataFrame(list(done.values())))
        check("consecutive filings of a firm get a change level; the first has none", lv.loc[lv["accession"] == "A1", "level"].iloc[0] == "FIRST" and lv.loc[lv["accession"] == "A2", "level"].iloc[0] == "NONE", str(lv))


def test_revenue_labels_unclosed_caption():
    import gg_phases as P
    html = ("<table><caption>Net sales by segment<tr><td>Widgets</td><td>100</td><td>90</td></tr>"
            "<tr><td>Gadgets</td><td>50</td><td>40</td></tr></table>")
    try:
        labs = P.revenue_labels(html)
        ok = True
    except Exception as e:
        labs, ok = str(e), False
    check("a caption tag with no closing tag does not crash the reader", ok, str(labs))
    check("the reader returns a list for such a table", ok and isinstance(labs, list), str(labs))


def test_triage_runner():
    import json
    import gg_phase_run as R
    import gg_transparency as T
    pairs = pd.DataFrame({"gvkey": ["1", "2", "3"], "accession": ["a", "b", "c"], "filing_date": ["2016-01-01"] * 3,
                          "prev": [["x"], ["y"], ["z"]], "cur": [["p"], ["q"], ["r"]]})
    calls, state = [], {"limit": 1}

    def runner(prompt):
        calls.append(prompt)
        if "PREVIOUS YEAR: y" in prompt:
            raise T.RunnerTimeout("slow")
        if "PREVIOUS YEAR: z" in prompt and state["limit"]:
            state["limit"] = 0
            raise T.UsageLimit("limit reached")
        return '{"major": true, "reason": "segments replaced"}' if "PREVIOUS YEAR: x" in prompt else '{"major": false, "reason": "rename"}'
    waits = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "triage.jsonl"
        s = R.run_triage(pairs, out, runner, workers=1, sleeper=waits.append)
        got = {json.loads(l)["accession"]: json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()}
        check("answers are stored per filing with the reason", got["a"]["major"] is True and got["c"]["major"] is False and got["a"]["reason"] == "segments replaced", str(got))
        check("a usage limit waits and then retries the same pair", waits and "c" in got)
        check("a timed-out pair is not stored, so a rerun retries it", "b" not in got and s["failed"] == 1)
        n = len(calls)
        R.run_triage(pairs, out, runner, workers=1, sleeper=waits.append)
        check("a rerun asks only about the pair that has no answer", len(calls) == n + 1)
        check("stored answers feed phase_table as a dict", R.load_triage(out) == {"a": True, "c": False})


def test_choose_reads():
    import gg_phase_run as R
    reads = pd.DataFrame({"unit_id": ["u1", "u1", "u2", "u3", "u4"], "score": [5.0, 3.0, 4.0, 2.0, np.nan],
                          "input_sha256": ["old", "cur1", "old2", "c3", "c4"], "config_hash": "H"})
    cur = {"u1": "cur1", "u2": "cur2", "u3": "c3", "u4": "c4", "u5": "c5", "u6": "c6"}
    status = {"u1": "ok", "u2": "ok", "u3": "fetch_failed", "u4": "ok", "u5": "ok", "u6": "limited_disclosure"}
    out = R.choose_reads(reads, cur, status, "H").set_index("accession")
    check("the read made on the current text wins, not the highest score", out.loc["u1", "t_final"] == 3.0 and out.loc["u1", "input_sha256"] == "cur1")
    check("a read made on older text is not used", np.isnan(out.loc["u2", "t_final"]) and out.loc["u2", "status"] == "stale_input")
    check("a failed bundle keeps no score even when a read exists", np.isnan(out.loc["u3", "t_final"]) and out.loc["u3", "status"] == "fetch_failed")
    check("an unparseable read stays NaN with its status", np.isnan(out.loc["u4", "t_final"]) and out.loc["u4", "status"] == "ok")
    check("a unit that was never read is not_scored", np.isnan(out.loc["u5", "t_final"]) and out.loc["u6", "status"] == "not_scored")
    check("every unit with a status appears once", len(out) == 6 and out.index.is_unique)
    check("every row carries the config hash", set(out["config_hash"]) == {"H"})


def test_current_hashes_file():
    import gg_stage5 as S
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "current_hashes.jsonl"
        check("a missing hash file gives None", S.load_current_hashes(p) is None)
        p.write_text('{"unit_id": "u1", "sha": "a"}\n{"unit_id": "u1", "sha": "b"}\n{"unit_id": "u2", "sha": "c"}\n', encoding="utf-8")
        check("the latest hash per unit is returned", S.load_current_hashes(p) == {"u1": "b", "u2": "c"})


def test_scores_from_phases():
    import gg_phase_run as R
    pt = pd.DataFrame({"gvkey": ["1", "1", "1", "2"], "accession": ["a1", "a2", "a3", "b1"], "phase": [0, 0, 1, 0], "phase_first": ["a1", "a1", "a3", "b1"]})
    ps = pd.DataFrame({"accession": ["a1", "a3", "b1"], "t_final": [2.0, 4.0, np.nan], "status": ["ok", "limited_disclosure", "extraction_failed"],
                       "config_hash": "H", "input_sha256": ["s1", "s3", "sb"]})
    out = R.scores_from_phases(pt, ps)
    d = out.set_index("accession")
    check("every filing takes the score of its phase's first filing", d.loc["a2", "t_final"] == 2.0 and d.loc["a3", "t_final"] == 4.0 and d.loc["a1", "t_final"] == 2.0)
    check("a phase whose first filing failed stays unscored, with that status", np.isnan(d.loc["b1", "t_final"]) and d.loc["b1", "status"] == "extraction_failed")
    check("each row names the scored unit and carries the config hash", d.loc["a2", "unit_id"] == "a1" and set(out["config_hash"]) == {"H"} and len(out) == 4)
    pt2 = pt.assign(phase_first=["a1", "a1", "a3", "zz"])
    d2 = R.scores_from_phases(pt2, ps).set_index("accession")
    check("a phase that was never scored is not_scored, not zero", np.isnan(d2.loc["b1", "t_final"]) and d2.loc["b1", "status"] == "not_scored")


def test_phase_table():
    import gg_phase_run as R
    L1, L2 = ["widgets", "gadgets", "services"], ["cloud", "hardware", "licensing"]
    df = pd.DataFrame({"gvkey": ["1"] * 4 + ["2"] * 2, "accession": ["a1", "a2", "a3", "a4", "b1", "b2"],
                       "filing_date": ["2015-03-01", "2016-03-01", "2017-03-01", "2018-03-01", "2016-03-01", "2017-03-01"],
                       "labels": [L1, L1, L2, L2, L1, L1]})
    pairs = R.triage_pairs(df)
    check("only MAJOR changes are listed for triage, with both label lists", list(pairs["accession"]) == ["a3"] and pairs.iloc[0]["prev"] == L1 and pairs.iloc[0]["cur"] == L2, str(pairs))
    pt = R.phase_table(df, {"a3": True})
    ph = dict(zip(pt["accession"], pt["phase"]))
    check("a confirmed MAJOR change starts a new phase for that firm only", ph == {"a1": 0, "a2": 0, "a3": 1, "a4": 1, "b1": 0, "b2": 0}, str(ph))
    pt2 = R.phase_table(df, {"a3": False})
    check("a rejected MAJOR change does not split the phase", set(pt2["phase"]) == {0})
    pt3 = R.phase_table(df, {"a3": None})
    check("an unusable triage answer keeps the phase together and is flagged", set(pt3["phase"]) == {0} and bool(pt3.loc[pt3["accession"] == "a3", "triage_unusable"].iloc[0]))
    check("each phase names the filing it is scored from (its first)", dict(zip(pt["accession"], pt["phase_first"])) == {"a1": "a1", "a2": "a1", "a3": "a3", "a4": "a3", "b1": "b1", "b2": "b1"})
    check("a MAJOR change with no triage answer yet is reported as pending", R.phase_table(df, {}).loc[lambda d: d["accession"] == "a3", "triage_pending"].iloc[0])


def main():
    test_root_override()
    test_filings()
    test_links()
    test_build_index()
    test_extraction()
    test_strip_identity()
    test_load_sections_cache()
    test_parse_score()
    test_parse_reply()
    test_rule_check()
    test_rubric()
    test_config_hash()
    test_combine_reads()
    test_make_tasks()
    test_kappa()
    test_bunching_and_groups()
    test_viability()
    test_gate()
    test_freeze()
    test_motivating_env()
    test_select_units()
    test_disjoint()
    test_audit_sheet()
    test_leakage_and_tokens()
    test_measurement_config()
    test_score_reads()
    test_attach()
    test_attach_matches_lookup()
    test_standardisation_and_coverage()
    test_validation_report()
    test_t1_core()
    test_t1_pool_requires_announcement()
    test_event_T_uses_window_start_date()
    test_t1_reasons()
    test_t1_rejection_rates()
    test_t1_continuous()
    test_p1_T()
    test_p2_T()
    test_permute_T_blocks()
    test_t4_core()
    test_t4_rejection_rates()
    test_runner()
    test_p3_refactor_static()
    test_review_c1_viability_deferred()
    test_review_c2_audit_excerpt()
    test_review_i1_tickers()
    test_review_i2_item1_crossref()
    test_review_i3_runner_secondary()
    test_review_i4_cli_version()
    test_review_i5_stale_filing()
    test_review_i6_audit_disjoint()
    test_review_i7_input_hashes()
    test_sec_user_agent()
    test_review_runaway_item1()
    test_review_single_segment_context()
    test_rationale()
    test_runner_timeout()
    test_review_kpi_windows()
    test_rubric_derivable()
    test_review_multi_segment_count()
    test_review_item1_heading_variants()
    test_split_documents()
    test_extract_supplement()
    test_bundle_documents()
    test_bundle_documents_empty_cases()
    test_revenue_labels()
    test_revenue_table_labels()
    test_compare_labels()
    test_compare_noise_and_stop()
    test_audit_workbook()
    test_phase_runner()
    test_revenue_labels_unclosed_caption()
    test_triage_runner()
    test_choose_reads()
    test_current_hashes_file()
    test_scores_from_phases()
    test_phase_table()
    test_triage_change()
    test_load_bundle()
    test_document_index_forms()
    test_audit_excerpt_with_bundle()
    print(f"\n{len(FAILS)} failures {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
