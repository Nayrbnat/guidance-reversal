"""
gg_transparency: the transparency score T for Stage 5 (docs/STAGE5_SPEC.md): section extraction from a 10-K,
identity stripping, rubric, reliability statistics, gate and freeze guard.

T is an integer 0-5 per filing; higher means the filing lets an outsider rebuild more of the firm's profit engine
as price x quantity. This module holds everything that can be tested without a model.
"""
from __future__ import annotations

import re

EXTRACTION_VERSION = "ex8"
BUNDLE_MARKER = "\n\n=== OTHER COMPANY MATERIALS (10-Qs, earnings releases and presentations filed before the 10-K) ===\n\n"
SEGMENT_MARKER = "\n\n=== SEGMENT AND KPI TEXT ===\n\n"   # separates Item 1 from the segment windows in the extracted text
MAX_CHARS = 40_000
ITEM1_CHARS = 8_000
IDENTITY_VERSION = "id1"

_ITEM1 = re.compile(r"item\s*1\s*[\.\-–—:]?\s*(?:description\s+of\s+)?busines\s*s", re.I)
_ITEM1_END = re.compile(r"^\s*(?:item\s*1a\b|item\s*1b\b|item\s*2\b)", re.I | re.M)   # a heading line, not an in-text cross-reference
_SEGMENT_ADJ = re.compile(r"(?:reportable|operating|business|geographic)\s+segments?|"
                      r"segment\s+(?:information|reporting|results|operating|revenues?|profit|income)", re.I)
_SEGMENT_NAMED = re.compile(r"(?<![A-Za-z])[A-Z][A-Za-z&-]+\s+[Ss]egments?\b")   # a capitalised segment name, e.g. "Retail segment"
_MERGE_CONTEXT = re.compile(r"\b(?:into|combin\w*|merg\w*|reorganiz\w*|consolidat\w*)\b", re.I)
_SINGLE = re.compile(r"(?:one|single|sole)\s+(?:reportable|operating|business)\s+segment|"
                     r"operat\w*\s+(?:in|as)\s+(?:one|a\s+single)\s+(?:(?:business|operating|reportable)\s+)?segment", re.I)
_MULTI = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+(?:[a-z-]+\s+){0,2}(?:reportable|operating|business)\s+segments\b", re.I)
_WIN_BEFORE, _WIN_AFTER = 1_500, 3_500
_KPI = re.compile(
    r"available\s+seat\s+miles?|revenue\s+passenger\s+miles?|load\s+factor|\byield\b|carloads?|"
    r"revenue\s+per\s+(?:car|unit|user|member|membership|subscriber|room|store|ton|mile|passenger|customer|share|procedure|transaction)|"
    r"average\s+(?:selling\s+price|revenue\s+per|sales\s+price|realized\s+price|price\s+per)|\bARPU\b|\bARPA\b|"
    r"(?:same|comparable)[-\s]+(?:store\s+)?sales|units\s+sold|unit\s+(?:sales|volume|shipments)|shipments|subscribers?|"
    r"paid\s+(?:net\s+)?memberships?|per\s+procedure|procedure\s+(?:volume|growth)|procedures\s+(?:grew|increased|performed)|"
    r"systems\s+(?:placed|sold|shipped)|installed\s+base|price\s*/\s*volume|volume\s*/\s*(?:price|mix)|price\s+and\s+volume|volume\s+and\s+price|"
    r"policies\s+in\s+force|net\s+interest\s+margin|average\s+(?:loans|deposits)|kilowatt[-\s]?hours?|megawatt[-\s]?hours?|\bMWh\b|\bGWh\b|"
    r"barrels|ounces|tons\s+(?:shipped|sold)|billable\s+(?:hours|days)|customer\s+accounts|square\s+feet|average\s+daily\s+rate|\bRevPAR\b|"
    r"net\s+new\s+(?:subscribers|customers|members)|members\s+at|connections|churn", re.I)   # operating statistics: price x quantity drivers
_KPI_BEFORE, _KPI_AFTER = 600, 2_400


def _to_text(html: str) -> str:
    if not html or not html.strip():
        return ""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    text = soup.get_text("\n")
    text = re.sub("[\u200b\u200c\u200d\u2060\ufeff]", "", text)          # zero-width characters break headings such as "Item 1.\u200b Business"
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _item1_span(text: str):
    """(start, end) of the Item 1 body: the 'Item 1. Business' heading followed by the longest text before the next
    item heading, which skips the table of contents. None when no real Item 1 is found."""
    best = None
    for m in _ITEM1.finditer(text):
        nxt = _ITEM1_END.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        if end - m.start() >= 300 and (best is None or end - m.start() > best[1] - best[0]):
            best = (m.start(), end)
    return best


def _segment_matches(text: str):
    """Positions of segment mentions: adjective forms (reportable/operating/business/geographic segment, segment
    information ...) and capitalised segment names ('Retail segment'). Returned as objects with .start()."""
    ms = list(_SEGMENT_ADJ.finditer(text)) + list(_SEGMENT_NAMED.finditer(text))
    return sorted(ms, key=lambda m: m.start())


def _single_statement(text: str) -> bool:
    """True when the filing states that the company itself has one segment. Ignored: a merger of segments ('combined X and
    Y into one reportable segment'), and any 'one reportable segment' in a filing that also states a count of several
    reportable, operating or business segments ('four major reportable business segments'): that sentence is about a
    subsidiary registrant or a part of the business, not the filer."""
    if _MULTI.search(text):
        return False
    for m in _SINGLE.finditer(text):
        if not _MERGE_CONTEXT.search(text[max(0, m.start() - 60):m.start()]):
            return True
    return False


def _segment_windows(text: str, skip: tuple | None):
    """Merged windows around segment mentions (outside the Item 1 slice), as (start, end, n_mentions)."""
    spans = []
    for m in _segment_matches(text):
        if skip and skip[0] <= m.start() < skip[1]:
            continue
        spans.append((max(0, m.start() - _WIN_BEFORE), min(len(text), m.start() + _WIN_AFTER)))
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] += 1
        else:
            merged.append([a, b, 1])
    return [tuple(w) for w in merged]


def _kpi_windows(text: str, skip):
    """Merged windows around operating-statistics terms (seat miles, load factor, carloads, revenue per car, procedures,
    subscribers ...) outside the Item 1 body already taken, as (start, end, n_hits). These tables sit in MD&A far from the
    word 'segment', so segment windows alone would miss exactly the price x quantity data the rubric asks about."""
    merged = []
    for m in _KPI.finditer(text):
        if skip and skip[0] <= m.start() < skip[1]:
            continue
        a, b = max(0, m.start() - _KPI_BEFORE), min(len(text), m.start() + _KPI_AFTER)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] += 1
        else:
            merged.append([a, b, 1])
    return [tuple(w) for w in merged]


def split_documents(raw: str) -> list:
    """Split an EDGAR full-submission text into (TYPE, body) pairs, so the XBRL and graphics documents (tens of megabytes
    that carry no prose) can be skipped before any HTML parsing. A page without SGML wrappers is one ('MAIN', raw) document."""
    out = []
    for blk in re.split(r"<DOCUMENT>", raw, flags=re.I)[1:]:
        m = re.search(r"<TYPE>\s*([^\s<]+)", blk, flags=re.I)
        t = re.search(r"<TEXT>(.*?)(?:</TEXT>|$)", blk, flags=re.I | re.S)
        if m and t:
            out.append((m.group(1).upper(), t.group(1)))
    return out or [("MAIN", raw)]


def select_docs(raw: str, prefixes) -> str:
    """The bodies of the documents whose TYPE starts with one of `prefixes`, joined; a page with no SGML wrapper is
    returned as is; '' when nothing matches."""
    docs = split_documents(raw)
    if len(docs) == 1 and docs[0][0] == "MAIN":
        return docs[0][1]
    return "\n".join(b for t, b in docs if any(t.startswith(p.upper()) for p in prefixes))


def extract_supplement(html: str, max_chars: int) -> str:
    """Operating-statistics and segment windows from a 10-Q or an earnings release / presentation exhibit, capped at
    max_chars; '' when the document has none (a cover page, a thank-you slide)."""
    text = _to_text(html)
    if not text:
        return ""
    wins = _kpi_windows(text, None) + _segment_windows(text, None)
    spans = _union(_pick_windows(wins, max_chars))
    seg = "\n\n".join(text[a:b] for a, b in spans)
    return seg[:max_chars]


def _pick_windows(wins, budget: int):
    """Densest windows first until `budget` characters are used; the last one is cut to fit. Returns [(start, end)]."""
    out = []
    for a, b, _ in sorted(wins, key=lambda w: (-w[2], w[0])):
        if budget <= 0:
            break
        b = min(b, a + budget)
        out.append((a, b))
        budget -= b - a
    return out


def _union(spans):
    """Merge overlapping (start, end) spans in document order."""
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def extract_sections(html: str, max_chars: int = MAX_CHARS) -> tuple[str, str]:
    """(text, status) from a 10-K's HTML: Item 1 (first ITEM1_CHARS characters), then SEGMENT_MARKER, then windows around
    segment text and around operating statistics (seat miles, carloads, subscribers, procedures ...) in MD&A and the notes,
    capped at max_chars. About 55% of the window budget goes to segment windows and 45% to operating-statistics windows;
    whichever family is short leaves its share to the other.

    status:
      ok                  segment note or MD&A segment discussion found (at least two mentions)
      limited_disclosure  Item 1 found and the filing states one reportable segment or has no segment note;
                          scored, because thin segment disclosure is what T measures
      extraction_failed   no Item 1 and no segment text: the 10-K body could not be located (technical failure)
    """
    text = _to_text(html)
    item1 = _item1_span(text) if text else None
    # only the Item 1 characters already taken as the body are skipped: later text in Item 1 (or after a runaway Item 1
    # span that never found its closing heading) is still searched
    taken = (item1[0], min(item1[1], item1[0] + ITEM1_CHARS)) if item1 else None
    seg_w = _segment_windows(text, taken) if text else []
    kpi_w = _kpi_windows(text, taken) if text else []
    n_mentions = sum(w[2] for w in seg_w)
    if item1 is None and n_mentions < 2:
        return "", "extraction_failed"
    body = text[taken[0]:taken[1]] if taken else ""
    budget = max_chars - len(body) - len(SEGMENT_MARKER)
    chosen = _pick_windows(kpi_w, int(budget * 0.45)) + _pick_windows(seg_w, int(budget * 0.55))
    used = sum(b - a for a, b in _union(chosen))
    for fam in (seg_w, kpi_w):                                     # hand any unused budget to whichever family has more
        extra = [(a, b) for a, b in _pick_windows(fam, budget - used) if not any(a >= c and b <= d for c, d in chosen)]
        if extra:
            chosen += extra
            used = sum(b - a for a, b in _union(chosen))
    spans = _union(chosen)
    # trim to the budget if merging left us over (keep earlier windows)
    trimmed, left = [], budget
    for a, b in spans:
        b = min(b, a + left)
        if b > a:
            trimmed.append((a, b))
            left -= b - a
    seg = "\n\n".join(text[a:b] for a, b in trimmed)
    out = (body + (SEGMENT_MARKER + seg if seg else ""))[:max_chars]
    status = "ok" if (n_mentions >= 2 and not _single_statement(text)) else "limited_disclosure"
    return out, status


def strip_identity(text: str, names: list, tickers=()) -> str:
    """Replace each company name with [COMPANY] (case-insensitive, whole words) and each ticker with [COMPANY] (CASE-
    SENSITIVE, standalone upper-case tokens only, so a ticker such as IT, LOW or KEY cannot mangle 'it', 'low-margin' or
    'key drivers'). Names shorter than two characters and tickers shorter than two are ignored."""
    out = text
    for n in sorted({n.strip() for n in names if n and len(n.strip()) >= 2}, key=len, reverse=True):
        out = re.sub(r"(?<!\w)" + re.escape(n) + r"(?!\w)", "[COMPANY]", out, flags=re.I)
    for t in sorted({t.strip().upper() for t in tickers if t and len(t.strip()) >= 2}, key=len, reverse=True):
        out = re.sub(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", "[COMPANY]", out)
    return out


# ---------------------------------------------------------------- rubric, parsing, config hash, reads
RUBRIC_VERSION = "r3"       # a draft until the calibration set has been read and the measurement is frozen

RUBRIC = """You are scoring how transparent a company's 10-K is about the economics of its business. The company name is
hidden. Read the excerpt (business description, segment note, MD&A segment text) and score what an outsider could
rebuild from it as PRICE x QUANTITY, weighted by where the profit and recurring cash flow actually come from.

Step 1. Identify the PROFIT ENGINES: the lines, products or contracts that generate the bulk of gross or operating profit
and recurring cash flow. Use segment profit and margin where disclosed. Otherwise use the filing's own language:
consumables, aftermarket, installed base, attach rates, subscriptions, royalties, float or interest income, high-margin
mix. A razor-and-blade business earns most of its profit on the blades (the recurring consumable), not the razor.
Step 2. For each engine, say whether the filing discloses its PRICE (average selling price, rates, take rates, attach
rates) and its QUANTITY (units, subscribers, procedures, bookings, volumes processed).
A price or quantity counts as disclosed when it is stated OR can be derived by dividing two disclosed figures for the same line and period (for example wireless revenue divided by wireless connections gives revenue per connection; revenue divided by units gives price per unit). The excerpt may combine several sources for the same company: the annual report (10-K), the quarterly reports (10-Qs) and the earnings releases and investor presentations furnished to the SEC; use all of them, and treat a figure given in any of them as disclosed.
Step 3. Score the share of the profit pool that disclosed price x quantity lets an outsider rebuild. Coverage is measured
against profit, not revenue: price and quantity for a small or low-margin line do not make a firm transparent.

T = 5  Price and quantity disclosed for the lines behind essentially all of the profit and recurring cash flow.
T = 4  Disclosed for the lines behind most (about three quarters or more) of the profit pool; the rest only in dollars.
T = 3  Disclosed for about half of the profit pool, or price OR quantity (not both) for the main profit engine.
T = 2  Revenue and profit visible by line or segment, but no price or quantity drivers for the main profit engine. This
       includes the razor-and-blade case: drivers disclosed only for a minor or low-margin line while the high-margin
       recurring line is dollars only. Also use at most 2 when you cannot tell where the profit comes from.
T = 1  Revenue only in total or by geography, with at most a few drivers named in prose and no figures.
T = 0  One undifferentiated revenue figure: no breakdown and no drivers at all.

The score is neutral to the number of segments: do not reward or penalise a firm for having one segment or many. A
single-segment firm that discloses price and quantity for its engine scores high; a many-segment firm that discloses only
dollars scores low. Judge only what the excerpt discloses, not what you know about the company.

Reply with JSON only, no other text:
{"engines": [{"line": "<name>", "profit_weight": "high|medium|low", "price": true|false, "quantity": true|false}], "T": <integer 0-5>, "rationale": "<one or two sentences: which lines earn the profit and what price and quantity data the excerpt gives for them>"}
Use an empty "engines" list only when you cannot identify where the profit comes from (then T must be 2 or less)."""

_WEIGHTS = {"high", "medium", "low"}


def parse_score(raw):
    """Strict integer 0-5: the int 3 or a string that is exactly one digit 0-5 (whitespace allowed); else None."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw <= 5 else None
    if isinstance(raw, str):
        m = re.fullmatch(r"\s*([0-5])\s*", raw)
        return int(m.group(1)) if m else None
    return None


def parse_reply(raw):
    """The scorer's JSON reply as a dict {"engines": [...], "T": int}, or None when it does not satisfy the contract.

    Tolerates code fences or a sentence around the JSON object. An empty engine list is valid only with T <= 2
    (the rubric's rule for a filing whose profit source cannot be identified).
    """
    import json
    if not isinstance(raw, str) or "{" not in raw:
        return None
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("engines"), list):
        return None
    t = parse_score(obj.get("T"))
    if t is None:
        return None
    for e in obj["engines"]:
        if not (isinstance(e, dict) and isinstance(e.get("line"), str) and e.get("profit_weight") in _WEIGHTS
                and isinstance(e.get("price"), bool) and isinstance(e.get("quantity"), bool)):
            return None
    if not obj["engines"] and t > 2:
        return None
    rat = obj.get("rationale", "")
    if rat is None:
        rat = ""
    if not isinstance(rat, str):
        return None
    return {"engines": obj["engines"], "T": t, "rationale": rat[:600]}


def rule_check(reply: dict) -> list:
    """Reasons a parsed reply contradicts itself (the razor-and-blade check): T >= 4 while a high-profit engine lacks
    price or quantity, or T > 2 with no high-profit engine named."""
    why = []
    engines = reply.get("engines", [])
    high = [e for e in engines if e["profit_weight"] == "high"]
    if reply["T"] >= 4:
        for e in high:
            if not (e["price"] and e["quantity"]):
                why.append(f"T={reply['T']} but high-profit engine '{e['line']}' lacks price or quantity")
    if reply["T"] > 2 and not high:
        why.append(f"T={reply['T']} but no high-profit engine named")
    return why


def config_hash(cfg: dict) -> str:
    """sha256 of a canonical JSON dump of the measurement config (key order and whitespace do not matter)."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def combine_reads(r1, r2, r3=None):
    """Merge two (or three) reads per unit. Final = mean of the two rounded half up; a gap of 2 or more needs a third
    read and the final is then the median of the three (NaN until the third read exists). A missing read gives no final
    score. Inputs have columns unit_id and score."""
    import numpy as np
    import pandas as pd
    out = (r1[["unit_id", "score"]].rename(columns={"score": "t1"})
           .merge(r2[["unit_id", "score"]].rename(columns={"score": "t2"}), on="unit_id", how="outer"))
    out["t3"] = np.nan
    if r3 is not None:
        out = out.drop(columns="t3").merge(r3[["unit_id", "score"]].rename(columns={"score": "t3"}), on="unit_id", how="left")
    both = out["t1"].notna() & out["t2"].notna()
    out["needs_third"] = both & ((out["t1"] - out["t2"]).abs() >= 2)
    final = np.floor((out["t1"] + out["t2"]) / 2 + 0.5)
    third = out[["t1", "t2", "t3"]].median(axis=1, skipna=False)
    out["t_final"] = np.where(out["needs_third"], third, np.where(both, final, np.nan))
    return out[["unit_id", "t1", "t2", "t3", "t_final", "needs_third"]]


def make_tasks(units, pass_id: str, out_dir, per_file: int = 40) -> int:
    """Export units as blind JSONL task files of `per_file` rows (id, text, rubric_version) for inspection or offline
    reading. Returns the number of files. Not part of the gate: the pilot and the full run use the production scorer."""
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"id": str(u), "text": t, "rubric_version": RUBRIC_VERSION} for u, t in zip(units["unit_id"], units["text"])]
    n = 0
    for k in range(0, len(rows), per_file):
        n += 1
        with open(out_dir / f"{pass_id}_{n}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows[k:k + per_file]:
                fh.write(json.dumps(r) + "\n")
    return n


# ---------------------------------------------------------------- reliability statistics, group rule, gate
GATE = {"kappa_reads": 0.70, "within_one": 0.90, "kappa_audit": 0.60, "max_adjacent_share": 0.80}
CUT_RULE = "p33_p67_v2"
MIN_FLAGGED = 8          # flagged events needed in each of the low and high groups
POOL_RATIO = 5.0         # non-flagged placebo pool needed per flagged event in each group


def _pairs(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep].astype(int), b[keep].astype(int)


def weighted_kappa(a, b, k: int = 6) -> float:
    """Quadratic-weighted Cohen's kappa between two integer ratings on levels 0..k-1 (NaN pairs dropped)."""
    import numpy as np
    a, b = _pairs(a, b)
    n = len(a)
    if n == 0:
        return float("nan")
    obs = np.zeros((k, k))
    for i, j in zip(a, b):
        obs[i, j] += 1
    i_, j_ = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    w = 1.0 - ((i_ - j_) / (k - 1)) ** 2
    po = (w * obs).sum() / n
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    pe = (w * exp).sum() / n
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def within_one(a, b) -> float:
    """Share of rated pairs that differ by at most one point."""
    a, b = _pairs(a, b)
    return float((abs(a - b) <= 1).mean()) if len(a) else float("nan")


def bunching(t) -> float:
    """Largest share of firm-years held by any two adjacent levels of T."""
    import numpy as np
    t = np.asarray(t, dtype=float)
    t = t[~np.isnan(t)].astype(int)
    if len(t) == 0:
        return float("nan")
    share = np.bincount(t, minlength=6)[:6] / len(t)
    return float(max(share[i] + share[i + 1] for i in range(5)))


def cut_points(t_universe):
    """(lo, hi): the 33rd percentile (lower interpolation) and 67th percentile (higher interpolation) of T."""
    import numpy as np
    t = np.asarray(t_universe, dtype=float)
    t = t[~np.isnan(t)]
    return float(np.percentile(t, 33, method="lower")), float(np.percentile(t, 67, method="higher"))


def assign_group(t, cuts):
    """Group rule p33_p67_v2: low = T <= lo, high = T >= hi, mid otherwise. If lo == hi == c (a six-level score can pile
    up on one value), low = T < c, high = T > c and mid = T == c. Equal T values always share a group; NaN stays NaN."""
    import numpy as np
    import pandas as pd
    t = pd.Series(t, dtype=float)
    lo, hi = cuts
    if lo == hi:
        low, high = t < lo, t > hi
    else:
        low, high = t <= lo, t >= hi
    g = pd.Series(np.where(low, "low", np.where(high, "high", "mid")), index=t.index, dtype=object)
    g[t.isna()] = np.nan
    return g


def group_viability(t_events, flagged, pool, cuts, *, min_flagged: int = MIN_FLAGGED, pool_ratio: float = POOL_RATIO) -> dict:
    """Whether the tercile tests T1 and T4 can run: the low and high groups each need at least `min_flagged` flagged
    events and a non-flagged placebo pool at least `pool_ratio` times the flagged count."""
    import pandas as pd
    g = assign_group(t_events, cuts)
    flagged, pool = pd.Series(flagged).astype(bool), pd.Series(pool).astype(bool)
    groups, reasons = {}, []
    for name in ("low", "high"):
        nf = int(((g == name) & flagged).sum())
        npool = int(((g == name) & pool).sum())
        groups[name] = {"flagged": nf, "pool": npool}
        if nf < min_flagged:
            reasons.append(f"{name}: {nf} flagged events (< {min_flagged})")
        elif npool < pool_ratio * nf:
            reasons.append(f"{name}: placebo pool {npool} is under {pool_ratio:g}x the {nf} flagged events")
    return {"ok": not reasons, "reasons": reasons, "groups": groups}


def evaluate_gate(reads, audit, t_all, viability, config_hash: str) -> dict:
    """Reliability gate (spec 6). `reads` has t1, t2 (pilot); `audit` has t_model, t_hand from the REPRESENTATIVE audit
    sample; `t_all` is T over the pilot firm-years.

    `viability` is group_viability's output, or None to defer it: group viability depends on how many FLAGGED events fall
    in each T group, which a 200-filing pilot of non-flagged filings cannot show, so it is decided at run time from the
    full scores (run_t1 and run_t4 refuse a non-viable group and the runner records the skip).

    passed: all four reliability criteria met. tests_enabled: T1-T4 when viability holds or is deferred, T2 and T3 when
    viability was checked and failed, none when reliability fails. Tied to the measurement config by `config_hash`.
    """
    m = {"kappa_reads": weighted_kappa(reads["t1"], reads["t2"]), "within_one": within_one(reads["t1"], reads["t2"]),
         "kappa_audit": weighted_kappa(audit["t_model"], audit["t_hand"]), "max_adjacent_share": bunching(t_all)}
    failures = [k for k in ("kappa_reads", "within_one", "kappa_audit") if not (m[k] >= GATE[k])]
    if not (m["max_adjacent_share"] <= GATE["max_adjacent_share"]):
        failures.append("max_adjacent_share")
    passed = not failures
    deferred = viability is None
    if not passed:
        enabled = []
    elif deferred or viability["ok"]:
        enabled = ["T1", "T2", "T3", "T4"]
    else:
        enabled = ["T2", "T3"]
    return {"passed": passed, "tests_enabled": enabled, "failures": failures, "metrics": m, "thresholds": dict(GATE),
            "viability": viability, "viability_deferred": deferred,
            "viability_failures": [] if deferred else list(viability["reasons"]), "cut_rule": CUT_RULE,
            "config_hash": config_hash}


# ---------------------------------------------------------------- freeze guard, sample selection, audit, leakage
class NotFrozen(Exception):
    """The measurement config is not frozen, or differs from the frozen one."""


class OverlapError(Exception):
    """Two samples share a filing accession."""


def freeze(path, cfg: dict) -> str:
    """Write the freeze record for a measurement config and return its hash. The config holds the model id, the prompt
    templates, the extraction and rubric versions and the rubric hash; any change afterwards breaks assert_frozen."""
    import hashlib
    import json
    from pathlib import Path
    h = config_hash(cfg)
    rec = {"config_hash": h, "cut_rule": CUT_RULE, "model": cfg.get("model"), "rubric_version": cfg.get("rubric_version"),
           "rubric_sha256": cfg.get("rubric_sha256"), "extraction_version": cfg.get("extraction_version"),
           "cli_version": cfg.get("cli_version"),
           "prompt_hashes": [hashlib.sha256(p.encode()).hexdigest() for p in cfg.get("prompts", [])]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return h


def assert_frozen(path, cfg: dict) -> None:
    """Raise NotFrozen unless the freeze file exists and `cfg` hashes to the frozen config hash."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise NotFrozen(f"no freeze file at {p}")
    frozen = json.loads(p.read_text(encoding="utf-8")).get("config_hash")
    if frozen != config_hash(cfg):
        raise NotFrozen("measurement config differs from the frozen one (model, prompts, extraction or rubric changed)")


def motivating_gvkeys() -> set:
    """gvkeys of the motivating firm(s), from GG_MOTIVATING_GVKEYS (comma separated) in the local environment only."""
    import os
    raw = os.environ.get("GG_MOTIVATING_GVKEYS", "")
    return {g.strip().zfill(6) for g in raw.split(",") if g.strip()}


def sic_division(sic) -> str:
    """SIC division letter A-J from a SIC code ('?' when missing)."""
    try:
        m = int(str(sic)[:2])
    except ValueError:
        return "?"
    for hi, d in [(9, "A"), (14, "B"), (17, "C"), (39, "D"), (49, "E"), (51, "F"), (59, "G"), (67, "H"), (89, "I"), (99, "J")]:
        if m <= hi:
            return d
    return "?"


def era(date) -> str:
    y = date.year
    return "le2009" if y <= 2009 else ("2010s" if y <= 2019 else "ge2020")


def _round_robin(groups: dict, n: int, rng) -> list:
    """Take one item from each group in turn (groups in random order, items shuffled) until n are taken."""
    order = list(groups)
    rng.shuffle(order)
    pools = {k: list(rng.permutation(len(v))) for k, v in groups.items()}
    out = []
    while len(out) < n and any(pools.values()):
        for k in order:
            if pools[k] and len(out) < n:
                out.append(groups[k][pools[k].pop()])
    return out


def select_units(events, index, n: int, seed: int, exclude_gvkeys=(), exclude_event_ids=(), exclude_accessions=()):
    """Outcome-blind sample of n filings for scoring: one row per accession, stratified by SIC division x era (equal
    allocation across strata, so every era is represented). For each event the filing is the last 10-K strictly before
    the announcement. Excludes the given gvkeys, event ids and accessions. Only event_id, gvkey and dep_ann are read
    from `events`; no return column is touched or returned."""
    import numpy as np
    import pandas as pd
    import gg_filings as F
    rng = np.random.default_rng(seed)
    ev = events[["event_id", "gvkey", "dep_ann"]].copy()
    ev["gvkey"] = ev["gvkey"].astype(str).str.zfill(6)
    ev = ev[~ev["gvkey"].isin(set(exclude_gvkeys)) & ~ev["event_id"].isin(set(exclude_event_ids))]
    ev = ev.iloc[rng.permutation(len(ev))]
    seen, rows = set(exclude_accessions), []
    for r in ev.itertuples():
        f = F.last_10k_before(index, r.gvkey, r.dep_ann)
        if f is None or f["accession"] in seen:
            continue
        seen.add(f["accession"])
        rows.append({"unit_id": f["accession"], "event_id": r.event_id, "gvkey": r.gvkey, "accession": f["accession"],
                     "cik": f["cik"], "url": f.get("url", ""), "sic_div": sic_division(f["sic"]), "era": era(r.dep_ann),
                     "dep_ann": r.dep_ann, "filing_date": f["filing_date"]})
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    groups = {k: [i for i in g.index] for k, g in cand.groupby(["sic_div", "era"])}
    picked = _round_robin({k: v for k, v in groups.items()}, n, rng)
    return cand.loc[picked].reset_index(drop=True)


def assert_disjoint(**named_sets) -> None:
    """Raise OverlapError when any two named sets of accessions share a member."""
    names = list(named_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(named_sets[a]) & set(named_sets[b])
            if shared:
                raise OverlapError(f"{a} and {b} share {len(shared)} accession(s): {sorted(shared)[:5]}")


def make_audit_sheet(df, path, n: int, seed: int, design: str, exclude_unit_ids=()) -> None:
    """CSV for the author's hand audit: unit_id, excerpt, t_hand blank. Model scores are never written. The excerpt is the
    segment and KPI text (what follows SEGMENT_MARKER in the extracted text; the start of the text when there is none),
    8,000 characters at most, so the author reads what the model read. The full text goes to <name>_full.jsonl.
    design='representative': simple random sample (the gate's sample). design='diagnostic': equal draws from each T level,
    interleaved across SIC divisions, to find rubric weaknesses (never in the gate). `exclude_unit_ids` removes units
    already audited, which keeps the two samples disjoint."""
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    rng = np.random.default_rng(seed)
    df = df[~df["unit_id"].isin(set(exclude_unit_ids))].reset_index(drop=True)
    if design == "representative":
        pick = df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]
    elif design == "diagnostic":
        ordered = {}
        for lvl, g in df.groupby("t_final"):
            g = g.iloc[rng.permutation(len(g))]
            if "sic_div" in g.columns:
                g = g.assign(_k=g.groupby("sic_div").cumcount()).sort_values("_k", kind="stable")
            ordered[lvl] = list(g.index)
        levels, picked = list(ordered), []
        rng.shuffle(levels)
        while len(picked) < n and any(ordered.values()):
            for lvl in levels:
                if ordered[lvl] and len(picked) < n:
                    picked.append(ordered[lvl].pop(0))
        pick = df.loc[picked]
    else:
        raise ValueError(f"unknown design {design!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def _excerpt(t):
        head, _, other = t.partition(BUNDLE_MARKER)
        part = (head.split(SEGMENT_MARKER)[-1] if SEGMENT_MARKER in head else head)[:8000]
        return part + (BUNDLE_MARKER + other[:6000] if other else "")

    excerpt = pick["text"].map(_excerpt)
    pd.DataFrame({"unit_id": pick["unit_id"].values, "excerpt": excerpt.values, "t_hand": ""}).to_csv(path, index=False)
    with open(path.with_name(path.stem + "_full.jsonl"), "w", encoding="utf-8") as fh:
        for u, t in zip(pick["unit_id"], pick["text"]):
            fh.write(json.dumps({"unit_id": u, "text": t}) + "\n")


def leakage_rate(guesses) -> float:
    """Share of rows where the scorer's guessed company matches the true one (case-insensitive, trimmed)."""
    g = guesses["guess"].astype(str).str.strip().str.lower()
    t = guesses["truth"].astype(str).str.strip().str.lower()
    return float((g == t).mean()) if len(g) else float("nan")


def estimate_tokens(texts) -> int:
    """Rough token count: characters // 4, summed."""
    return int(sum(len(t) for t in texts) // 4)


# ---------------------------------------------------------------- production scorer: Claude Code headless, subscription
EXCERPT_SLOT = "<<EXCERPT>>"
PROMPT_A = RUBRIC + "\n\nExcerpt:\n" + EXCERPT_SLOT
PROMPT_B = ("Task: grade the 10-K excerpt at the end for disclosure transparency, using the rubric below.\n\n" + RUBRIC +
            "\n\n--- 10-K EXCERPT ---\n" + EXCERPT_SLOT + "\n--- END OF EXCERPT ---\nReturn the JSON object only.")
# Isolation for every read: no tools, no saved session, no skills or MCP, no user or project settings (so no hooks,
# memory or CLAUDE.md), and a one-line system prompt. --bare is deliberately not used: it needs an API key, not the
# subscription login. Reads also run from an empty working directory.
DEFAULT_FLAGS = ("--tools", "", "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config",
                 "--setting-sources", "", "--system-prompt", "You are a careful scoring function. Reply with JSON only.")
IDENTITY_STRIP_VERSION = IDENTITY_VERSION


class RunnerTimeout(Exception):
    """A headless call did not finish in time (a hung or very slow call); safe to retry."""


class UsageLimit(Exception):
    """The subscription's usage limit (or an overload) stopped a call; wait and retry."""


_CLI_VERSION = []


def _claude_version() -> str:
    """The installed claude CLI version (cached); 'unknown' when it cannot be read."""
    if not _CLI_VERSION:
        import subprocess
        import shutil
        try:
            r = subprocess.run([shutil.which("claude") or "claude", "--version"], capture_output=True, text=True, timeout=30)
            _CLI_VERSION.append((r.stdout.strip().splitlines() or ["unknown"])[0])
        except Exception:
            _CLI_VERSION.append("unknown")
    return _CLI_VERSION[0]


def measurement_config(model: str, *, cli_flags: tuple = DEFAULT_FLAGS, cli_version: str | None = None) -> dict:
    """The whole measurement pipeline as one dict (hash it with config_hash, freeze it with freeze). Headless mode has no
    temperature setting, so it is recorded as 'default'. The claude CLI version is included because an update can change
    the default context, tools or model-alias resolution of a headless call."""
    import hashlib
    return {"model": model, "prompts": [PROMPT_A, PROMPT_B], "cli_flags": list(cli_flags), "temperature": "default",
            "cli_version": cli_version or _claude_version(),
            "rubric_version": RUBRIC_VERSION, "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
            "extraction_version": EXTRACTION_VERSION, "identity_version": IDENTITY_STRIP_VERSION}


def claude_runner(model: str, flags: tuple = DEFAULT_FLAGS, timeout: int = 300):
    """A runner `prompt -> reply text` that calls `claude -p` on the author's subscription, from an empty temporary
    directory. Raises UsageLimit when the CLI reports a limit or overload. Tokens per call are appended to `.usage`."""
    import json
    import shutil
    import subprocess
    import tempfile
    exe = shutil.which("claude") or "claude"
    scratch = tempfile.mkdtemp(prefix="gg_claude_cwd_")
    usage = []

    def run(prompt: str) -> str:
        try:
            r = subprocess.run([exe, "-p", "--model", model, "--output-format", "json", *flags], input=prompt, text=True,
                               encoding="utf-8", capture_output=True, cwd=scratch, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunnerTimeout(f"claude -p did not answer within {timeout}s")
        out = r.stdout.strip()
        try:
            obj = json.loads(out[out.index("{"):])
        except ValueError:
            obj = {"is_error": True, "result": (r.stderr or out)[:500]}
        if r.returncode != 0 or obj.get("is_error"):
            msg = str(obj.get("result") or r.stderr)[:300]
            if re.search(r"limit|quota|overload|rate", msg, re.I):
                raise UsageLimit(msg)
            raise RuntimeError(f"claude -p failed: {msg}")
        u = obj.get("usage", {})
        usage.append({"in": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
                      "out": u.get("output_tokens", 0), "ms": obj.get("duration_ms")})
        return obj["result"]

    run.usage = usage
    return run


def score_reads(units, *, cfg: dict, cache_path, runner=None, reads=(1, 2), max_retries: int = 1,
                require_frozen=None, sleeper=None, wait_seconds: int = 900, max_waits: int = 96, max_timeouts: int = 3):
    """Score each unit (columns unit_id, text) once per read id with the production scorer; returns one row per
    (unit, read): unit_id, read, score, config_hash, input_sha256, engines (JSON), flags (rule_check), raw.

    Read 1 uses PROMPT_A, read 2 PROMPT_B, read 3 PROMPT_A again (an independent sample). An unparseable reply is retried
    `max_retries` times, then scored NaN. Results are appended to `cache_path` (JSONL) as they arrive, keyed by (unit,
    read, input hash, config hash), so a resume never reuses a result from another config or another text. A usage
    limit makes the call wait `wait_seconds` and retry (up to `max_waits` times). With `require_frozen` (the freeze
    file), the config must match the frozen one, else NotFrozen.
    """
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    if require_frozen is not None:
        assert_frozen(require_frozen, cfg)
    runner = runner or claude_runner(cfg["model"], tuple(cfg["cli_flags"]))
    sleeper = sleeper or time.sleep
    chash = config_hash(cfg)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["unit_id"], r["read"], r["input_sha256"], r["config_hash"])] = r
    prompts = {1: cfg["prompts"][0], 2: cfg["prompts"][1], 3: cfg["prompts"][0]}
    rows = []
    for uid, text in zip(units["unit_id"], units["text"]):
        ih = hashlib.sha256(text.encode()).hexdigest()
        for rd in reads:
            key = (uid, rd, ih, chash)
            if key in done:
                rows.append(done[key])
                continue
            prompt = prompts[rd].replace(EXCERPT_SLOT, text)
            parsed, raw, timed_out = None, "", False
            for attempt in range(max_retries + 1):
                waits = tmo = 0
                while True:
                    try:
                        raw = runner(prompt if attempt == 0 else prompt + "\n\nYour previous reply was not valid. Reply with the JSON object only.")
                        break
                    except UsageLimit:
                        waits += 1
                        if waits > max_waits:
                            raise
                        sleeper(wait_seconds)
                    except RunnerTimeout:
                        tmo += 1
                        if tmo > max_timeouts:
                            timed_out, raw = True, ""
                            break
                        sleeper(5 * tmo)
                if timed_out:
                    break
                parsed = parse_reply(raw)
                if parsed is not None:
                    break
            rec = {"unit_id": uid, "read": rd, "score": float(parsed["T"]) if parsed else float("nan"),
                   "config_hash": chash, "input_sha256": ih,
                   "engines": json.dumps(parsed["engines"]) if parsed else "",
                   "rationale": parsed["rationale"] if parsed else "",
                   "flags": "; ".join(rule_check(parsed)) if parsed else ("timeout" if timed_out else "unparseable"), "raw": raw[:2000]}
            if not timed_out:                       # a timed-out read is never cached: a resume must retry it
                with open(cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                done[key] = rec
            rows.append(rec)
    cols = ["unit_id", "read", "score", "config_hash", "input_sha256", "engines", "rationale", "flags", "raw"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
