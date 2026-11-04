"""
gg_phases: reporting phases for Stage 5 (docs/STAGE5_SPEC.md).

Companies rarely change how they report revenue from one year to the next, so one transparency score can cover a whole
stretch of filings (a "phase") instead of scoring every 10-K. A phase ends only when the revenue split visibly changes:

  1. revenue_labels   pull the revenue-breakdown labels (segment, product, geography lines) out of a 10-K       [computation]
  2. compare_labels   fuzzy-compare two consecutive filings' labels: NONE / MINOR / MAJOR / UNKNOWN              [computation]
  3. triage_change    only for a MAJOR change: a small, cheap model check on just the two label lists            [model]
  4. assign_phases    a phase starts only at a MAJOR change the triage confirms                                  [computation]

To keep the no-look-ahead rule, a phase is scored from its FIRST filing (using only what was filed up to then) and that
score is applied to every later filing in the phase; a later filing is never used to score an earlier quarter.
"""
from __future__ import annotations

import difflib
import json
import re

# where a revenue breakdown lives: "net sales by segment", "revenue by product", "disaggregation of revenue", segment notes
_ANCHOR = re.compile(
    r"(?:net\s+)?(?:sales|revenues?)\s+by\s+(?:reportable\s+|operating\s+|business\s+)?"
    r"(?:segments?|products?|product\s+lines?|geograph\w+|business\w*|lines?|categor\w+|types?|region\w*|markets?|customers?)"
    r"|disaggregat\w+\s+of\s+(?:net\s+)?revenues?|segment\s+(?:information|results|reporting)|(?:reportable|operating)\s+segments?", re.I)
# a sentence that lists the segments: "We have three reportable segments: A, B and C."
_SEGLIST = re.compile(r"(?:reportable|operating|business|principal)\s+segments?\s*(?:are|were|include|consist\s+of|comprise|:)\s*:?\s*([^.]{8,400})", re.I)
_NUMERIC = re.compile(r"[\$\(\)\s\d,\.%\-—–]+")
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9&/,'’\-\. ()]{2,60}")
_STOP = re.compile(
    r"\b(?:total|net income|net earnings|income|earnings|profit|loss|cost|costs|expenses?|assets|liabilities|equity|depreciation|"
    r"amortization|interest|taxes|per share|shares|segment|segments|year|years|december|september|june|march|millions?|thousands?|"
    r"unaudited|note|notes|table|page|other income|operating income|operating profit|gross|margin|percent|change|increase|decrease|"
    r"consolidated|elimination|eliminations|corporate|unallocated|reconciliation|quarter|quarters|first|second|third|fourth|receivable|receivables|"
    r"inventor\w*|raw materials|work in process|finished goods|allowance|unbilled|deferred|accrued|payable|payables|goodwill|intangible\w*|"
    r"impairment|restructuring|charges?|property|plant|equipment|research and development|selling|general and administrative|compensation|"
    r"stock|dividends?|cash|debt|borrowings|derivatives?|pension|lease|leases|warrants?|capital|surplus|outstanding|weighted|diluted|basic)\b", re.I)
_NOISE_WORDS = re.compile(r"\b(?:net|revenues?|sales|and other|reportable|segment|segments)\b", re.I)
WINDOW = 4_000
MAX_LABELS = 60
MAJOR_BELOW, MINOR_BELOW = 0.60, 0.90        # similarity thresholds: < 0.60 MAJOR, < 0.90 MINOR, otherwise NONE
FUZZY = 0.80                                  # difflib ratio at which two labels are the same line of business


def _norm(label: str) -> str:
    """Lower-case a label and drop parentheticals, years, filler words ('net', 'revenues', 'segment') and punctuation."""
    s = re.sub(r"\(.*?\)", " ", label.lower())
    s = re.sub(r"\b(?:19|20)\d\d\b", " ", s)
    s = _NOISE_WORDS.sub(" ", s)
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _good(norm: str) -> bool:
    return 3 <= len(norm) <= 55 and not _STOP.search(norm) and not norm.isdigit()


_TABLE = re.compile(r"<table\b.*?</table>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
# a table is a revenue breakdown when its caption (the text just before it, its <caption>, or its first rows) names revenue or sales
# AND a way of splitting it (segment, product, line, region, geography, business, category, market, customer, brand)
_REV_WORD = re.compile(r"\b(?:net\s+)?(?:sales|revenues?)\b", re.I)
_SPLIT_WORD = re.compile(r"\b(?:segments?|products?|product\s+lines?|lines?|regions?|geograph\w+|business\w*|categor\w+|markets?|customers?|brands?|types?)\b", re.I)
CAPTION_CHARS = 450


def _plain(chunk: str) -> str:
    import html as H
    return re.sub(r"\s+", " ", H.unescape(_TAG.sub(" ", chunk))).strip()


def _table_rows(chunk: str) -> list:
    """Rows of one HTML table as lists of cell strings (empty cells dropped)."""
    from lxml import html as LH
    try:
        el = LH.fragment_fromstring(chunk, create_parent="div")
    except Exception:
        return []
    rows = []
    for tr in el.iter("tr"):
        cells = [" ".join(td.text_content().split()) for td in tr if td.tag in ("td", "th")]
        cells = [c for c in cells if c and c not in ("$", ")", "%")]
        if cells:
            rows.append(cells)
    return rows


def revenue_labels(html: str) -> list:
    """Normalised revenue-breakdown labels in a filing. Reads the HTML tables themselves: a table counts when the text just
    before it (or its caption or header rows) names revenue or sales together with a way of splitting it (segment, product line,
    region ...) and at least two of its rows are a label followed by two or more numbers. The first-column labels are returned,
    minus totals and income lines. Segments named in a sentence ("we have three reportable segments: A, B and C") are added.
    An empty list means none were found. Linear in the size of the document (no whole-document HTML tree)."""
    if not html or not html.strip():
        return []
    found, last = [], 0
    pieces = []
    for m in _TABLE.finditer(html):
        pieces.append(("text", html[last:m.start()]))
        pieces.append(("table", m.group(0)))
        last = m.end()
    pieces.append(("text", html[last:]))
    before = ""
    for kind, chunk in pieces:
        if kind == "text":
            plain = _plain(chunk)
            for sm in _SEGLIST.finditer(plain):
                for part in re.split(r",|;|\band\b", sm.group(1)):
                    n = _norm(part)
                    if _good(n) and len(part.split()) <= 7:
                        found.append(n)
            before = (before + " " + plain)[-CAPTION_CHARS:]
            continue
        rows = _table_rows(chunk)
        if len(rows) >= 2:
            cap = re.search(r"<caption.*?</caption>", chunk, re.I | re.S)       # a caption with no closing tag is simply ignored
            head = " ".join(" ".join(r) for r in rows[:2]) + (" " + _plain(cap.group(0)) if cap else "")
            caption = before + " " + head
            if _REV_WORD.search(caption) and _SPLIT_WORD.search(caption):
                good_rows = []
                for r in rows:
                    nums = [c for c in r[1:] if _NUMERIC.fullmatch(c) and re.search(r"\d", c)]
                    if len(nums) >= 2 and re.search(r"[A-Za-z]", r[0]):
                        good_rows.append(r[0])
                if len(good_rows) >= 2:
                    for lab in good_rows:
                        n = _norm(lab)
                        if _good(n):
                            found.append(n)
        before = ""
    seen, out = set(), []
    for n in found:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:MAX_LABELS]


def _same_line(a: str, b: str) -> bool:
    """Two labels name the same line of business: fuzzy text match, or every word of the shorter is in the longer."""
    if difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY:
        return True
    ta, tb = set(a.split()), set(b.split())
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return bool(short) and short <= long_


def compare_labels(a: list, b: list) -> dict:
    """Compare the revenue breakdowns of two consecutive filings.

    kept_old = share of the previous filing's labels still present; kept_new = share of the current filing's labels that already
    existed. A real restructuring loses old labels AND gains new ones, so it is MAJOR only when BOTH are below MAJOR_BELOW. A table
    that merely appears or vanishes (extra rows on one side) leaves one side fully matched, so it is never MAJOR on its own.
    similarity = (matched in a + matched in b) / (n_a + n_b) grades the rest: NONE (>= 0.90) or MINOR. UNKNOWN when either filing
    gave no labels (never an automatic break)."""
    if not a or not b:
        return {"level": "UNKNOWN", "similarity": None, "kept_old": None, "kept_new": None, "n_a": len(a), "n_b": len(b), "added": [], "removed": []}
    ma = [x for x in a if any(_same_line(x, y) for y in b)]
    mb = [y for y in b if any(_same_line(y, x) for x in a)]
    sim = (len(ma) + len(mb)) / (len(a) + len(b))
    kept_old, kept_new = len(ma) / len(a), len(mb) / len(b)
    level = "MAJOR" if (kept_old < MAJOR_BELOW and kept_new < MAJOR_BELOW) else ("NONE" if sim >= MINOR_BELOW else "MINOR")
    return {"level": level, "similarity": round(sim, 3), "kept_old": round(kept_old, 3), "kept_new": round(kept_new, 3), "n_a": len(a), "n_b": len(b),
            "added": [y for y in b if y not in mb], "removed": [x for x in a if x not in ma]}


def triage_change(prev: list, cur: list, runner) -> dict:
    """The small model check for a MAJOR change: was the revenue breakdown really changed drastically? The prompt holds only the
    two label lists, so it is cheap. Returns {"major": True/False/None, "reason": str}; None means an unusable answer."""
    prompt = ("Two consecutive annual reports of the same company list these revenue-breakdown labels (segments, product lines or "
              "regions).\nPREVIOUS YEAR: " + "; ".join(prev[:40]) + "\nCURRENT YEAR: " + "; ".join(cur[:40]) + "\n"
              "Did the company change its revenue breakdown drastically (segments added, dropped, merged or replaced, so that the two "
              "years are no longer comparable)? A renamed line, a reordering, or one small addition is NOT drastic.\n"
              'Reply with JSON only: {"major": true|false, "reason": "<one sentence>"}')
    raw = runner(prompt)
    try:
        obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        major = obj.get("major")
        return {"major": major if isinstance(major, bool) else None, "reason": str(obj.get("reason", ""))[:300]}
    except (ValueError, AttributeError):
        return {"major": None, "reason": "unparseable triage answer"}


def assign_phases(levels: list, triage) -> list:
    """Phase number for each filing of one firm, in date order. `levels[i]` is the change level from filing i-1 to filing i
    (the first entry is a placeholder). A new phase starts at filing i only when levels[i] == 'MAJOR' and `triage(i)` is True;
    triage is called for MAJOR changes only."""
    phases, cur = [], 0
    for i, lv in enumerate(levels):
        if i > 0 and lv == "MAJOR" and triage(i) is True:
            cur += 1
        phases.append(cur)
    return phases
