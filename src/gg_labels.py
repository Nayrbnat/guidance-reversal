"""
gg_labels.py -- name parsing, executive-to-insider matching, and departure-text classification
for the Guidance Reversal screen.

Everything here is deterministic and unit-tested (test_gg_labels.py) and validated against a
hand-labelled gold set (validation/), so the flag does not rest on unchecked keyword matches.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
GENERATION = {"JR", "SR", "II", "III", "IV", "V"}
PARTICLES = {"DE", "DA", "DI", "DEL", "DELLA", "DER", "VAN", "VON", "LA", "LE", "ST", "MC", "MAC", "O", "BIN", "AL"}

NICKNAMES = {
    "WILLIAM": {"BILL", "WILL", "BILLY", "WILLY"}, "ROBERT": {"BOB", "ROB", "BOBBY", "ROBBIE"},
    "RICHARD": {"RICK", "DICK", "RICH", "RICKY"}, "JAMES": {"JIM", "JIMMY", "JAMIE"},
    "JOHN": {"JACK", "JON", "JOHNNY"}, "MICHAEL": {"MIKE", "MICK"}, "THOMAS": {"TOM", "TOMMY"},
    "CHARLES": {"CHARLIE", "CHUCK", "CHAS"}, "EDWARD": {"ED", "TED", "EDDIE", "NED"},
    "DANIEL": {"DAN", "DANNY"}, "DAVID": {"DAVE", "DAVY"}, "JOSEPH": {"JOE", "JOEY"},
    "STEPHEN": {"STEVE", "STEVEN"}, "STEVEN": {"STEVE", "STEPHEN"}, "ANTHONY": {"TONY"},
    "ANDREW": {"ANDY", "DREW"}, "GREGORY": {"GREG"}, "KENNETH": {"KEN", "KENNY"},
    "DONALD": {"DON", "DONNY"}, "RONALD": {"RON", "RONNIE"}, "TIMOTHY": {"TIM", "TIMMY"},
    "PATRICK": {"PAT"}, "LAWRENCE": {"LARRY", "LAURENCE"}, "GERALD": {"JERRY", "GERRY"},
    "JEFFREY": {"JEFF", "GEOFFREY"}, "MATTHEW": {"MATT"}, "NICHOLAS": {"NICK", "NICKY"},
    "CHRISTOPHER": {"CHRIS", "KIT"}, "ALEXANDER": {"ALEX", "AL", "SANDY"}, "BENJAMIN": {"BEN", "BENNY"},
    "SAMUEL": {"SAM", "SAMMY"}, "FREDERICK": {"FRED", "FREDDY", "FREDERIC"}, "HENRY": {"HANK", "HARRY"},
    "ELIZABETH": {"BETH", "LIZ", "BETSY", "LIZZIE", "BETTY", "ELIZA"}, "KATHERINE": {"KATE", "KATHY", "KATIE", "CATHERINE", "KATHRYN"},
    "CATHERINE": {"KATE", "KATHY", "KATIE", "KATHERINE", "CATHY"}, "MARGARET": {"PEGGY", "MAGGIE", "MEG", "MARGE"},
    "JENNIFER": {"JEN", "JENNY"}, "SUSAN": {"SUE", "SUSIE"}, "PATRICIA": {"PAT", "PATTY", "TRISH"},
    "DEBORAH": {"DEB", "DEBBIE", "DEBRA"}, "BARBARA": {"BARB", "BARBIE"}, "VICTORIA": {"VICKI", "VICKY"},
    "RAYMOND": {"RAY"}, "LEONARD": {"LEN", "LEO", "LENNY"}, "PHILIP": {"PHIL", "PHILLIP"}, "PHILLIP": {"PHIL", "PHILIP"},
    "THEODORE": {"TED", "TEDDY"}, "WALTER": {"WALT"}, "ALBERT": {"AL", "BERT"}, "ARTHUR": {"ART"},
    "DOUGLAS": {"DOUG"}, "GORDON": {"GORDY"}, "RANDALL": {"RANDY"}, "TERRENCE": {"TERRY", "TERENCE"}, "TERENCE": {"TERRY", "TERRENCE"},
    "VINCENT": {"VINCE", "VINNY"}, "ZACHARY": {"ZACH", "ZAK"}, "NATHANIEL": {"NATE", "NAT"}, "ROSS": set(),
    "PETER": {"PETE"}, "FRANCIS": {"FRANK", "FRAN"}, "FRANK": {"FRANCIS"}, "GEORGE": set(), "JONATHAN": {"JON", "JONNY"},
    "CHRISTINE": {"CHRIS", "CHRISTY", "TINA"}, "STEPHANIE": {"STEPH"}, "RACHEL": {"RACH"}, "ALEXANDRA": {"ALEX", "SANDRA", "SANDY"},
}
# make nickname relation symmetric
for _k, _v in list(NICKNAMES.items()):
    for _n in _v:
        NICKNAMES.setdefault(_n, set()).add(_k)


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def parse_name(raw) -> dict:
    """Parse an ExecuComp-style name, e.g. 'Howard T. Widra, J.D.' / 'John W. Robinson, III, C.F.A.'.

    Everything after the first comma is credentials and is discarded (ExecuComp writes 'First M. Last, CRED'),
    which fixes the old tokenizer that took 'D' from 'J.D.' or 'CFA' as the last name.
    Returns dict(first, given[list], middle_initials[list], last, last_tokens[list], generation).
    """
    s = _ascii(raw).strip()
    head = s.split(",")[0]
    words = [w for w in re.split(r"\s+", head) if w]
    gen = ""
    while words and re.sub(r"[^A-Za-z]", "", words[-1]).upper() in GENERATION:
        gen = re.sub(r"[^A-Za-z]", "", words[-1]).upper()
        words.pop()
    toks = [re.sub(r"[^A-Za-z']", "", w).upper() for w in words]
    toks = [t for t in toks if t]
    if not toks:
        return dict(first="", given=[], middle_initials=[], last="", last_tokens=[], generation=gen)
    last_word = words[len(toks) - 1] if len(words) >= len(toks) else words[-1]
    last_parts = [re.sub(r"[^A-Za-z']", "", p).upper() for p in re.split(r"[-]", last_word)]
    last_parts = [p for p in last_parts if p]
    last_tokens = last_parts[:]
    i = len(toks) - 1
    while i - 1 >= 1 and toks[i - 1] in PARTICLES:       # van / de la / mc ...
        last_tokens.insert(0, toks[i - 1])
        i -= 1
    given = toks[: i] if i >= 1 else []
    # a lone first token is the first name; drop single initials from the "given" names but keep them as initials
    first = given[0] if given else ""
    mid = [g[0] for g in given[1:]]
    return dict(first=first, given=given, middle_initials=mid, last=last_tokens[-1] if last_tokens else "",
                last_tokens=last_tokens, generation=gen)


def last_name(raw) -> str:
    return parse_name(raw)["last"]


def first_name(raw) -> str:
    return parse_name(raw)["first"]


def _given_equivalent(a: str, b: str) -> bool:
    return a == b or b in NICKNAMES.get(a, set()) or a in NICKNAMES.get(b, set())


def match_owner(exec_name, owner) -> int:
    """Convenience wrapper: parse the executive name, then match_parsed()."""
    return match_parsed(parse_name(exec_name), owner)


def match_parsed(p: dict, owner) -> int:
    """Score how well a Thomson insider 'owner' string ('LAST FIRST M') matches a parsed ExecuComp name.

    3 = last name and a full first/given name (or nickname) agree
    2 = last name agrees and the owner gives only an initial that matches the first or a given name
    0 = no match
    The caller works within one firm (already restricted by CUSIP), which keeps false positives low.
    """
    if not p["last"]:
        return 0
    ot = [t for t in re.sub(r"[^A-Za-z' ]", " ", _ascii(owner)).upper().split() if t not in GENERATION]
    if not ot:
        return 0
    n = len(p["last_tokens"])
    # Thomson writes the last name first; accept the last-name tokens appearing as a leading block
    # (any order for hyphenated names), with a fallback to "all tokens present".
    lead = ot[:n]
    if not (set(p["last_tokens"]) <= set(ot) and (set(lead) == set(p["last_tokens"]) or p["last"] in ot[:2])):
        return 0
    rest = [t for i, t in enumerate(ot) if t not in p["last_tokens"] or i >= n]
    rest = [t for t in rest if t not in p["last_tokens"]]
    if not rest or not p["given"]:
        return 0
    given_full = [g for g in p["given"] if len(g) > 1]
    exec_initials = {g[0] for g in p["given"]}
    first_o = rest[0]                       # Thomson: LAST FIRST [MIDDLE...]; the first token after the last name
    if len(first_o) > 1:
        # a full first name must agree (allowing nicknames and the exec's other given names, e.g. 'W. Douglas');
        # a conflicting full first name is a different person even if a middle initial happens to match
        if any(_given_equivalent(g, first_o) for g in given_full):
            return 3
        if not given_full and first_o[0] in exec_initials:
            return 2                         # ExecuComp gave only initials
        return 0
    # owner gives only an initial
    return 2 if first_o in exec_initials else 0


# --------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c(])')


_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Jr|Sr|Inc|Corp|Co|Ltd|St|Prof|Gen|Gov|Sen|Rep|No|vs|Messrs|U\.S|Cos|Bros)\.|\b[A-Z]\.(?=\s+[A-Z])")
_DOT = "․"


def split_sentences(text) -> list:
    """Split into sentences without breaking after initials ('William H. Osborne'), titles ('Mr.') or 'Inc.'."""
    t = re.sub(r"\s+", " ", _ascii(text if isinstance(text, str) else "")).strip()
    t = _ABBREV.sub(lambda m: m.group(0).replace(".", _DOT), t)
    return [s.replace(_DOT, ".") for s in _SENT_SPLIT.split(t) if s]


def mentions_name(sentence: str, last_tokens) -> bool:
    """Case-insensitive whole-word mention of the executive's (full) last name."""
    if not last_tokens:
        return False
    pat = r"\b" + r"[\s-]+".join(re.escape(t) for t in last_tokens) + r"\b"
    return re.search(pat, sentence, re.I) is not None


def name_excerpt(text, last_tokens, max_chars=1500) -> str:
    """Headline sentence + every sentence that mentions the executive (with one neighbour each side)."""
    sents = split_sentences(text)
    if not sents:
        return ""
    keep = {0}
    for i, s in enumerate(sents):
        if mentions_name(s, last_tokens):
            keep.update({max(0, i - 1), i, min(len(sents) - 1, i + 1)})
    out = " ".join(sents[i] for i in sorted(keep))
    return out[:max_chars]


# --------------------------------------------------------------------------------------
# Departure classification (sentence level, anchored on the executive's name)
# --------------------------------------------------------------------------------------
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_MONTH_NUM = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}
_MONTH_NUM.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})

DEP_VERB = re.compile(
    r"\b(resign\w*|step(?:s|ped|ping)?\s+down|steps?\s+aside|retir\w*|terminat\w*|depart\w*|"
    r"(?:has|have|had|will|is|to)\s+leav\w*|left\s+the|leaving|cease[sd]?\s+(?:to\s+serve|serving|to\s+be|being)|relieved|"
    r"removed|ousted|dismiss\w*|fired|separat\w*|part(?:ed|ing|s)?\s+ways|transition(?:ing|s)?\s+(?:out|from)|"
    r"will\s+not\s+(?:stand|seek|stay)|no\s+longer\s+(?:serv|with|employ)|passed\s+away|died|death)\b", re.I)
APPOINT = re.compile(r"\b(appoint\w*|named|promot\w*|elect\w*|hire[sd]?|select\w*|join(?:s|ed)?|succeed\w*|replac\w*|assum\w*|become|will\s+serve|to\s+serve\s+as|new\s+(?:chief|ceo|cfo|president))\b", re.I)
IMMEDIATE = re.compile(r"\b(effective\s+immediately|with\s+immediate\s+effect|immediately(?!\s+(?:following|after|prior|preceding|upon))|effective\s+today|effective\s+(?:as\s+of\s+)?(?:the\s+)?date\s+hereof|effective\s+upon\s+(?:the\s+)?(?:filing|execution|announcement))\b", re.I)
TERMINATION = re.compile(r"\b(terminat\w*|removed|ousted|dismiss\w*|fired|relieved\s+of|not\s+(?:to\s+)?renew\w*|non-?renew\w*|separation|part(?:ed|ing|s)?\s+ways|asked\s+(?:him|her)\s+to\s+(?:resign|leave|step)|without\s+cause|for\s+cause|requested\s+(?:his|her)\s+resignation)\b", re.I)
HEALTH = re.compile(r"\b(health|medical|illness|family\s+(?:reasons|matters|circumstances)|personal\s+(?:reasons|matters|circumstances)|spend\s+more\s+time\s+with\s+(?:his|her)\s+family)\b", re.I)
DEATH = re.compile(r"\b(passed\s+away|died|death\s+of|untimely|deceased|sudden\s+death|tragically)\b", re.I)
PLANNED = re.compile(r"\b(will\s+retire|plans?\s+to\s+retire|intends?\s+to\s+retire|announced\s+(?:his|her)\s+(?:plans?\s+to\s+)?retire\w*|retire\w*\s+(?:effective|on|at|as\s+of|from|in|upon|after)|"
                     r"(?:until|when|upon|once|after)\s+(?:a\s+|his\s+|her\s+)?(?:successor|replacement)|successor\s+(?:is|has\s+been|will\s+be)\s+(?:named|identified|appointed|selected)|search\s+for\s+(?:a\s+)?(?:successor|new)|"
                     r"succession\s+plan\w*|transition\s+(?:period|plan|role|to\s+(?:the\s+role|executive\s+chair|chairman|advisor|senior\s+advisor|consult))|"
                     r"remain\s+(?:as|an?|with|in)|continue\s+(?:to\s+serve|as|in)|assist\s+(?:with|in)\s+the\s+transition|through\s+(?:the\s+end|december|june|march|september|year)|"
                     r"will\s+step\s+down\s+(?:as\s+of|on|at|effective|in|when|upon))\b", re.I)
GOV = re.compile(
    r"\b(?:nominat\w*|appoint\w*|select\w*|tapped|chosen|join\w*|accept\w*|named|picked|confirmed)\b[^.]{0,120}?"
    r"\b(white\s+house|administration\s+of|(?:biden|trump|obama|bush|clinton)\s+administration|cabinet|"
    r"secretary\s+of\s+(?:the\s+)?(?:treasury|state|defense|commerce|labor|energy|transportation|health|homeland|agriculture|education|interior|veterans|housing|war|army|navy|air\s+force)|"
    r"(?:deputy|assistant|under)\s+secretary\b|ambassador\s+to|u\.?s\.?\s+ambassador|federal\s+reserve\s+(?:board|bank\s+of)|federal\s+reserve\s+governor|"
    r"u\.?s\.?\s+(?:senate|house\s+of\s+representatives|department\s+of|treasury|trade\s+representative)|"
    r"(?:fda|sec|ftc|fcc|cftc|fdic|nasa|faa)\s+(?:commissioner|chair\w*|administrator|director)|governor\s+of|state\s+of\s+[A-Z]\w+\s+(?:cabinet|department)|"
    r"(?:congress|senate)\b|run\s+for\s+(?:office|governor|senate|congress))\b", re.I)
OTHER_JOB = re.compile(r"\b(?:to|will)\s+(?:become|join|accept|assume|serve\s+as|take|lead|head|be\s+named|be\s+appointed)\b[^.]{0,100}?\b(?:chief|ceo|president|chairman|head|officer|partner)\b[^.]{0,60}?\b(?:of|at)\s+(?!the\s+(?:board|company)\b)(?:the\s+)?(?-i:[A-Z])", re.I)


def _eff_days(sentence: str, ann_date):
    """Days between the announcement and an 'effective <date>' mention (None if absent; 0 if immediate)."""
    import datetime as _dt
    if IMMEDIATE.search(sentence):
        return 0
    if ann_date is None:
        return None
    lead = r"(?:effective|as\s+of|on|by|until|through|beginning|starting|in)\s+(?:the\s+close\s+of\s+business\s+on\s+)?"
    m = re.search(lead + "(" + _MONTHS + r")\.?\s+(\d{1,2})(?!\d)(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", sentence, re.I)
    d = None
    if m:
        mon, day = _MONTH_NUM[m.group(1).lower()], int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else ann_date.year
        try:
            d = _dt.date(yr, mon, min(day, 28))
        except ValueError:
            d = None
        if d is not None and not m.group(3) and d < ann_date.date() - _dt.timedelta(days=30):
            d = d.replace(year=d.year + 1)
    else:
        m = re.search(lead + "(" + _MONTHS + r")\.?,?\s+(\d{4})", sentence, re.I)       # 'by May 2014'
        if m:
            try:
                d = _dt.date(int(m.group(2)), _MONTH_NUM[m.group(1).lower()], 28)
            except ValueError:
                d = None
    if d is not None:
        return (d - ann_date.date()).days
    m = re.search(r"(?:effective|as\s+of|by|at\s+the\s+end\s+of|through)\s+(?:the\s+)?(?:end\s+of\s+)?(?:fiscal\s+)?(?:year|20\d\d|(?:first|second|third|fourth)\s+quarter|q[1-4])", sentence, re.I)
    return 120 if m else None


def departing_sentences(text, last_tokens):
    """Sentences in which the executive is the one leaving (not the one being appointed)."""
    sents = split_sentences(text)
    out = []
    for i, s in enumerate(sents):
        if not mentions_name(s, last_tokens):
            continue
        name_pat = r"\b" + r"[\s-]+".join(re.escape(t) for t in last_tokens) + r"\b"
        nm = re.search(name_pat, s, re.I)
        if not nm:
            continue
        before, after = s[: nm.start()], s[nm.end():]
        dv_after = DEP_VERB.search(after[:160])
        if dv_after:
            seg = after[: dv_after.start()]
            # the verb must attach to this person: not after another person's clause ("... following the resignation of X")
            if len(seg) > 90 or re.search(r"\b(following|after|succeed\w*|replac\w*|as\s+(?:interim\s+)?(?:chief|president|ceo|cfo|chair\w*))\b", seg, re.I):
                dv_after = None
        dv_before = DEP_VERB.search(before[-120:])
        ap_before = APPOINT.search(before[-70:])
        # "succeed(s)/replace(s) NAME" -> NAME leaves
        succeeds_name = re.search(r"\b(succeed\w*|replac\w*|terminat\w*|remov\w*|dismiss\w*|fired|ousted|relieved)\s+(?:\w+\.?\s+){0,4}$", before[-50:], re.I)
        poss = re.match(r"['’]s\s+(?:resignation|departure|retirement|decision\s+to|termination|separation|removal|dismissal|ouster)", after, re.I) or \
            re.search(r"\b(?:resignation|departure|retirement|termination|separation|removal|dismissal|ouster|firing)\s+of\s+(?:\w+\.?\s+){0,4}$", before[-60:], re.I)
        appointee_first = bool(ap_before) and not succeeds_name and not re.search(r"\bresign|retir|step", before[-70:], re.I)
        # The text each rule reads is a window around the name, so cues about OTHER people in the same
        # sentence ("... Jerome will be appointed CEO") are not attributed to this executive.
        win_obj = s[max(0, nm.start() - 90): nm.end() + 130]
        win_subj = s[nm.start():]
        if succeeds_name or poss:
            out.append((i, win_obj)); continue
        if dv_after and not (appointee_first and not re.search(r"\b(resign|retir|step|depart|terminat)\w*", after[:90], re.I) and re.match(r"\s*(?:,|\s)(?:has|will|is|was)\s+(?:been\s+)?(?:appoint|named|promot|elect)", after, re.I)):
            # the verb follows the name: require the name be the subject (no appointment verb directly before it)
            if not appointee_first or re.search(r"\b(?:resign|retir|step\w*\s+down|depart|leav|terminat)\w*", after[:60], re.I):
                out.append((i, win_subj)); continue
        if dv_before and not ap_before and re.search(r"\b(?:of|by|that)\s+(?:\w+\.?\s+){0,3}$", before[-30:], re.I):
            out.append((i, win_obj))
    return out


FUTURE_DEP = re.compile(
    r"\b(?:will|plans?\s+to|intends?\s+to|expects?\s+to|is\s+expected\s+to|to\s+be)\s+(?:be\s+)?(?:step\w*\s+down|retir\w*|resign\w*|leav\w*|depart\w*|transition|relinquish\w*|succeeded)|"
    r"\b(?:intention|plans?|decision)\s+to\s+(?:retire|step\s+down|resign|leave)|\bsucceeds?\b(?:(?!resign|stepp|left|depart)[^.]){0,60}\bon\s+(?:" + _MONTHS + r")\b", re.I)
PAST_DEP = re.compile(
    r"\b(?:has|have|had)\s+(?:resigned|stepped\s+down|left|departed|retired)|\bresigned\b|\bstepped\s+down\b|\bis\s+(?:resigning|stepping\s+down|leaving|departing)|"
    r"\bstepping\s+down\b|\bresignation\b|\bresigns?\b|\bsteps?\s+down\b|\bdeparture\b|\bcessation\b|\bleaves\b|\bdeparts\b|\bsteps?\s+aside\b|\bleft\s+the\b", re.I)
OTHER_JOB2 = re.compile(r"\b(?:accepted|accept|pursue|pursuing|take|taking)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:career\s+|executive\s+|leadership\s+)?(?:opportunit\w*|position|role|job)\s+(?:at|with|in|outside|elsewhere)", re.I)


def classify_item(text, last_tokens, ann_date=None) -> dict:
    """Classify one announcement for one executive. Returns dict(departing, dep_type, abrupt, gov, eff_days).

    abrupt = the departure was sudden: effective immediately / interim named / terminated / 'has resigned'
    without an announced orderly transition. A future-tense retirement or step-down with a succession plan
    is planned, however short the notice.
    """
    ds = departing_sentences(text, last_tokens)
    if not ds:
        return dict(departing=False, dep_type="not_a_departure", abrupt=None, gov=False, eff_days=None)
    sents = split_sentences(text)
    idx = {i for i, _ in ds}
    core = " ".join(w for _, w in ds)                     # windows around the executive's name
    ctx = " ".join(sents[i] for i in sorted(idx | {i + 1 for i in idx}) if i < len(sents))
    eff = None
    for _, w in ds:
        e = _eff_days(w, ann_date)
        if e is not None:
            eff = e if eff is None else min(eff, e)
    if eff is not None and eff < 0:
        eff = 0                                            # already effective at announcement
    interim = bool(re.search(r"\binterim\b", ctx, re.I))
    immediate = bool(IMMEDIATE.search(core)) or eff == 0
    future = bool(FUTURE_DEP.search(core))
    short_notice = eff is not None and eff <= 14
    planned_cue = bool(PLANNED.search(core)) or future or (eff is not None and eff > 45)
    past = bool(PAST_DEP.search(core))
    no_chair = re.sub(r"(?:non-?executive|executive|vice)\s+chair\w*", "", core, flags=re.I)
    if DEATH.search(core):
        t = "death"
    elif GOV.search(core):
        t = "government"
    elif TERMINATION.search(core):
        t = "terminated_forced"
    elif HEALTH.search(core):
        t = "health_personal"
    elif OTHER_JOB.search(no_chair) or OTHER_JOB2.search(core):
        t = "left_for_other_job"
    elif (immediate or interim or (future and short_notice)) and not (future and (eff is None or eff > 45)):
        t = "resigned_abrupt"
    elif planned_cue:
        t = "retirement_planned"
    elif past:
        t = "resigned_abrupt"
    elif re.search(r"\bretir\w*", core, re.I):
        t = "retirement_planned"
    else:
        t = "other_unclear"
    if t == "left_for_other_job":
        abrupt = eff is None or eff <= 30
    else:
        abrupt = True if t in ("resigned_abrupt", "terminated_forced") else (None if t == "other_unclear" else False)
    return dict(departing=True, dep_type=t, abrupt=abrupt, gov=(t == "government"), eff_days=eff)


def classify_event(items, exec_name) -> dict:
    """items: list of dicts with date (Timestamp/str) and text. Picks the EARLIEST item that announces this
    executive leaving (the departure announcement) and classifies it."""
    import pandas as pd
    p = parse_name(exec_name)
    best = None
    for k in sorted(items, key=lambda x: str(x["date"])):
        d = pd.Timestamp(k["date"])
        c = classify_item(k["text"], p["last_tokens"], d)
        if c["departing"]:
            best = dict(c, item=k.get("i"), ann_date=d)
            break
    return best or dict(departing=False, dep_type="not_a_departure", abrupt=None, gov=False, eff_days=None, item=None, ann_date=None)
