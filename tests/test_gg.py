"""
Tests for gg_labels (names, sentences, classification) and gg_tests (portfolio / alpha machinery).
Run:  python tests/test_gg.py    (from the repo root; no pytest needed)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # modules live in src/
import gg_labels as L
import gg_tests as T
import gg_econ as E

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


# ---------------------------------------------------------------- names
def test_names():
    for raw, last in [("Howard T. Widra, J.D.", "WIDRA"), ("William Douglas Parker, BEC, M.B.A.", "PARKER"),
                      ("John W. Robinson, III, C.F.A.", "ROBINSON"), ("Fred B. Parks, Ph.D.", "PARKS"),
                      ("Brandon R. Elliott, CFA", "ELLIOTT"), ("Terry E. London, CPA", "LONDON")]:
        check(f"last name of {raw!r} = {last}", L.last_name(raw) == last, L.last_name(raw))
    pairs = [("Michael W. Rayden", "RAYDEN MICHAEL W", True), ("Robert Smith", "SMITH BOB", True),
             ("Robert Smith", "SMITH JOHN R", False), ("Robert Smith", "SMITH ROBERTA", False),
             ("Jan van der Berg", "VAN DER BERG JAN", True), ("David Jones", "JONES THOMAS", False),
             ("William Douglas Parker, BEC, M.B.A.", "PARKER DOUGLAS", True), ("Howard T. Widra, J.D.", "WIDRA HOWARD T", True)]
    for e, o, exp in pairs:
        check(f"match {e!r} ~ {o!r} -> {exp}", (L.match_owner(e, o) >= 3) == exp)


# ---------------------------------------------------------------- sentences
def test_sentences():
    s = L.split_sentences("Avery has been named CEO, following the resignation of William H. Doe. Mr. Doe leaves today.")
    check("no split after middle initial or 'Mr.'", len(s) == 2 and s[0].endswith("Doe."), str(s))
    s2 = L.split_sentences("Acme, Inc. Announces Management Changes. John will step down.")
    check("no split after 'Inc.'", len(s2) == 2, str(s2))


# ---------------------------------------------------------------- classification
D = pd.Timestamp("2020-06-01")
CASES = [  # (text, last tokens, expected dep_type, expected abrupt)
    ("Acme Corp Announces CEO Resignation. Acme announced that John Smith has resigned as CEO, effective immediately. The Board named Jane Doe interim CEO.", ["SMITH"], "resigned_abrupt", True),
    ("Acme Corp Announces CEO Retirement. John Smith will retire as CEO on December 31, 2020 after a transition period; the Board has begun a search for a successor.", ["SMITH"], "retirement_planned", False),
    ("Acme Corp. The Board terminated John Smith as CEO for cause.", ["SMITH"], "terminated_forced", True),
    ("Acme Corp. John Smith, CEO, is stepping down for health reasons.", ["SMITH"], "health_personal", False),
    ("Acme Corp. Jane Doe has been named CEO, succeeding John Smith who resigned.", ["SMITH"], "resigned_abrupt", True),
    ("Acme Corp. The Board appointed John Smith as interim chief executive officer following the resignation of Bob Jones.", ["SMITH"], "not_a_departure", None),
    ("Acme Corp. Pat Lee was appointed Secretary of the Company. Kim Roe holds a Master of Business Administration.", ["LEE"], "not_a_departure", None),
    ("Acme Corp. John Smith has been nominated by President Biden to serve as Secretary of the Treasury and will resign as CEO.", ["SMITH"], "government", False),
]


def test_classifier():
    for text, lt, exp_t, exp_a in CASES:
        c = L.classify_item(text, lt, D)
        check(f"classify -> {exp_t}: {text[:50]!r}", c["dep_type"] == exp_t, f"got {c['dep_type']}")
        if exp_a is not None:
            check(f"   abrupt={exp_a}", c["abrupt"] == exp_a, f"got {c['abrupt']}")


def test_no_false_government():
    c = L.classify_item("Acme. Pat Lee, General Counsel and Secretary of Acme, will cease serving as an officer effective August 31, 2020.", ["LEE"], D)
    check("'Secretary of Acme' is not a government role", c["dep_type"] != "government", c["dep_type"])


# ---------------------------------------------------------------- calendar-time test machinery
def _synthetic_market(T_days=2500, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2005-01-03", periods=T_days)
    fac = pd.DataFrame({"mktrf": rng.normal(0.0004, 0.010, T_days), "smb": rng.normal(0, 0.005, T_days),
                        "hml": rng.normal(0, 0.005, T_days), "umd": rng.normal(0, 0.006, T_days),
                        "rmw": 0.0, "cma": 0.0, "rf": 0.00005}, index=cal)
    return cal, fac, rng


def _events_and_windows(cal, fac, rng, n_events=900, alpha_daily=0.0, treated_frac=0.1):
    """Random stocks with factor exposure + idiosyncratic noise; a subset gets a planted daily alpha."""
    T_days = len(cal)
    ev = []
    wins = {}
    for k in range(n_events):
        start = rng.integers(0, T_days - 300)
        idx = np.arange(start + 2, start + 254)
        beta = rng.uniform(0.7, 1.4)
        treated = k < int(n_events * treated_frac)
        r = (fac.rf.values[idx] + beta * fac.mktrf.values[idx] + rng.normal(0, 0.018, len(idx))
             + (alpha_daily if treated else 0.0))
        wins[f"e{k}"] = (idx, r)
        ev.append({"event_id": f"e{k}", "year": cal[start].year, "treated": treated})
    return pd.DataFrame(ev), wins


def test_alpha_recovery_and_size():
    cal, fac, rng = _synthetic_market()
    ev, wins = _events_and_windows(cal, fac, rng, alpha_daily=-0.0008, treated_frac=0.12)   # about -20% a year
    T_ = len(cal)
    treated = ev[ev.treated].event_id.tolist()
    port, cnt = T.portfolio(wins, treated, T_)
    r = T.alpha_test(port, cnt, fac)
    check("planted alpha (-20%/yr) is recovered with the right sign", r["alpha_ann"] < -0.10, f"{r['alpha_ann']:.3f}")
    # size: with NO effect, a random flagged set should be 'significant' about 5% of the time against the placebo
    rej, reps = 0, 60
    for s in range(reps):
        cal2, fac2, rng2 = _synthetic_market(seed=100 + s)
        ev2, w2 = _events_and_windows(cal2, fac2, rng2, n_events=600, alpha_daily=0.0, treated_frac=0.1)
        fl = ev2[ev2.treated].event_id.tolist()
        p_, c_ = T.portfolio(w2, fl, len(cal2))
        obs = T.alpha_test(p_, c_, fac2)["alpha_ann"]
        null = T.placebo_alphas(w2, fl, ev2, fac2, len(cal2), reps=60, seed=s)
        if T.placebo_p(obs, null, side="two") < 0.05:
            rej += 1
    check("false-positive rate under the null is near 5%", rej / reps <= 0.15, f"{rej}/{reps} rejected")


def test_power():
    """With ~65 flagged events, what annual alpha can the placebo test detect? (reported, not asserted)"""
    out = []
    for a in (0.0, -0.0004, -0.0008, -0.0012):
        hit, reps = 0, 25
        for s in range(reps):
            cal, fac, rng = _synthetic_market(seed=500 + s)
            ev, wins = _events_and_windows(cal, fac, rng, n_events=600, alpha_daily=a, treated_frac=0.11)
            fl = ev[ev.treated].event_id.tolist()
            p_, c_ = T.portfolio(wins, fl, len(cal))
            obs = T.alpha_test(p_, c_, fac)["alpha_ann"]
            null = T.placebo_alphas(wins, fl, ev, fac, len(cal), reps=60, seed=s)
            hit += T.placebo_p(obs, null, side="lower") < 0.05
        out.append((a * 252, hit / reps))
    print("POWER (66 flagged events, placebo test, one-sided 5%):", ", ".join(f"alpha {a:+.0%}/yr -> detect {p:.0%}" for a, p in out))
    check("power rises with effect size", out[-1][1] >= out[0][1])


def test_alpha_fast_matches_full():
    cal, fac, rng = _synthetic_market()
    ev, wins = _events_and_windows(cal, fac, rng, alpha_daily=-0.0004, treated_frac=0.2)
    port, cnt = T.portfolio(wins, ev[ev.treated].event_id.tolist(), len(cal))
    a_full = T.alpha_test(port, cnt, fac)["alpha_ann"]
    a_fast = T.alpha_fast(port, cnt, fac)
    check("alpha_fast equals the full WLS alpha", abs(a_full - a_fast) < 1e-6, f"{a_full} vs {a_fast}")


def test_split_permutation():
    """Null: labels unrelated to returns -> p roughly uniform; planted difference -> small p."""
    rej, reps = 0, 40
    for s in range(reps):
        cal, fac, rng = _synthetic_market(seed=900 + s)
        ev, wins = _events_and_windows(cal, fac, rng, n_events=240, alpha_daily=0.0, treated_frac=0.0)
        flags = rng.random(len(ev)) < 0.5
        obs, a1, a0, null = T.split_permutation(wins, ev.event_id, flags, ev.year, fac, len(cal), reps=150, seed=s)
        rej += T.placebo_p(obs, null, side="lower") < 0.05
    check("split permutation: false-positive rate near 5% when labels are random", rej / reps <= 0.15, f"{rej}/{reps}")
    cal, fac, rng = _synthetic_market(seed=77)
    ev, wins = _events_and_windows(cal, fac, rng, n_events=240, alpha_daily=-0.0010, treated_frac=0.5)
    flags = ev.treated.values
    obs, a1, a0, null = T.split_permutation(wins, ev.event_id, flags, ev.year, fac, len(cal), reps=300, seed=1)
    check("split permutation detects a planted -25%/yr difference", T.placebo_p(obs, null, side="lower") < 0.05, f"obs {obs:.3f}")


def test_holm():
    h = T.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    check("holm adjusts upward and keeps order", h["a"] <= h["b"] <= h["c"] and h["a"] >= 0.01)


def _fe_sim(seed, exposure, n_firms=300, n_q=40, beta=0.0):
    r = np.random.default_rng(seed)
    f = np.repeat(np.arange(n_firms), n_q); q = np.tile(np.arange(n_q), n_firms)
    keep = r.random(len(f)) < 0.5; f, q = f[keep], q[keep]
    e = r.normal(0, 1, n_firms); w = r.normal(0, 1, n_q); sh = r.normal(0, 1, n_q); fe = r.normal(0, 1, n_firms)
    x = r.normal(0, 1, len(f)) + (e[f] * w[q] if exposure else 0)
    y = beta * x + fe[f] + r.normal(0, 1, len(f)) + (e[f] * sh[q] if exposure else 0)
    return pd.DataFrame({"y": y, "x": x, "firm": f, "q": q})


def test_fe_ols():
    d = _fe_sim(1, False, beta=0.5)
    r = E.fe_ols(d, "y", ["x"], year="q")
    check("fe_ols recovers a planted slope with firm effects", abs(r["x"]["coef"] - 0.5) < 0.05, f"{r['x']['coef']:.3f}")
    # same coefficient as OLS with explicit firm and time dummies
    import statsmodels.formula.api as smf
    small = d[d.firm < 60]
    m = smf.ols("y ~ x + C(firm) + C(q)", small).fit()
    check("fe_ols slope equals the dummy-variable OLS slope", abs(E.fe_ols(small, "y", ["x"], year="q")["x"]["coef"] - m.params["x"]) < 1e-8)
    R = 120
    f1 = f2 = 0
    for s_ in range(R):
        dd = _fe_sim(s_, True)
        f1 += E.fe_ols(dd, "y", ["x"], year="q")["x"]["p_two"] < .05
        f2 += E.fe_ols(dd, "y", ["x"], year="q", cluster2="q")["x"]["p_two"] < .05
    print(f"  null size with firm-specific exposure to time shocks: firm-only {f1/R:.0%}, two-way {f2/R:.0%}")
    check("firm-only clustering is badly oversized under common shocks (why two-way is primary)", f1 / R > 0.3, f"{f1/R:.0%}")
    check("two-way clustering is close to nominal under common shocks", f2 / R < 0.2, f"{f2/R:.0%}")
    pw = np.mean([E.fe_ols(_fe_sim(500 + s_, False, beta=0.1), "y", ["x"], year="q", cluster2="q")["x"]["p_two"] < .05 for s_ in range(60)])
    check("two-way clustering still has power for a real slope", pw > 0.5, f"{pw:.0%}")


if __name__ == "__main__":
    for fn in (test_names, test_sentences, test_classifier, test_no_false_government, test_holm,
               test_alpha_fast_matches_full, test_split_permutation, test_fe_ols,
               test_alpha_recovery_and_size, test_power):
        fn()
    print("\n%d failures" % len(FAILS), FAILS)
    sys.exit(1 if FAILS else 0)
