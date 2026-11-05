"""Regression and results tables for MEMO.md, generated from outputs/results/*.json so no number is typed by hand.

Convention (as in an econometrics paper): each column is one specification, each variable takes two rows (coefficient with stars,
then the standard error in parentheses), followed by rows for fixed effects, controls, observations and clusters.
Stars: * p<0.10, ** p<0.05, *** p<0.01 on the two-sided p-value of the coefficient. A blank cell means the variable is not in that
specification. Used by scripts/build_memo.py."""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "outputs" / "results"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def f(x, d=3):
    return "" if x is None else f"{x:,.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def reg_table(specs, var_labels, extra_rows, d=3):
    """specs: list of (column title, result dict with variable -> {coef, se, p_two}). var_labels: ordered (key, label)."""
    header = [""] + [t for t, _ in specs]
    rows = []
    for key, label in var_labels:
        if not any(isinstance(r.get(key), dict) for _, r in specs):
            continue
        c, s = [label], [""]
        for _, r in specs:
            v = r.get(key)
            if isinstance(v, dict) and "coef" in v:
                c.append(f"{v['coef']:.{d}f}{stars(v['p_two'])}")
                s.append(f"({v['se']:.{d}f})")
            else:
                c.append("")
                s.append("")
        rows += [c, s]
    for label, vals in extra_rows:
        rows.append([label] + list(vals))
    return md_table(header, rows)


def n_row(specs, key="n"):
    return [f"{r[key]:,}" if r.get(key) is not None else "" for _, r in specs]


# ------------------------------------------------------------------ Stage 1 (calendar-time alpha, placebo test)
def table_stage1():
    s = load("study_results.json")
    cols = [("Primary", s["primary"] | {"holm_p": None})] + [(x["name"].split(" ")[0], x) for x in s["secondary"]]
    header = [""] + [c[0] for c in cols]
    rows = [["Four-factor alpha, % per year"] + [f"{100 * c[1]['alpha_ann']:.1f}" for c in cols],
            ["Naive t-statistic"] + [f"{c[1]['t']:.2f}" for c in cols],
            ["Placebo p-value (one-sided, lower tail)"] + [f"{c[1]['placebo_p']:.2f}" for c in cols],
            ["Holm-adjusted p"] + ["n/a" if c[1].get("holm_p") is None else f"{c[1]['holm_p']:.2f}" for c in cols],
            ["Flagged events"] + [f"{c[1]['n_events']}" for c in cols],
            ["Mean positions held"] + [f"{c[1]['mean_pos']:.1f}" for c in cols],
            ["Factors: Mkt-RF, SMB, HML, UMD"] + ["Yes"] * len(cols)]
    legend = "; ".join(f"**{x['name'].split(' ')[0]}** {' '.join(x['name'].split(' ')[1:])}" for x in s["secondary"])
    return md_table(header, rows) + chr(10) * 2 + "*Columns: Primary is the pre-specified test. " + legend + ".*"


def table_stage2():
    s = load("stage2_results.json")
    F1 = s["F1"]
    h1 = [""] + [x["name"].replace("F1 ", "") for x in F1]
    r1 = [["Events"] + [str(x["n_events"]) for x in F1],
          ["Four-factor alpha, % per year"] + [f"{100 * x['alpha_ann']:.1f}" for x in F1],
          ["Placebo p-value"] + [f"{x['placebo_p']:.2f}" for x in F1]]
    F2 = s["F2"]
    h2 = [""] + [x["name"].replace("F2 ", "") for x in F2]
    r2 = [["Optimistic-guidance events"] + [str(x["n_optimistic"]) for x in F2],
          ["Other events"] + [str(x["n_other"]) for x in F2],
          ["Alpha, optimistic, % per year"] + [f"{100 * x['alpha_optimistic']:.1f}" for x in F2],
          ["Alpha, other, % per year"] + [f"{100 * x['alpha_other']:.1f}" for x in F2],
          ["Difference (optimistic minus other)"] + [f"{100 * x['diff']:.1f}" for x in F2],
          ["Permutation p (one-sided)"] + [f"{x['perm_p']:.3f}" for x in F2],
          ["Holm-adjusted p (family of 14)"] + [f"{x['holm_p']:.3f}" if x.get("holm_p") is not None else "" for x in F2]]
    return "**Panel A. Alpha of the whole group (no optimism split)**\n\n" + md_table(h1, r1) + "\n\n**Panel B. Optimistic versus other guidance**\n\n" + md_table(h2, r2)


# ------------------------------------------------------------------ Stage 3 (panel regressions)
P1_VARS = [("O", "Optimistic guidance (O)"), ("S", "Sale since guidance (S)"), ("OS", "O x S"),
           ("sale12", "Sale in prior 12 months"), ("lmcap", "Log market cap"), ("ret12", "12-month return"), ("ltenure", "Log tenure"), ("age", "Age")]
P2_VARS = [("O", "Optimistic guidance (O)"), ("S", "Sale (S)"), ("X", "Exit (X)"), ("OS", "O x S"), ("OX", "O x X"), ("SX", "S x X"),
           ("OSX", "O x S x X"), ("lmcap", "Log market cap"), ("ret12", "12-month return"), ("ldays", "Log days to period end"),
           ("qtr", "Quarterly release"), ("ltenure", "Log tenure"), ("age", "Age")]


def table_p1():
    s = load("stage3_results.json")
    R = s["robustness"]
    specs = [("(1) Baseline", s["primary"]["P1"]), ("(2) O continuous", R["P1_R1_continuous"]), ("(3) Sale >= $1M", R["P1_R2_sale_1m"]),
             ("(4) Excl. retirements", R["P1_R9_excl_retired"]), ("(5) 2004-14", R["P1_R12_2004_2014"]), ("(6) 2015-25", R["P1_R12_2015_2025"]),
             ("(7) Skip a quarter", R["P1_R13_skip_quarter"])]
    extra = [("Firm fixed effects", ["Yes"] * 7), ("Year-quarter fixed effects", ["Yes"] * 7), ("Observations (CEO-quarters)", n_row(specs)),
             ("Firm clusters", n_row(specs, "n_clusters")), ("One-sided p, O x S > 0", [f"{r['p_one']:.2f}" for _, r in specs])]
    return reg_table(specs, P1_VARS, extra)


def table_p2():
    s = load("stage3_results.json")
    R = s["robustness"]
    specs = [("(1) Baseline", s["primary"]["P2"]), ("(2) O continuous", R["P2_R1_continuous"]), ("(3) Sale >= $1M", R["P2_R2_sale_1m"]),
             ("(4) vs consensus", R["P2_R4_vs_consensus"]), ("(5) First release", R["P2_R5_first_release"]), ("(6) Excl. retired", R["P2_R9_excl_retired"]),
             ("(7) Pre-guidance sale", R["P2_R11_pre_guidance_sale"]), ("(8) 2004-14", R["P2_R12_2004_2014"]), ("(9) 2015-25", R["P2_R12_2015_2025"]),
             ("(10) Sale ends 90d pre-exit", R["P2_R13_sale_ends_90d_before_exit"]), ("(11) Drop near exit", R["P2_R14_drop_near_exit"])]
    k = len(specs)
    extra = [("Firm fixed effects", ["Yes"] * k), ("Year-quarter fixed effects", ["Yes"] * k), ("Observations (releases)", n_row(specs)),
             ("Firm clusters", n_row(specs, "n_clusters")), ("One-sided p, O x S x X < 0", [f"{r['p_one']:.2f}" for _, r in specs])]
    return reg_table(specs, P2_VARS, extra)


def table_p3():
    p = load("stage3_results.json")["P3"]
    rows = [["Leavers (guided, sold, left)", f"{p['n_leavers']}", f"{100 * p['alpha_leavers']:.1f}"],
            ["Stayers (guided, sold, stayed)", f"{p['n_stayers']:,} events from {p['n_stayer_execs']:,} executives", f"{100 * p['alpha_stayers']:.1f}"],
            ["Difference, leavers minus stayers", "", f"{100 * p['diff']:.1f}"],
            ["Permutation p (one-sided, lower tail)", "", f"{p['perm_p']:.2f}"]]
    return md_table(["", "Events", "Four-factor alpha, % per year"], rows)


# ------------------------------------------------------------------ Stage 5
def table_t2():
    s3, s5 = load("stage3_results.json"), load("stage5_results.json")
    t2 = s5["tests"]["T2"]["result"]
    specs = [("(1) Stage 3 baseline (no T)", s3["primary"]["P1"]), ("(2) Test 2: with T", t2)]
    var = [("O", "Optimistic guidance (O)"), ("S", "Sale since guidance (S)"), ("OS", "O x S"), ("T", "T (standardised, Z)"), ("TO", "Z x O"),
           ("TS", "Z x S"), ("TOS", "**Z x O x S**")] + P1_VARS[3:]
    extra = [("Firm fixed effects", ["Yes"] * 2), ("Year-quarter fixed effects", ["Yes"] * 2), ("Observations (CEO-quarters)", n_row(specs)),
             ("Firm clusters", n_row(specs, "n_clusters")), ("Exits", [f"{s3['primary']['P1']['n_exits']:,}", f"{t2['n_exits']:,}"]),
             ("H5 test, one-sided p", ["", f"{t2['p_one']:.2f} (H5: Z x O x S < 0)"]), ("Minimum detectable Z x O x S", ["", f"{t2['mde']:.2f}"])]
    return reg_table(specs, var, extra)


def table_t3():
    s3, s5 = load("stage3_results.json"), load("stage5_results.json")
    t3 = s5["tests"]["T3"]["result"]
    specs = [("(1) Stage 3 baseline (no T)", s3["primary"]["P2"]), ("(2) Test 3: with T", t3)]
    var = [("O", "Optimistic guidance (O)"), ("S", "Sale (S)"), ("X", "Exit (X)"), ("OS", "O x S"), ("OX", "O x X"), ("SX", "S x X"), ("OSX", "O x S x X"),
           ("T", "T (standardised, Z)"), ("TO", "Z x O"), ("TS", "Z x S"), ("TX", "Z x X"), ("TOS", "Z x O x S"), ("TOX", "Z x O x X"), ("TSX", "Z x S x X"),
           ("TOSX", "**Z x O x S x X**")] + P2_VARS[7:]
    extra = [("Firm fixed effects", ["Yes"] * 2), ("Year-quarter fixed effects", ["Yes"] * 2), ("Observations (releases)", n_row(specs)),
             ("Firm clusters", n_row(specs, "n_clusters")), ("Releases with sale and exit", [f"{s3['primary']['P2']['n_SX']:,}", f"{t3['n_SX']:,}"]),
             ("H5 test, one-sided p", ["", f"{t3['p_one']:.2f} (H5: Z x O x S x X > 0)"]), ("Minimum detectable Z x O x S x X", ["", f"{t3['mde']:.2f}"])]
    return reg_table(specs, var, extra)


def table_t1():
    r = load("stage5_results.json")["tests"]["T1"]["result"]
    g = r["groups"]
    lo, hi = g["low"], g["high"]
    rows = [["Flagged events", lo["n"], hi["n"], ""],
            ["Four-factor alpha of flagged firms, % per year", f"{100 * lo['alpha_ann']:.1f}", f"{100 * hi['alpha_ann']:.1f}", f"{100 * (lo['alpha_ann'] - hi['alpha_ann']):.1f}"],
            ["Mean alpha of same-size placebo groups, % per year", f"{100 * lo['placebo_mean']:.1f}", f"{100 * hi['placebo_mean']:.1f}", ""],
            ["**Excess alpha (flagged minus placebo), % per year**", f"{100 * lo['excess_ann']:.1f}", f"{100 * hi['excess_ann']:.1f}", f"**{100 * r['did']:.1f}**"],
            ["Placebo pool (non-flagged events, announcement identified)", f"{lo['pool']:,}", f"{hi['pool']:,}", ""],
            ["95% bootstrap interval for the difference (calendar-quarter blocks)", "", "", f"{100 * r['ci'][0]:.1f} to {100 * r['ci'][1]:.1f}"],
            ["One-sided placebo p, H5: difference < 0", "", "", f"{r['perm_p']:.3f}"],
            ["Calendar-time spread, low-T minus high-T portfolio alpha (t)", "", "", f"{100 * r['spread_alpha']:.1f} (t = {r['spread_t']:.2f})"],
            ["Minimum detectable difference, % per year", "", "", f"{100 * r['mde']:.0f}"]]
    return md_table(["", "(1) Low T (T <= 2)", "(2) High T (T >= 3)", "(3) Low minus high"], rows)


def table_t4():
    s5 = load("stage5_results.json")
    r = s5["tests"]["T4"]["result"]
    lo, hi, pr, ct = r["groups"]["low"], r["groups"]["high"], r["primary"], r["continuous"]
    rows = [["Leavers (guide, sell, leave)", lo["leavers"], hi["leavers"], ""],
            ["Stayers (guide, sell, stay)", f"{lo['stayers']:,}", f"{hi['stayers']:,}", ""],
            ["Leaver minus stayer alpha, % per year", f"{100 * lo['effect']:.1f}", f"{100 * hi['effect']:.1f}", ""],
            ["**Interaction d = (high-T effect) - (low-T effect), % per year**", "", "", f"**{100 * pr['coef']:.1f}**"],
            ["Standard error (cluster by announcement quarter)", "", "", f"({100 * pr['se']:.1f})"],
            ["One-sided p, wild cluster bootstrap, H5: d > 0", "", "", f"{pr['p_one']:.3f}"],
            ["Continuous version: Leaver x Z coefficient", "", "", f"{ct['coef']:.3f} ({ct['se']:.3f}), one-sided p {ct['p_one']:.2f}"],
            ["Observations; clusters", "", "", f"{pr['n']:,}; {pr['n_clusters']}"]]
    return md_table(["", "(1) Low T", "(2) High T", "(3) Difference"], rows)


def table_s5_summary():
    s = load("stage5_results.json")
    t = s["tests"]
    rows = []
    for k, label, key, sign in (("T1", "Test 1: returns, excess alpha low minus high T", "did", "< 0"), ("T2", "Test 2: exits, Z x O x S", "TOS", "< 0"),
                                ("T3", "Test 3: misses, Z x O x S x X", "TOSX", "> 0"), ("T4", "Test 4: leavers vs stayers, d", "primary", "> 0")):
        r = t[k]["result"]
        est = r["did"] if k == "T1" else (r["primary"]["coef"] if k == "T4" else r[key]["coef"])
        rows.append([label, sign, f"{est:+.3f}", f"{s['family_p'][k]:.3f}", f"{s['holm'][k]:.2f}"])
    return md_table(["Test", "H5 predicts", "Estimate", "One-sided p", "Holm-adjusted p"], rows)


def table_s6_sample():
    r = load("stage6_results.json")
    c, p = r["coverage"], r["power"]
    rows = [["CEO sale-days, any value", f"{c['sale_days_any']:,}"], ["Sale-days of at least $100,000", f"{c['sale_days_ge100k']:,}"],
            ["... with a transparency score T (strictly prior 10-K, not stale)", f"{c['with_T']:,}"],
            ["Primary sample (T and the outcome available)", f"{c['primary_sample']:,} sale-days from {c['firms']:,} firms"],
            ["Sale-days after guidance in the prior 180 days (G)", f"{c['G']:,}"], ["Sale-days followed by an exit within 365 days (X)", f"{c['X']:,}"],
            ["**Guide-sell-leave sale-days (G and X)**", f"**{c['GX']:,}**"], ["... of which guidance was optimistic (O and X)", f"{c['OX']:,}"],
            ["Standard error of the H5 coefficient (pp per sd of T)", f"{p['se']:.2f}"],
            ["Minimum detectable effect (one-sided 5%, 80% power), pp per sd of T", f"{p['mde']:.2f}"]]
    sim = p["sim"]
    rows.append(["Simulated rejection rate at planted effects (pp per sd): " + ", ".join(f"{float(k):.1f}" for k in sim),
                 ", ".join(f"{100 * v:.0f}%" for v in sim.values())])
    miss = c["missing_T_reasons"]
    rows.append(["Sale-days of at least $100,000 with no T, by reason", ", ".join(f"{k or 'scored'}: {v:,}" for k, v in miss.items() if k)])
    return md_table(["", "Count or value"], rows)


def table_s6():
    r = load("stage6_results.json")
    s = r["secondary"]
    specs = [("(1) Primary, h=63", r["primary"]), ("(2) O in place of G", s["S2_optimistic_O"]), ("(3) High-T dummy for Z", s["S3_high_T_dummy"]),
             ("(4) h=21", s["S4_h21"]), ("(5) h=126", s["S4_h126"]), ("(6) h=252", s["S4_h252"]), ("(7) Sales >= $1M", s["S5_ge1m"]),
             ("(8) All sale values", s["S5_all_values"]), ("(9) First sale-day per quarter", s["S6_first_per_quarter"]),
             ("(10) 2004-14", s["S7_2004_2014"]), ("(11) 2015-24", s["S7_2015_2024"])]
    var = [("G", "Guided in prior 180 days (G)"), ("X", "Leaves within 365 days (X)"), ("GX", "G x X"), ("Z", "Transparency (Z)"), ("ZG", "Z x G"),
           ("ZX", "Z x X"), ("ZGX", "**Z x G x X**"), ("lmcap", "Log market cap"), ("ret12", "12-month return"), ("lval", "Log sale value"),
           ("ltenure", "Log tenure"), ("age", "Age")]
    k = len(specs)
    extra = [("Firm fixed effects", ["Yes"] * k), ("Year-month fixed effects", ["Yes"] * k), ("Observations (sale-days)", n_row(specs)),
             ("Guide-sell-leave sale-days (G x X)", [f"{r_['n_GX']:,}" for _, r_ in specs]),
             ("One-sided p, Z x G x X < 0", [f"{r_['p_one']:.2f}" for _, r_ in specs]),
             ("Minimum detectable Z x G x X", [f"{r_['mde']:.1f}" for _, r_ in specs])]
    return reg_table(specs, var, extra, d=2)


def table_s6_s1():
    r = load("stage6_results.json")["secondary"]["S1_all_sales_Z_only"]
    specs = [("Avoided loss on transparency, all sales (S1)", r)]
    var = [("Z", "Transparency (Z)"), ("lmcap", "Log market cap"), ("ret12", "12-month return"), ("lval", "Log sale value"), ("ltenure", "Log tenure"), ("age", "Age")]
    extra = [("Firm and year-month fixed effects", ["Yes"]), ("Observations (sale-days)", n_row(specs)), ("One-sided p, Z < 0", [f"{r['p_one']:.2f}"])]
    return reg_table(specs, var, extra, d=2)


def table_s6_mod():
    r = load("stage6_moderators.json")
    specs = [("(1) Primary, same sample", r["primary_same_sample"]), ("(2) + litigation-prone industry (L)", r["L"]),
             ("(3) + volatility (V)", r["V"]), ("(4) + both", r["L_and_V"])]
    var = [("G", "Guided (G)"), ("X", "Leaves within 365 days (X)"), ("GX", "G x X"), ("Z", "Transparency (Z)"), ("ZG", "Z x G"), ("ZX", "Z x X"),
           ("ZGX", "**Z x G x X**"), ("L", "L"), ("LG", "L x G"), ("LX", "L x X"), ("LGX", "**L x G x X**"),
           ("V", "V"), ("VG", "V x G"), ("VX", "V x X"), ("VGX", "**V x G x X**")]
    k = len(specs)
    extra = [("Controls (market cap, 12-month return, sale value, tenure, age)", ["Yes"] * k), ("Firm and year-month fixed effects", ["Yes"] * k),
             ("Observations (sale-days)", n_row(specs)), ("One-sided p, Z x G x X < 0", [f"{x['p_one']:.2f}" for _, x in specs])]
    return reg_table(specs, var, extra, d=2)


TABLES = {"s6_mod": table_s6_mod, "s6_sample": table_s6_sample, "s6": table_s6, "s6_s1": table_s6_s1, "stage1": table_stage1, "stage2": table_stage2, "p1": table_p1, "p2": table_p2, "p3": table_p3, "t1": table_t1, "t2": table_t2,
          "t3": table_t3, "t4": table_t4, "s5_summary": table_s5_summary}

if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or TABLES):
        print(f"\n=== {name}\n" + TABLES[name]())
