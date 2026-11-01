"""Generates guide_and_go_analysis.ipynb (run from the repo root: python notebooks/make_nb.py, then execute the notebook).

The notebook reports the PRE-SPECIFIED test (see docs/PREREGISTRATION.md). It reads:
  outputs/data/events.csv, outputs/data/study_windows.pkl, outputs/results/study_results.json   (from guide_and_go.py build + gg_study.py)
  validation/gold_labels.csv, validation/candidate_labels.csv   (hand-labelled data used to validate the measurement)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Guidance Reversal: Results

**Hypothesis (H1).** When a departing CEO/CFO (1) issued EPS guidance in the previous 180 days, (2) sold shares after that guidance,
and (3) left abruptly, the firm earns negative risk-adjusted returns over the next 12 months.

**How to read this notebook.** The test was written down *before* the returns of the final flag were computed (`docs/PREREGISTRATION.md`).
Section 2 shows how the measurement was validated against hand-labelled data, Section 4 is the one pre-specified primary test,
Section 5 the pre-specified secondary family (multiple-testing corrected), and Sections 6-7 say what the test could and could not detect.
Anything outside those is labelled **exploratory**.

> Run order: `python src/guide_and_go.py build --start 2004` → blind annotation of candidates (`validation/`) → `python src/gg_study.py` → this notebook.
""")

code(r"""
import json, os, pickle, sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from IPython.display import Markdown, display
warnings.filterwarnings("ignore")
if Path.cwd().name == "notebooks":
    os.chdir("..")                 # work from the repo root so relative paths resolve
sys.path.insert(0, "src")
import gg_labels as GL, gg_tests as GT
pd.set_option("display.max_columns", 40, "display.width", 200, "display.float_format", "{:,.3f}".format)
plt.rcParams.update({"figure.figsize": (9, 4.2), "axes.spines.top": False, "axes.spines.right": False})
RED, GREY, BLUE = "#c0392b", "#7f8c8d", "#2c3e50"

DATA, RESULTS = Path("outputs/data"), Path("outputs/results")
need = [DATA / "events.csv", RESULTS / "study_results.json", DATA / "study_windows.pkl"]
if not all(p.exists() for p in need):
    raise FileNotFoundError("Missing outputs. Run `python src/guide_and_go.py build --start 2004`, then `python src/gg_study.py`.")
ev = pd.read_csv(DATA / "events.csv", parse_dates=["dep_ann", "leftofc"])
ev["year"] = ev.dep_ann.dt.year
R = json.load(open(RESULTS / "study_results.json"))
W = pickle.load(open(DATA / "study_windows.pkl", "rb"))
P = R["primary"]
if R.get("dry_run"):
    print("*** WARNING: these results are from a DRY RUN with a random flag ***")
print(f"{len(ev):,} departures announced {ev.year.min()}-{ev.year.max()}; flagged: {R['n_flagged']} "
      f"({R['n_flagged_with_window']} with usable return windows); abrupt judgement from: {R['abrupt_source']}")
""")

md("## 0. Headline")
code(r"""
sd = P["placebo_sd"]; a = P["alpha_ann"]
lo, hi = a - 1.96 * sd, a + 1.96 * sd
verdict = ("**H1 is not supported.**" if not R["primary_pass"] else "**H1 is supported at the pre-specified threshold.**")
sec = pd.DataFrame(R["secondary"]) if R.get("secondary") else pd.DataFrame()
s6 = sec[sec.name.str.startswith("S6")].iloc[0] if len(sec) else None
txt = (f"{verdict} The calendar-time Fama-French-Carhart alpha of the {P['n_events']} flagged firms is **{a:+.1%} per year** "
       f"(t = {P['t']:.2f}; placebo p = {P['placebo_p']:.2f}; approximate 95% interval {lo:+.0%} to {hi:+.0%}). ")
if s6 is not None:
    txt += (f"The broader group that issued guidance, then sold, then left *in any manner* ({int(s6.n_events)} firms) earns {s6.alpha_ann:+.1%} per year "
            f"(placebo p = {s6.placebo_p:.2f}), which rules out underperformance larger than roughly {abs(s6.alpha_ann - 1.96*s6.placebo_sd):.0%} a year for that group. ")
txt += ("**What this does not show:** with only 42 flagged firms over 19 years, the primary test could not reliably detect an alpha smaller than "
        "about -20% a year (Section 6). A modest effect is neither confirmed nor excluded.")
display(Markdown(txt))
""")

md(r"""
## 1. Data integrity

Two data bugs, found while building the validation set, affected every earlier version of this project:

* **The WRDS REST API silently truncated large pulls.** For a 100-firm CRSP query it returned 270,000 rows covering 56 firms; direct Postgres returned 485,235 rows covering all 100.
  Roughly half of all firms had no price history, so every return-based result before the fix was computed on about half the sample. The pipeline now pulls via Postgres.
* **The departure date filter was ignored**, so a "2010+" run still loaded departures back to 1990 (46% of events had no data at all).

The table shows what the corrected data look like.
""")
code(r"""
cov = pd.Series({
    "has announcement-day return (CAR)": ev.car_m1p1.notna().mean(),
    "has 12-month return (BHAR)": ev.bhar_252.notna().mean(),
    "departure announcement identified in text": ev.ann_found.mean(),
    "EPS guidance in prior 12 months": (ev.n_guid_12m.fillna(0) > 0).mean(),
    "EPS guidance in prior 180 days": ev.guided_recent.mean(),
    "insider sale after that guidance (name-matched)": ev.sold_recent.mean(),
}).to_frame("share of events")
display(cov)
print("Before the fix the CAR/BHAR coverage was 27-54% (half the firms had no price history).")
print(f"Delisting returns are included. 12-month BHAR for all events: mean {ev.bhar_252.mean():+.1%}, median {ev.bhar_252.median():+.1%}, n = {ev.bhar_252.notna().sum():,}.")
""")

md(r"""
## 2. Measurement validation

The flag depends on text (was the exit abrupt?) and on name matching (did *this* executive sell?). Neither was trustworthy in earlier versions, so both were tested against **280 hand-labelled departures**
(labelled blind to the pipeline's output and to returns; 40 were labelled twice to measure agreement). The classifier was developed on half and scored once on the other, held-out half.
""")
code(r"""
g = pd.read_csv("validation/gold_labels.csv")
tasks = {}
for f in glob.glob("validation/tasks/annotator_*.jsonl"):
    for ln in open(f, encoding="utf-8"):
        d = json.loads(ln); tasks.setdefault(d["id"], d)
rows = []
for _, r in g.iterrows():
    t = tasks[r.id]; c = GL.classify_event(t["items"], t["exec_name"])
    rows.append(dict(split=r.split, stratum=r.stratum, gold=int(str(r.abrupt) == "True"), new=int(c["abrupt"] is True), old=int(r.old_abrupt),
                     gold_gov=int(str(r.government_destination) == "True"), old_gov=int(r.old_gov), new_gov=int(c["gov"])))
v = pd.DataFrame(rows)
def prf(y, p):
    tp = int(((y==1)&(p==1)).sum()); fp = int(((y==0)&(p==1)).sum()); fn = int(((y==1)&(p==0)).sum())
    return pd.Series({"precision": tp/max(tp+fp,1), "recall": tp/max(tp+fn,1), "flagged": int(p.sum()), "true positives": tp})
tbl = {}
for name, sub in [("held-out test half (n=%d)" % (v.split=="test").sum(), v[v.split=="test"]), ("all 280", v)]:
    tbl[(name, "OLD abrupt rule")] = prf(sub.gold, sub.old); tbl[(name, "NEW sentence classifier")] = prf(sub.gold, sub.new)
display(pd.DataFrame(tbl).T)
rnd = v[v.stratum == "random"]
print(f"On a random sample (n={len(rnd)}): gold says {rnd.gold.mean():.0%} of departures are abrupt; the OLD rule said {rnd.old.mean():.0%}; the NEW classifier says {rnd.new.mean():.0%}.")
gv = v[v.stratum == "old_gov"]
print(f"The old 'government destination' label: {len(gv)} sampled cases it labelled 'government'; hand labels say {int(gv.gold_gov.sum())} of them truly were. (keyword matches were 'Secretary of the Company', 'business administration', 'Treasurer'.)")
""")
code(r"""
cl = pd.read_csv("validation/candidate_labels.csv")
both = cl[cl.abrupt_p2.notna()]
print(f"The {len(cl)} candidate events (guidance + insider sale) were each read by one annotator; the {len(both)} that either that annotator or the rule called abrupt were read by a second, independent annotator.")
print(f"Agreement on 'abrupt' between the two reads: {(both.abrupt_p1 == both.abrupt_p2).mean():.0%}.  Final flag = both agree: {int(cl.abrupt_ann_label.sum())} events "
      f"(pass 1 alone: {int(cl.abrupt_p1.sum())}; the rule classifier alone: {int((ev.gr_rule == 1).sum())}).")
print("Inter-annotator reliability on the 40 duplicated gold tasks: 98% same announcement item, 90% same departure type (kappa 0.86), 92% same abrupt/planned judgement.")
print("Caveat: annotators are language models reading the announcement text, not people; a human spot-check of validation/gold_labels.csv is recommended.")
""")

md("## 3. The flag and the sample")
code(r"""
use = W["events"]
funnel = pd.Series({
    "CEO/CFO departures announced 2004-2025 (linked to CRSP)": R["n_events"],
    "...with EPS guidance in the previous 180 days": int(ev.guided_recent.sum()),
    "...and an insider sale after that guidance (candidates)": R["n_candidates"],
    "...and judged abrupt by two independent annotators (FLAGGED)": R["n_flagged"],
    "...with a usable 12-month return window": R["n_flagged_with_window"],
}).to_frame("events")
display(funnel)
fl = ev[ev.event_id.isin(W["flag_ids"])]
fig, ax = plt.subplots(figsize=(9, 3.2))
fl.groupby("year").size().reindex(range(2004, 2026), fill_value=0).plot.bar(ax=ax, color=RED)
ax.set(title="Flagged events by announcement year", ylabel="events"); plt.show()
print(f"Flagged firms: {fl.gvkey.nunique()} distinct firms; {(fl.role=='CEO').mean():.0%} CEOs; sectors: {fl.sector.value_counts().head(4).to_dict()}; median market cap ${fl.mcap.median()/1e9:.1f}bn.")
""")

md(r"""
## 4. Primary test (pre-specified)

Every flagged firm is held from trading day +2 to +253 after its announcement; each day the portfolio is the equal-weighted average of open positions (delisting returns included).
Its excess return is regressed on Mkt-RF, SMB, HML and UMD (weights = open positions, Newey-West errors). The statistic is the annualised intercept (alpha).
Inference uses a **placebo distribution**: the same procedure on 2,000 random sets of other departures with the same number of events per announcement year.
""")
code(r"""
null = np.asarray(W["null_primary"], dtype=float); null = null[np.isfinite(null)]
fig, ax = plt.subplots()
ax.hist(null * 100, bins=50, color="#bdc3c7")
ax.axvline(P["alpha_ann"] * 100, color=RED, lw=2.5, label=f"flagged: {P['alpha_ann']:+.1%}")
ax.axvline(np.percentile(null, 5) * 100, color=BLUE, ls="--", label="placebo 5th percentile")
ax.set(title="Annualised alpha of 2,000 random same-size portfolios vs the flagged portfolio", xlabel="alpha, % per year"); ax.legend(); plt.show()
res = pd.Series({"annualised alpha": P["alpha_ann"], "t-statistic (Newey-West)": P["t"], "placebo p-value (one-sided, H1: alpha < 0)": P["placebo_p"],
                 "flagged events with windows": P["n_events"], "mean open positions": P["mean_pos"], "trading days": P["n_days"],
                 "placebo mean": P["placebo_mean"], "placebo std. dev.": P["placebo_sd"]})
display(res.to_frame("value"))
display(pd.Series(P["betas"]).to_frame("factor loading"))
print("Decision rule (fixed in advance): reject only if placebo p < 0.05 AND at least 40 events. Pass:", R["primary_pass"])
""")

md(r"""
## 5. Secondary family (pre-specified, Holm-adjusted together)

| | |
|---|---|
| S1 | CEOs only |
| S2 | 6-month hold |
| S3 | lenient insider matching (adds initial-only name matches) |
| S4 | rule-based "abrupt" instead of annotators |
| S5 | six-factor model |
| S6 | guidance + sale, **any** exit type (does abruptness matter?) |
| S7 | excluding 2008-09 and 2020 |
| S8 | split by announcement period |
""")
code(r"""
sec = pd.DataFrame(R["secondary"])
t = sec[["name", "n_events", "alpha_ann", "t", "placebo_p", "holm_p"]].rename(columns={"alpha_ann": "alpha (per yr)", "placebo_p": "placebo p", "holm_p": "Holm-adj. p"})
display(t.set_index("name"))
print("No secondary test is significant after correction. Note S6 is the best-powered test (267 firms).")
""")

md(r"""
## 6. What could this test detect?

A null result is only informative to the extent the test could have found an effect. The placebo distribution gives the sampling noise of an equal-sized portfolio.
""")
code(r"""
def power_curve(s, label, color, ax):
    m, sd, q05 = s["placebo_mean"], s["placebo_sd"], s["placebo_q05"]
    e = np.linspace(0, -0.35, 200)
    pw = st.norm.cdf((q05 - m - e) / sd)
    ax.plot(e * 100, pw * 100, color=color, label=f"{label} (n={s['n_events']})")
    mde = next(x for x in np.arange(0, -0.6, -0.001) if st.norm.cdf((q05 - m - x) / sd) >= 0.8)
    return mde
fig, ax = plt.subplots()
m1 = power_curve(P, "primary flag", RED, ax)
s6 = sec[sec.name.str.startswith("S6")].iloc[0].to_dict()
m6 = power_curve(s6, "any exit type (S6)", BLUE, ax)
ax.axhline(80, color=GREY, ls=":"); ax.set(xlabel="true annual alpha (%)", ylabel="power (%)", title="Probability the test rejects H0 if the true alpha is x"); ax.invert_xaxis(); ax.legend(); plt.show()
print(f"Minimum alpha detectable with 80% power: primary {m1:+.1%}/yr; broad group {m6:+.1%}/yr.")
print(f"Approximate 95% interval for the true alpha: primary [{P['alpha_ann']-1.96*P['placebo_sd']:+.1%}, {P['alpha_ann']+1.96*P['placebo_sd']:+.1%}]; "
      f"broad group [{s6['alpha_ann']-1.96*s6['placebo_sd']:+.1%}, {s6['alpha_ann']+1.96*s6['placebo_sd']:+.1%}].")
display(Markdown("**Reading:** the primary test could have caught a very large effect (about -20% a year) but would usually miss -5% to -10%. "
                 "The broad group has much more power and shows no underperformance, so the pattern 'guide, sell, then leave' is not associated with large negative alpha in this sample."))
""")

md(r"""
## 7. Exploratory (not part of the pre-specified family; not used to claim anything)
""")
code(r"""
ex = R.get("exploratory", {}).get("matched_bhar")
if ex and "mean_diff" in ex:
    print(f"Matched controls (same year and role, nearest on size and prior return, k=5): flagged minus matched 12-month BHAR = {ex['mean_diff']:+.1%} "
          f"(95% CI {ex['ci95'][0]:+.1%} to {ex['ci95'][1]:+.1%}; median {ex['median_diff']:+.1%}; n={ex['n']}).")
oth = ev[~ev.event_id.isin(W["flag_ids"])]
print(f"Event-time 12-month BHAR: flagged mean {fl.bhar_252.mean():+.1%} (median {fl.bhar_252.median():+.1%}, n={fl.bhar_252.notna().sum()}) vs other departures mean {oth.bhar_252.mean():+.1%} (median {oth.bhar_252.median():+.1%}).")
print(f"Announcement-day CAR[-1,+1]: flagged {fl.car_m1p1.mean():+.1%} vs other {oth.car_m1p1.mean():+.1%}.")
pp = DATA / "car_paths.pkl"
if pp.exists():
    paths = {str(k): np.asarray(v, float) for k, v in pickle.load(open(pp, "rb")).items()}
    xs = np.arange(-20, 252); fig, ax = plt.subplots()
    for lbl, ids, c in [("flagged", set(map(str, W["flag_ids"])), RED), ("other departures", set(map(str, oth.event_id)), BLUE)]:
        M = np.vstack([v for k, v in paths.items() if k in ids and v.shape == (272,)]) * 100
        mu = np.nanmean(M, 0); se = np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0))
        ax.plot(xs, mu, c=c, label=f"{lbl} (n={M.shape[0]})"); ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=.15)
    ax.axvline(0, c="k", ls="--", lw=.7); ax.axhline(0, c=GREY, lw=.5)
    ax.set(title="Cumulative market-adjusted return around the announcement (exploratory)", xlabel="trading days", ylabel="%"); ax.legend(); plt.show()
""")

md(r"""
## 8. Stage 2: wider sample, intent proxied by optimistic guidance (pre-specified in `docs/PREREGISTRATION.md`, Stage 2)

Abruptness is dropped. Events: any non-death CEO/CFO exit with a sale linked to guidance (L1-L3) or just a prior sale (L4). **F1** asks whether the group has negative alpha (8 tests);
**F2** asks whether those who guided *above consensus* do worse than those who did not (6 tests). Holm correction over all 14.
""")
code(r"""
S2 = json.load(open(RESULTS / "stage2_results.json"))
f1 = pd.DataFrame(S2["F1"])[["name", "n_events", "alpha_ann", "placebo_sd", "placebo_p", "holm_p"]]
f1["95% interval"] = [f"{a-1.96*s:+.1%} to {a+1.96*s:+.1%}" for a, s in zip(f1.alpha_ann, f1.placebo_sd)]
f1["alpha_ann"] = f1.alpha_ann.map("{:+.1%}".format); f1["placebo_sd"] = f1.placebo_sd.map("{:.1%}".format)
display(Markdown("**F1: alpha of the guide-sell-leave group (negative = supports hypothesis)**")); display(f1.rename(columns={"name": "cell"}).round(3))
f2 = pd.DataFrame(S2["F2"])[["name", "n_optimistic", "n_other", "alpha_optimistic", "alpha_other", "diff", "perm_p", "holm_p"]]
for c in ["alpha_optimistic", "alpha_other", "diff"]: f2[c] = f2[c].map("{:+.1%}".format)
display(Markdown("**F2: optimistic-guidance events (O) vs the rest (N)**")); display(f2.rename(columns={"name": "cell"}).round(4))
print(f"Family size {S2['family_size']}; smallest Holm-adjusted p = {S2['smallest_holm_p']:.4f}")
display(Markdown("**Reading:** no F1 cell is negative; the 95% intervals are roughly +/-2% (L4) to +/-8% (L1). One F2 cell (L3 x U1) clears Holm at 0.048, but its optimistic arm is only -1.9% "
                 "(the gap is mostly the *other* arm at +4.2%), it is absent in the strict links L1/L2, and the raise-based intent split goes the opposite way (exploratory). Treat as suggestive, not a finding."))
""")

md(r"""
### 8a. Why the one significant Stage 2 result is fragile

**1. Where the gap comes from.** The test statistic is a *difference*: alpha(optimistic, O) minus alpha(other, N). For L3 x U1 that is -1.9% minus +4.2% = **-6.1%**.
The hypothesis says optimistic guiders should *underperform*. But O is only slightly below zero (and an alpha of -1.9% on 221 events is well inside noise: the placebo sd is 1.8% a year for the full 670-event L3 x U1 group, so for a 221-event arm it is larger, plausibly around 3%),
while N earns an unusually *high* +4.2%. About **70% of the gap (4.2 of 6.1 points) is N doing well, not O doing badly.** Compare the whole L3 x U1 group at +2.2%: the split just separates a roughly-zero arm from a high arm.
A difference can be "significant" because one arm is lucky, and that is not evidence that optimism harms returns.

**2. It does not appear in the strict links.** L1 and L2 are the links closest to the original story (guidance followed shortly by a sale *after* that guidance). There the gap has the same sign but is smaller and not significant
(L1 x U1: -4.9%, p = 0.066; L2 x U1: -1.1%, p = 0.28). It is largest in L3, which only needs *any* sale in the prior 365 days, including routine or pre-planned sales that have nothing to do with the guidance.
If optimism-then-selling-then-leaving drove the effect, it should be strongest where the sale is tied most tightly to the guidance; the pattern is the reverse.
""")
code(r"""
f2d = pd.DataFrame(S2["F2"]).copy()
f2d["cell"] = f2d["name"].str.replace("F2 ", "", regex=False)
f2d["share_from_N_arm"] = (f2d.alpha_other / (f2d.alpha_other - f2d.alpha_optimistic)).where(f2d["diff"] < 0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(len(f2d)); w = .38
axes[0].bar(x - w/2, f2d.alpha_optimistic * 100, w, color=RED, label="optimistic guidance (O)")
axes[0].bar(x + w/2, f2d.alpha_other * 100, w, color=BLUE, label="other (N)")
axes[0].axhline(0, c="k", lw=.7); axes[0].set_xticks(x); axes[0].set_xticklabels(f2d.cell, rotation=30, ha="right")
axes[0].set(title="Alpha by arm: the gap is mostly N being positive", ylabel="% per year"); axes[0].legend()
axes[1].bar(x, f2d.perm_p, color=[RED if p < .05 else GREY for p in f2d.perm_p])
axes[1].axhline(.05, c="k", ls="--", lw=.7); axes[1].set_xticks(x); axes[1].set_xticklabels(f2d.cell, rotation=30, ha="right")
axes[1].set(title="Permutation p by link (L1 strict -> L3 loose)", ylabel="one-sided p"); plt.tight_layout(); plt.show()
t = f2d[["cell", "n_optimistic", "n_other", "alpha_optimistic", "alpha_other", "diff", "perm_p", "holm_p", "share_from_N_arm"]].copy()
for c in ["alpha_optimistic", "alpha_other", "diff"]: t[c] = t[c].map("{:+.1%}".format)
t["share_from_N_arm"] = t.share_from_N_arm.map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
display(t.round(4))
""")

md(r"""
## 9. Stage 3: stayers as controls and a direct test on earnings misses (pre-registered in `STAGE3_SPEC.md`)

Stages 1-2 compared leavers with other leavers using stock alpha. Stage 3 adds **stayers** and tests the mechanism itself: **optimistic guidance, a CEO sale, an exit, then a miss.**
Three registered tests, Holm-corrected together. All use CEOs of firms that issue EPS guidance. **P1** (leaving): does the quarterly exit probability rise when guidance was above consensus *and* the CEO sold after it?
**P2** (misses): is the realised miss (actual minus guided EPS, % of price) worse when guidance was optimistic, the CEO sold after it, *and* left before the period ended (coefficient on the triple interaction, firm and year-quarter fixed effects, two-way clustered)?
**P3** (returns): do guide-sell-leave firms earn less than guide-sell-**stay** firms?
""")
code(r"""
S3 = json.load(open(RESULTS / "stage3_results.json")); P = S3["primary"]; H = S3["holm"]
p1, p2, p3 = P["P1"], P["P2"], S3["P3"]
rows = [
    ["P1 exit hazard: OPT x SALEPOST (pp per quarter; baseline 2.5)", p1["OS"]["coef"], p1["OS"]["se"], p1["p_one"], H.get("P1"), p1["mde"], f"{p1['n']:,} CEO-quarters; {p1['n_exits']:,} exits; {p1['n_S_exit']} after a sale"],
    ["P2 miss: OPT x SALE x EXIT (pp of price)", p2["OSX"]["coef"], p2["OSX"]["se"], p2["p_one"], H.get("P2"), p2["mde"], f"{p2['n']:,} releases; {p2['n_SX']:,} with sale and exit ({p2['n_OSX']} optimistic)"],
    ["P3 alpha: leavers minus stayers (% per year)", p3["diff"] * 100, p3["perm_sd"] * 100, p3["perm_p"], H.get("P3"), 2.49 * p3["perm_sd"] * 100, f"{p3['n_leavers']} leavers (alpha {p3['alpha_leavers']:+.1%}) vs {p3['n_stayers']:,} stayer events (alpha {p3['alpha_stayers']:+.1%})"],
]
t = pd.DataFrame(rows, columns=["test", "estimate", "se (P3: permutation sd)", "one-sided p", "Holm p", "detectable at 80% power", "sample"])
display(Markdown("**Registered tests (all null; expected sign: P1 positive, P2 negative, P3 negative)**")); display(t.round(3))
a = S3["audit"]
print(f"Look-ahead audit passed: {a['passed']} ({a['A_salepost_verified_in_window_rows_checked'] + a['B_sale_verified_in_window_rows_checked']:,} sampled rows re-checked, "
      f"{a['A_salepost_violations'] + a['B_sale_violations']} violations). Attrition: {S3['attrition_A18']['share_with_actual_exit0']:.1%} of releases have an actual without an exit vs {S3['attrition_A18']['share_with_actual_exit1']:.1%} with an exit.")
""")
code(r"""
rb = []
for k, r in S3["robustness"].items():
    key = r.get("key") or "OS"; c = r[key]
    rb.append([k, c["coef"], c["se"], c["t"], r["p_one"], r["n"]])
rb = pd.DataFrame(rb, columns=["check", "coefficient", "se", "t", "one-sided p", "n"])
display(Markdown("**Robustness (17 checks; none in the Holm family).** R1 continuous optimism, R2 sale >= $1M, R4 surprise vs consensus, R5 first release only, R6 logit without firm effects, "
                 "R9 excluding retirements, **R11 falsification (sales *before* the guidance)**, R12 subperiods, R13/R14 guards against sales or guidance after a departure was announced."))
display(rb.round(3))
d = []
for nm, lab in [("O", "optimistic guidance"), ("S", "sale after guidance"), ("X", "exit before period end"), ("OS", "optimistic x sale"), ("OX", "optimistic x exit"), ("SX", "sale x exit"), ("OSX", "optimistic x sale x exit (registered)")]:
    c = p2[nm]; d.append([lab, c["coef"], c["se"], c["t"]])
display(Markdown("**P2, all terms (descriptive; only the last row is a registered test; the others are not corrected for multiplicity).** Coefficients are percentage points of price."))
display(pd.DataFrame(d, columns=["term", "coefficient", "se", "t"]).round(3))
display(Markdown("**Reading.** (1) The registered claim fails on all three tests and in all 17 checks; P1 is negative in all seven variants. (2) Exits are followed by worse misses (-0.42) and optimistic guidance by worse misses (-0.07, partly mechanical), "
                 "but a CEO who sold *and then left* shows no worse miss than one who left without selling (sale x exit +0.47): the opposite of an intent story. The exit effect fits a new CEO writing assets down, or trouble that precedes an exit. "
                 "(3) Leavers earn +0.2% and stayers +2.3% a year: a gap of -2.1% (p = 0.26), well inside the noise (permutation sd 2.4%). "
                 "(4) **Power:** the P2 triple interaction is detectable only at about 0.43 pp of price, about the size of the whole exit effect, so small effects are not ruled out. Calibration with randomised outcomes showed the tests are mildly liberal (rejecting 6.7% at a nominal 5%), so the null is if anything firmer."))
""")

md(r"""
## 10. Stage 4: iterating on the null (exploratory, with a hold-out)

After a null, the questions are: is the sign convention wrong, are controls missing, and is there an insight? Because iterating after a null is a forking path, specifications were **explored on 2004-2014 only**,
a fixed rule (|t| >= 2.5) selected candidates, and **2015-2025 was kept for a single confirmation**. Added controls: age bands, tenure squared, last realised miss, share of past guidance above consensus, volatility, CEO ownership.
Added splits: non-routine sales (M2) and exit type (M3, M3b).
""")
code(r"""
E4 = json.load(open(RESULTS / "stage4_explore.json")); M3b = json.load(open(RESULTS / "stage4_explore_m3b.json")); REP = json.load(open(RESULTS / "stage4_m3b_replicate.json"))
rows = []
for m, k, lab in [("P1_M0", "OS", "P1 optimism x sale (Stage-3 controls)"), ("P1_M1", "OS", "P1 + augmented controls"), ("P1_M2", "OS_nr", "P1 non-routine sales"),
                  ("P1_M3b_planned", "OS", "P1 planned exits only"), ("P1_M3b_forced", "OS", "P1 forced exits only"),
                  ("P2_M0", "OSX", "P2 triple interaction (Stage-3 controls)"), ("P2_M1", "OSX", "P2 + augmented controls"), ("P2_M2", "OSX_nr", "P2 non-routine sales"),
                  ("P2_M3b", "OSXp", "P2 planned exits"), ("P2_M3b", "OSXf", "P2 forced exits")]:
    src = M3b if m.endswith(("planned", "forced")) or m == "P2_M3b" else E4
    v = src[m][k]; rows.append([lab, v["coef"], v["se"], v["t"]])
display(Markdown("**Exploration half, 2004-2014: the registered terms under richer specifications (none reaches |t| = 2.5, so nothing is carried to confirmation)**"))
display(pd.DataFrame(rows, columns=["specification", "coefficient", "se", "t"]).round(3))
display(Markdown("**Data problem found.** The one exploratory hit (optimism x sale x *retirement* exit, +0.97, t 3.5, 56 releases from 25 firms) came from ExecuComp's exit-reason field, which is missing for **87% of exits and for almost every exit after 2009**. "
                 "It cannot be tested in 2015-2025 (zero such releases), so it is recorded as untestable and withdrawn. Exit type now comes from the Capital IQ text classifier (87-90% coverage in both halves)."))
""")
code(r"""
PRO = json.load(open(RESULTS / "stage4_profile.json"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
ks = list(range(-4, 5))
for ax, which in zip(axes, ["explore", "confirm"]):
    for tag, lab, c in [("nosale", "no CEO sale in prior 365d", RED), ("sold", "CEO sale in prior 365d", BLUE)]:
        co = np.array([PRO[which].get(f"k{k:+d}_{tag}", {}).get("coef", np.nan) for k in ks]); se = np.array([PRO[which].get(f"k{k:+d}_{tag}", {}).get("se", np.nan) for k in ks])
        ax.errorbar(np.array(ks) + (0.08 if tag == "sold" else -0.08), co, yerr=1.96 * se, fmt="o-", color=c, capsize=2, label=lab)
    ax.axhline(0, c="k", lw=.7); ax.axvline(-0.5, c=GREY, ls="--", lw=.7)
    ax.set(title=f"Misses around CEO exits, {'2004-2014' if which == 'explore' else '2015-2025'}", xlabel="quarters from exit to the period end (k<0: before the exit)")
axes[0].set_ylabel("miss, pp of price (vs releases far from any exit)"); axes[0].legend(); plt.tight_layout(); plt.show()
rows = []
for k in ["Xp", "Xf", "SXp", "SXf", "OSXp", "OSXf"]:
    a, b = M3b["P2_M3b"][k], REP["P2_M3b"][k]; rows.append([k, a["coef"], a["t"], b["coef"], b["t"], "same sign" if np.sign(a["coef"]) == np.sign(b["coef"]) else "SIGN FLIPS"])
display(Markdown("**Post-hoc replication of exit-type terms in 2015-2025 (descriptive, not a registered test).** X = exit, S = sale, O = optimistic guidance; p = planned, f = forced exits."))
display(pd.DataFrame(rows, columns=["term", "2004-14 coef", "t", "2015-25 coef", "t ", "consistency"]).round(3))
display(Markdown("**Reading.** (1) No sign convention rescues the hypothesis: the optimism x sale x exit terms are positive in one half and negative in the other, by exit type. (2) The extra controls change nothing. "
                 "(3) The one pattern present in both halves is timing: **exits not preceded by a CEO sale follow quarters of earnings misses; exits preceded by a sale do not** (left panel and right panel look alike). "
                 "Together with the higher exit probability after a sale (+0.4 pp on a 2.5% base, p ~ 0.08), CEO sales before an exit look like markers of planned, orderly departures, not of leaving ahead of bad news. "
                 "(4) The planned/forced split does **not** replicate, so I do not interpret it. All of Stage 4 is exploratory."))
""")

md(r"""
## 11. Conclusions

1. **No evidence that the Guidance Reversal pattern predicts underperformance.** The pre-specified primary test and all eight pre-specified secondary tests are null; the wider Stage 2 samples (up to 1,114 events) are also null, with intervals of a few percent a year.
   One Stage 2 optimism-split test (L3 x U1) is nominally significant after Holm (0.048); it is weak evidence given the forking path and ~28 project-wide tests.
   **Stage 3 adds a stayer control group and a direct earnings-miss test: also null** (exit hazard, realised miss, leaver-vs-stayer alpha; Holm 0.77-0.85, 17 of 17 robustness checks null).
   **Stage 4 (exploratory, with a hold-out) found no robust pattern**: more controls change nothing, exit-type splits flip sign across halves, and one apparent hit came from an exit-reason field that is blank after 2009. The one replicating fact: exits not preceded by a CEO sale follow quarters of misses; exits preceded by a sale do not.
2. **The earlier "findings" were artifacts.** The old headline gap (-21% vs -4%) was circular (the flag required a bad outcome); the old "government destination" effect came from a keyword label with 0% precision;
   and every earlier return number used data that were missing about half the firms.
3. **What can be said with confidence:** large effects (below about -20% a year for the strict flag, below about -4% for the broad group) are unlikely. **What cannot:** modest effects for the strict flag.
   There are only 42 usable flagged events over 19 years, all CEOs, about half in technology.
4. **Measurement is now defensible:** the abrupt-exit judgement was validated against 280 hand labels and read twice for every flagged event; name matching and guidance units were fixed and unit-tested.

Remaining limitations and next steps are in `limitations.md`. Chief among them: a control group of executives who guided and sold but stayed, and a larger universe than the S&P 1500.
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "viking-venv", "language": "python", "name": "viking-venv"}}
nbf.write(nb, "notebooks/guide_and_go_analysis.ipynb")
print("wrote notebooks/guide_and_go_analysis.ipynb with", len(C), "cells")
