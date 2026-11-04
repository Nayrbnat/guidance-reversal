"""
gg_stage5: attach the transparency score T to events and panels, and (later tasks) the tests T1-T4 and the runner.
See docs/STAGE5_SPEC.md. T for a departure event comes from the last 10-K filed strictly before the announcement; for a
CEO-quarter, from the last 10-K filed on or before the quarter start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gg_filings as F


def score_meta(scores: pd.DataFrame) -> dict:
    """Mean and sd of T over unique scored filings (one row per accession), the single standardisation population."""
    t = scores.drop_duplicates("accession")["t_final"].dropna()
    return {"mu": float(t.mean()), "sd": float(t.std(ddof=1)), "n": int(len(t))}


def attach_T(df: pd.DataFrame, scores: pd.DataFrame, index: pd.DataFrame, meta: dict, *, date_col: str,
             gvkey_col: str = "gvkey", strict: bool = True, max_age_days: int | None = 550) -> pd.DataFrame:
    """Add accession, T_status, T (NaN when absent, never zero-filled), T_missing_reason (no_filing | stale_filing |
    extraction_failed | not_scored | ''), T_filing_age_days and T_z = (T - mu) / sd using the stored `meta`. Row order and
    index of `df` are preserved. `strict=True`: last 10-K filed strictly before the date (events); False: on or before
    (quarter starts). A filing older than `max_age_days` (default 550: an annual filer plus a late filing) is stale and
    scores NaN rather than silently carrying an old T across a link gap or a missed filing; None switches this off."""
    eff = F.effective_index(index)[["gvkey", "filing_date", "accession"]].copy()
    eff["filing_date"] = pd.to_datetime(eff["filing_date"]).astype("datetime64[ns]")
    eff = eff.sort_values(["filing_date", "accession"]).rename(columns={"gvkey": "_g"})
    dates = pd.to_datetime(df[date_col]).astype("datetime64[ns]").values
    left = pd.DataFrame({"_g": df[gvkey_col].astype(str).str.zfill(6).values, "_d": dates, "_i": np.arange(len(df))})
    valid = left[left["_d"].notna()].sort_values("_d")
    m = pd.merge_asof(valid, eff, left_on="_d", right_on="filing_date", by="_g", direction="backward",
                      allow_exact_matches=not strict)
    acc = pd.Series(np.nan, index=range(len(df)), dtype=object)
    acc.loc[m["_i"].values] = m["accession"].values
    fdate = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    fdate[m["_i"].values] = m["filing_date"].values.astype("datetime64[ns]")
    age = (dates - fdate) / np.timedelta64(1, "D")
    sc = scores.drop_duplicates("accession").set_index("accession")
    t = acc.map(sc["t_final"]).astype(float)
    status = acc.map(sc["status"])
    reason = pd.Series("", index=acc.index, dtype=object)
    reason[acc.isna()] = "no_filing"
    reason[acc.notna() & ~acc.isin(sc.index)] = "not_scored"
    reason[status == "extraction_failed"] = "extraction_failed"
    reason[(reason == "") & t.isna()] = "not_scored"
    if max_age_days is not None:
        reason[acc.notna() & (age > max_age_days)] = "stale_filing"
    t[reason != ""] = np.nan
    out = df.copy()
    out["accession"] = acc.values
    out["T_status"] = status.values
    out["T"] = t.values
    out["T_missing_reason"] = reason.values
    out["T_filing_age_days"] = age
    out["T_z"] = (out["T"] - meta["mu"]) / meta["sd"]
    return out


def attach_event_T(events: pd.DataFrame, ev_final: pd.DataFrame, scores: pd.DataFrame, index: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """T for the Stage 2 events, dated by `dep_ann_final`: the announcement date the return windows are built on (the annotator's
    date where one was found, the rule date otherwise). Using the raw rule date instead could pick a 10-K filed after the window
    starts. `ev_final` has event_id, gvkey, dep_ann_final; `ann_quarter` is taken from the same date."""
    ev = events.merge(ev_final[["event_id", "gvkey", "dep_ann_final"]], on="event_id", how="left")
    ev["gvkey"] = ev["gvkey"].astype(str).str.zfill(6)
    ev["ann_quarter"] = ev["dep_ann_final"].dt.to_period("Q").astype(str)
    return attach_T(ev, scores, index, meta, date_col="dep_ann_final", strict=True)


def coverage_report(df: pd.DataFrame, date_col: str) -> dict:
    """Counts per missing reason and, by calendar year, the share of rows with a score and the share of scored rows that
    were limited_disclosure filings; also the distribution of filing age (days between the filing and the date)."""
    year = pd.to_datetime(df[date_col]).dt.year
    scored = df["T_missing_reason"] == ""
    limited = (df["T_status"] == "limited_disclosure") & scored
    out = {"reasons": df["T_missing_reason"].value_counts().to_dict(),
           "scored_share_by_year": scored.groupby(year).mean().to_dict(),
           "limited_share_by_year": (limited.groupby(year).sum() / scored.groupby(year).sum().replace(0, np.nan)).to_dict()}
    if "T_filing_age_days" in df.columns and df["T_filing_age_days"].notna().any():
        a = df["T_filing_age_days"].dropna()
        out["filing_age_days"] = {"min": float(a.min()), "median": float(a.median()), "p90": float(a.quantile(0.9)), "max": float(a.max())}
    return out


def validation_report(scores: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """How T relates to business complexity, size, industry and year (construct validity, spec 2). `scores`: accession,
    t_final, status; `meta`: accession, n_segments, log_assets, sic_div, year."""
    from scipy.stats import spearmanr
    d = scores.drop_duplicates("accession").merge(meta, on="accession", how="inner").dropna(subset=["t_final"])
    out = {"n": int(len(d)),
           "spearman_segments": float(spearmanr(d["t_final"], d["n_segments"]).statistic),
           "spearman_log_assets": float(spearmanr(d["t_final"], d["log_assets"]).statistic),
           "mean_T_by_division": d.groupby("sic_div")["t_final"].mean().round(3).to_dict(),
           "mean_T_by_year": d.groupby("year")["t_final"].mean().round(3).to_dict(),
           "limited_share_by_division": (d["status"] == "limited_disclosure").groupby(d["sic_div"]).mean().round(3).to_dict()}
    X = pd.get_dummies(d[["sic_div", "year"]].astype(str), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    y = d["t_final"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    ss_res = float(((y - X.to_numpy() @ beta) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out["r2_industry_year"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


# ---------------------------------------------------------------- T1: calendar-time alpha by T group
def _t1_core(windows, events, fac, H, flagged_ids, cuts, reps, seed, n_boot, with_spread, strata):
    import gg_tests as GT
    import gg_transparency as T
    ev = events[events["event_id"].isin(windows.keys()) & events["T"].notna()].copy()
    ev["_grp"] = T.assign_group(ev["T"], cuts)
    ev["_flag"] = ev["event_id"].isin(flagged_ids)
    ev["_s"] = ev[list(strata)].astype(str).agg("|".join, axis=1)
    # Stage 2 drew placebos only from events whose announcement was identified (ann_found == 1); flagged events are not filtered
    ev["_pool"] = ~ev["_flag"] & (ev["ann_found"] == 1 if "ann_found" in ev.columns else True)
    via = T.group_viability(ev["T"], ev["_flag"], ev["_pool"], cuts)
    if not via["ok"]:
        return {"nan_reason": "; ".join(via["reasons"]), "groups": via["groups"]}
    part = {g: (ev[(ev["_grp"] == g) & ev["_flag"]], ev[(ev["_grp"] == g) & ev["_pool"]]) for g in ("low", "high")}
    obs, draws, series = {}, {}, {}
    for k, g in enumerate(("low", "high")):
        fl, pool = part[g]
        port, cnt = GT.portfolio(windows, fl["event_id"], H)
        series[g] = (port, cnt)
        obs[g] = GT.alpha_fast(port, cnt, fac)
        draws[g] = GT.placebo_draws(windows, fl, pool, fac, H, strata_col="_s", reps=reps, seed=seed + k)
    m = {g: float(np.nanmean(draws[g])) for g in draws}
    excess = {g: obs[g] - m[g] for g in obs}
    did = excess["low"] - excess["high"]
    f = {g: GT.fpc_factor(len(part[g][0]), len(part[g][1])) for g in ("low", "high")}
    null = f["low"] * (draws["low"] - m["low"]) - f["high"] * (draws["high"] - m["high"])
    out = {"groups": {g: {"n": int(len(part[g][0])), "alpha_ann": float(obs[g]), "placebo_mean": m[g],
                          "excess_ann": float(excess[g]), "pool": int(len(part[g][1])), "fpc": f[g]} for g in obs},
           "did": float(did), "perm_p": GT.placebo_p(did, null, "lower"), "cuts": list(cuts), "reps": int(reps),
           "mde": float((1.645 + 0.8416) * np.nanstd(null))}
    if n_boot > 0:
        rng = np.random.default_rng(seed + 99)
        qcol = "ann_quarter" if "ann_quarter" in ev.columns else "year"
        by_q = {g: part[g][0].groupby(qcol)["event_id"].apply(list).to_dict() for g in ("low", "high")}
        qs = sorted(set(by_q["low"]) | set(by_q["high"]))
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(len(qs), len(qs))
            a = {}
            for g in ("low", "high"):
                ids = [i for q in pick for i in by_q[g].get(qs[q], [])]
                port, cnt = GT.portfolio(windows, ids, H)
                a[g] = GT.alpha_fast(port, cnt, fac)
            boots.append((a["low"] - m["low"]) - (a["high"] - m["high"]))
        out["ci"] = [float(x) for x in np.nanpercentile(boots, [2.5, 97.5])]
    if with_spread:
        (pl, cl), (ph, ch) = series["low"], series["high"]
        res = GT.alpha_test(pl - ph, np.minimum(cl, ch), fac.assign(rf=0.0))
        out["spread_alpha"], out["spread_t"] = res["alpha_ann"], res["t"]
        out["spread_se"] = float(res["alpha_ann"] / res["t"]) if res["t"] else float("nan")
    return out


def run_t1(windows: dict, events: pd.DataFrame, fac: pd.DataFrame, H: int, flagged_ids, *, cuts, reps: int = 5000,
           seed: int = 0, broad_ids=None, strata_cols=("year",), n_boot: int = 1000, with_spread: bool = True,
           with_sens: bool = True) -> dict:
    """T1: do flagged firms earn a more negative alpha where T is low than where T is high?

    `events` has event_id, T, year, ann_quarter (and sic_div / size_tercile for the sensitivities). Primary statistic
    `did`: excess alpha of flagged firms (own alpha minus the mean alpha of same-size placebo portfolios from the SAME T
    group) at low T minus the same at high T; its p-value is the lower tail of the centred placebo contrast (null of equal
    effects). `ci` is a calendar-block bootstrap over announcement quarters. `spread_*` is the alpha of the low-minus-high
    calendar-time long-short series (Newey-West). Returns {"nan_reason": ...} when groups are not viable.
    """
    out = _t1_core(windows, events, fac, H, set(flagged_ids), cuts, reps, seed, n_boot, with_spread, strata_cols)
    if "nan_reason" in out:
        return out
    if broad_ids is not None:
        out["broad"] = _t1_core(windows, events, fac, H, set(broad_ids), cuts, reps, seed, 0, False, strata_cols)
    if with_sens:
        sens = {f"strata_year_{extra}": _t1_core(windows, events, fac, H, set(flagged_ids), cuts, reps, seed, 0, False,
                                                  tuple(strata_cols) + (extra,))
                for extra in ("sic_div", "size_tercile") if extra in events.columns}
        if sens:
            out["sens"] = sens
    return out


def run_t1_continuous(events: pd.DataFrame, outcomes: pd.Series, *, cluster: str = "ann_quarter") -> dict:
    """Corroboration for T1: event-level OLS of an abnormal-return outcome on flag, T_z and flag x T_z, standard errors
    clustered by announcement quarter. Supports H5 when flag x T_z is POSITIVE (more transparent, less negative)."""
    import statsmodels.api as sm
    d = events.assign(y=events["event_id"].map(outcomes)).dropna(subset=["y", "T_z", "flag"])
    d["flag"] = d["flag"].astype(float)
    X = sm.add_constant(pd.DataFrame({"flag": d["flag"], "T_z": d["T_z"], "flag_T_z": d["flag"] * d["T_z"]}))
    fit = sm.OLS(d["y"].to_numpy(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]}, use_t=True)
    coef, p2 = float(fit.params["flag_T_z"]), float(fit.pvalues["flag_T_z"])
    return {"coef": coef, "se": float(fit.bse["flag_T_z"]), "p_two": p2, "p_one": p2 / 2 if coef > 0 else 1 - p2 / 2,
            "n": int(len(d)), "n_clusters": int(d[cluster].nunique())}


# ---------------------------------------------------------------- T2 and T3: Stage 3 regressions with T
def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / (sd if sd and np.isfinite(sd) else 1.0)


def _add_interactions(d: pd.DataFrame, var: str, base: list) -> list:
    """Add var and var x each base term to `d`; return the new column names (var first, then e.g. TO, TS, TOS)."""
    names = [var]
    for b in base:
        d[f"{var}{b}"] = d[var] * d[b]
        names.append(f"{var}{b}")
    return names


def run_p1_T(Q: pd.DataFrame, *, opt="opt_pos", sale="salepost", cluster2="tq", rng=None, dry=False,
             complexity=False, t_sd=None) -> dict:
    """T2: the Stage 3 P1 regression (exit probability) plus T_z interacted with every term, key TOS. A NEGATIVE TOS
    supports H5 (the pattern O x S is positive-signed in P1, and H5 says it is stronger where T is low), so the one-sided
    p is taken for sign -1. `complexity=True` adds log assets and segment count (standardised) interacted the same way
    (the sensitivity fixed in advance). `dry` randomises the outcome within year-quarter, as in run_p1."""
    import gg_econ as E
    import gg_stage3 as S3
    d = S3.p1_frame(Q, opt=opt, sale=sale, rng=rng, dry=dry)
    d["T"] = d["T_z"].astype(float)
    base = ["O", "S", "OS"]
    xs = ["O", "S", "OS"] + _add_interactions(d, "T", base) + S3.P1_CTRL
    if complexity:
        d["LA"], d["NS"] = _z(d["log_assets"].astype(float)), _z(d["n_segments"].astype(float))
        xs += _add_interactions(d, "LA", base) + _add_interactions(d, "NS", base)
    res = E.fe_ols(d, "y", xs, firm="gvkey", year="tq", cluster="gvkey", cluster2=cluster2)
    s = S3.summarise(res, "TOS", -1)
    s.update({"n_exits": int(d.exit.sum()), "n_S": int(d.S.sum()), "n_scored": int(d["T"].notna().sum())})
    if t_sd:
        s["TOS_per_point"] = res["TOS"]["coef"] / t_sd
    return s


def run_p2_T(B: pd.DataFrame, *, opt="opt_pos", sale="sale", exit_="exit_in", cluster2="tq", rng=None, dry=False,
             complexity=False, t_sd=None) -> dict:
    """T3: the Stage 3 P2 regression (realised earnings miss) plus T_z interacted with every term, key TOSX. A POSITIVE
    TOSX supports H5 (O x S x X is negative-signed in P2), so the one-sided p is taken for sign +1."""
    import gg_econ as E
    import gg_stage3 as S3
    d = S3.p2_frame(B, opt=opt, sale=sale, exit_=exit_, rng=rng, dry=dry)
    d["T"] = d["T_z"].astype(float)
    base = ["O", "S", "X", "OS", "OX", "SX", "OSX"]
    xs = base + _add_interactions(d, "T", base) + S3.P2_CTRL
    if complexity:
        d["LA"], d["NS"] = _z(d["log_assets"].astype(float)), _z(d["n_segments"].astype(float))
        xs += _add_interactions(d, "LA", base) + _add_interactions(d, "NS", base)
    res = E.fe_ols(d, "y", xs, firm="gvkey", year="tq", cluster="gvkey", cluster2=cluster2)
    s = S3.summarise(res, "TOSX", +1)
    s.update({"n_S": int(d.S.sum()), "n_X": int(d.X.sum()), "n_SX": int((d.S * d.X).sum()), "n_scored": int(d["T"].notna().sum())})
    if t_sd:
        s["TOSX_per_point"] = res["TOSX"]["coef"] / t_sd
    return s


def permute_T_blocks(df: pd.DataFrame, *, block=("gvkey", "fiscal_year"), within: str = "year", rng,
                     cols=("T", "T_z")) -> pd.DataFrame:
    """Dry-run shuffle of T: whole blocks (default one firm-year, so every quarter of a filing keeps one T) swap T values
    with other blocks in the same `within` group (calendar or announcement year). Firm and time structure, and therefore
    the clustering, are untouched; only the link between T and the outcome is broken. For one-row-per-event data use
    block=("event_id",)."""
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    keys = list(block)
    first = out.drop_duplicates(keys)[list(dict.fromkeys(keys + [within] + present))].reset_index(drop=True)
    shuffled = first.copy()
    for _, idx in first.groupby(within).groups.items():
        idx = np.asarray(list(idx))
        perm = rng.permutation(len(idx))
        for c in present:
            shuffled.loc[idx, c] = first.loc[idx[perm], c].values
    out = out.drop(columns=present).merge(shuffled[keys + present], on=keys, how="left")
    return out[list(df.columns)].set_axis(df.index)


# ---------------------------------------------------------------- T4: leaver-minus-stayer alpha by T group
def event_alphas(windows: dict, ids, fac: pd.DataFrame, H: int, min_days: int = 120) -> pd.Series:
    """Annualised four-factor alpha of each event's own 252-day window (NaN when the window is too short)."""
    import gg_tests as GT
    out = {}
    for i in ids:
        port, cnt = GT.portfolio(windows, [i], H)
        out[i] = GT.alpha_fast(port, cnt, fac, min_days=min_days)
    return pd.Series(out, dtype=float)


def _cluster_fit(X: np.ndarray, y: np.ndarray, codes: np.ndarray):
    """OLS with cluster-robust standard errors (small-sample factor G/(G-1) x (N-1)/(N-K), as gg_econ)."""
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    G = int(codes.max()) + 1
    s = np.zeros((G, k))
    np.add.at(s, codes, X * u[:, None])
    v = xtx_inv @ (s.T @ s) @ xtx_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return b, np.sqrt(np.diag(v)), u


def _pooled_interaction(d: pd.DataFrame, x: str, *, cluster: str, reps: int, seed: int) -> dict:
    """y on leaver, x and leaver*x (x a 0/1 or standardised T); the interaction is the H5 contrast. p_one is the upper tail
    (a POSITIVE interaction supports H5) from a wild cluster bootstrap (Rademacher weights by cluster) that imposes the
    null on the interaction only, so equal non-zero leaver effects in the two groups are the null, not zero effects."""
    y = d["y"].to_numpy(float)
    lv = d["leaver"].to_numpy(float)
    xv = d[x].to_numpy(float)
    codes = pd.factorize(d[cluster])[0]
    one = np.ones(len(d))
    X = np.column_stack([one, lv, xv, lv * xv])
    b, se, _ = _cluster_fit(X, y, codes)
    t_obs = b[3] / se[3] if se[3] > 0 else np.nan
    Xr = X[:, :3]
    br, _, ur = _cluster_fit(Xr, y, codes)
    fit_r = Xr @ br
    G = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    t_star = np.empty(reps)
    for r in range(reps):
        w = rng.choice([-1.0, 1.0], G)[codes]
        bb, ss, _ = _cluster_fit(X, fit_r + w * ur, codes)
        t_star[r] = bb[3] / ss[3] if ss[3] > 0 else np.nan
    t_star = t_star[np.isfinite(t_star)]
    p_one = float((1 + np.sum(t_star >= t_obs)) / (1 + len(t_star))) if np.isfinite(t_obs) else np.nan
    return {"coef": float(b[3]), "se": float(se[3]), "t": float(t_obs), "p_one": p_one, "n": int(len(d)),
            "n_clusters": G, "reps": int(reps)}


def _secondary(ev: pd.DataFrame, windows, fac, H, cuts, n_boot: int, seed: int) -> dict:
    """Calendar-time contrast D_low - D_high (D_g = alpha of leavers minus alpha of stayers within group g) and its
    event-level bootstrap (resampling within group x leaver cells)."""
    import gg_tests as GT
    import gg_transparency as T
    e = ev[ev["event_id"].isin(windows.keys()) & ev["T"].notna()].copy()
    e["_g"] = T.assign_group(e["T"], cuts)
    cells = {(g, lv): e[(e["_g"] == g) & (e["leaver"] == lv)]["event_id"].to_numpy() for g in ("low", "high") for lv in (True, False)}

    def D(sel):
        a = lambda ids: GT.alpha_fast(*GT.portfolio(windows, ids, H), fac)
        return {g: a(sel[(g, True)]) - a(sel[(g, False)]) for g in ("low", "high")}

    d0 = D(cells)
    contrast = d0["low"] - d0["high"]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        samp = {k: (rng.choice(v, len(v)) if len(v) else v) for k, v in cells.items()}
        db_ = D(samp)
        boots[b] = db_["low"] - db_["high"]
    return {"contrast": float(contrast), "D": {g: float(v) for g, v in d0.items()}, "boots": boots}


def _recentred_contrast(ev, windows, fac, H, cuts, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of the calendar-time contrast recentred at its observed value (the null of equal effects)."""
    s = _secondary(ev, windows, fac, H, cuts, n_boot, seed)
    return s["boots"] - s["contrast"]


def run_t4(ev_all: pd.DataFrame, windows: dict, fac: pd.DataFrame, H: int, *, cuts, reps: int = 9999, seed: int = 0,
           n_boot: int = 1000, cluster: str = "ann_quarter") -> dict:
    """T4: is the leaver-minus-stayer alpha more negative where T is low?

    `ev_all`: event_id, leaver (bool), T, T_z, ann_quarter. Primary: pooled event-level interaction regression on the low
    and high T groups (see _pooled_interaction); positive interaction supports H5. Corroboration: the same regression with
    continuous leaver x T_z on all scored events (cluster-robust t, one-sided). Secondary: calendar-time D_low - D_high with
    a recentred bootstrap (lower tail supports H5). Returns {"nan_reason": ...} when the groups are not viable.
    """
    from scipy import stats as st
    import gg_transparency as T
    ev = ev_all[ev_all["event_id"].isin(windows.keys()) & ev_all["T"].notna()].copy()
    ev["leaver"] = ev["leaver"].astype(bool)
    via = T.group_viability(ev["T"], ev["leaver"], ~ev["leaver"], cuts)
    if not via["ok"]:
        return {"nan_reason": "; ".join(via["reasons"]), "groups": via["groups"]}
    ev["y"] = ev["event_id"].map(event_alphas(windows, ev["event_id"], fac, H))
    ev["_g"] = T.assign_group(ev["T"], cuts)
    ev = ev.dropna(subset=["y"])
    d = ev[ev["_g"].isin(["low", "high"])].copy()
    d["high"] = (d["_g"] == "high").astype(float)
    ccol = cluster if cluster in d.columns else "year"
    out = {"cuts": list(cuts), "primary": _pooled_interaction(d, "high", cluster=ccol, reps=reps, seed=seed)}
    out["groups"] = {g: {"leavers": int(((d["_g"] == g) & d["leaver"]).sum()), "stayers": int(((d["_g"] == g) & ~d["leaver"]).sum()),
                         "effect": float(d[(d["_g"] == g) & d["leaver"]]["y"].mean() - d[(d["_g"] == g) & ~d["leaver"]]["y"].mean())}
                     for g in ("low", "high")}
    c = ev.dropna(subset=["T_z"])
    ccodes = pd.factorize(c[ccol])[0]
    X = np.column_stack([np.ones(len(c)), c["leaver"].astype(float), c["T_z"], c["leaver"].astype(float) * c["T_z"]])
    b, se, _ = _cluster_fit(X, c["y"].to_numpy(float), ccodes)
    G = int(ccodes.max()) + 1
    t_ = b[3] / se[3] if se[3] > 0 else np.nan
    out["continuous"] = {"coef": float(b[3]), "se": float(se[3]), "t": float(t_), "p_one": float(st.t.sf(t_, G - 1)),
                         "n": int(len(c)), "n_clusters": G}
    if n_boot > 0:
        s = _secondary(ev, windows, fac, H, cuts, n_boot, seed + 5)
        centred = s["boots"] - s["contrast"]
        out["secondary"] = {"contrast": s["contrast"], "D": s["D"], "p_lower": float((1 + np.sum(centred <= s["contrast"])) / (1 + len(centred))),
                            "ci": [float(x) for x in np.nanpercentile(s["boots"], [2.5, 97.5])]}
    return out


# ---------------------------------------------------------------- the gated runner
PRIOR_STAGE_TESTS = 28            # approximate project-wide count before Stage 5 (spec 5)


class RunRefused(Exception):
    """A precondition of the Stage 5 run is not met (freeze, gate, config provenance)."""


def load_current_hashes(path) -> dict | None:
    """unit_id -> sha256 of the bundle text the scorer saw most recently (latest line wins); None if the file does not exist."""
    import json
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return None
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["unit_id"]] = r["sha"]
    return out


def default_paths() -> dict:
    import guide_and_go as G
    p = {"freeze": G.RESULTS / "stage5_freeze.json", "gate": G.RESULTS / "stage5_gate.json",
         "results": G.RESULTS / "stage5_results.json", "dryrun": G.RESULTS / "stage5_results_DRYRUN.json",
         "scores": G.DATA / "stage5_scores.pkl"}
    cur = load_current_hashes(G.ROOT / "validation" / "transp_scores" / "current_hashes.jsonl")
    if cur is not None:                                  # the production run checks every score against the text it was made on
        p["current_input_hashes"] = cur
    return p


def verify_score_inputs(scores: pd.DataFrame, current_hashes: dict) -> list:
    """Unit ids whose scored input text no longer matches the current text (sha256 differs). Units not in
    `current_hashes` are not checked. Catches a re-extraction (new name list, fixed bug) that was never re-scored."""
    s = scores.drop_duplicates(["unit_id", "input_sha256"])
    return sorted({u for u, h in zip(s["unit_id"], s["input_sha256"]) if u in current_hashes and current_hashes[u] != h})


def check_preconditions(paths: dict) -> list:
    """Verify the freeze, the gate and the provenance of the scores; return the tests the gate enables. Raises RunRefused."""
    import json
    import gg_transparency as T
    fp = paths["freeze"]
    if not fp.exists():
        raise RunRefused(f"measurement not frozen: no freeze file at {fp}")
    freeze = json.loads(fp.read_text(encoding="utf-8"))
    try:
        T.assert_frozen(fp, T.measurement_config(freeze["model"]))     # uses the CLI version installed NOW
    except T.NotFrozen as e:
        raise RunRefused(f"measurement config in code differs from the frozen one: {e}")
    gp = paths["gate"]
    if not gp.exists():
        raise RunRefused(f"no gate file at {gp}: the reliability gate has not been run")
    gate = json.loads(gp.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RunRefused(f"the reliability gate did not pass: {gate.get('failures')}")
    if gate.get("config_hash") != freeze["config_hash"]:
        raise RunRefused("the gate was computed under a different measurement config than the frozen one")
    hashes = paths.get("scores_config_hashes")
    if hashes is None:
        hashes = list(pd.read_pickle(paths["scores"])["config_hash"].unique())
    if set(hashes) != {freeze["config_hash"]}:
        raise RunRefused("the scores were produced under a different measurement config than the frozen one")
    cur = paths.get("current_input_hashes")
    if cur is not None:
        sf = paths.get("scores_frame")
        sf = pd.read_pickle(paths["scores"]) if sf is None else sf
        bad = verify_score_inputs(sf, cur)
        if bad:
            raise RunRefused(f"{len(bad)} scored unit(s) have input text that changed since scoring (first: {bad[:3]}); re-score them")
    return list(gate.get("tests_enabled", []))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def load_inputs(dry: bool) -> dict:
    """Real inputs for the run: the Stage 2 events (T1), the Stage 3 panels Q and B built by gg_stage3.build_panels (T2, T3) and
    the stayer and leaver events of Stage 3 P3 (T4), each with T attached from stage5_scores.pkl (accession-level scores)."""
    import pickle
    import guide_and_go as G
    import gg_stage3 as S3
    paths = default_paths()
    scores = pd.read_pickle(paths["scores"])
    meta = score_meta(scores)
    idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
    sw = pickle.load(open(G.DATA / "study_windows.pkl", "rb"))

    # T1: the Stage 2 event set (5,163 events with their return windows), dated exactly as Stage 2 dated its windows
    import gg_study as GS
    ev_final = GS.load_events(G.ROOT / "validation" / "candidate_labels.csv", False)[0]
    events = attach_event_T(sw["events"], ev_final, scores, idx, meta)

    # T2 and T3: the Stage 3 panels, rebuilt by the same function Stage 3 uses
    ep, sales = pd.read_pickle(G.DATA / "panel_episodes.pkl"), pd.read_pickle(G.DATA / "panel_sales.pkl")
    g, Q0 = pd.read_pickle(G.DATA / "panel_guid.pkl"), pd.read_pickle(G.DATA / "panel_A_quarters.pkl")
    db = G.WRDS(refresh=False)
    crsp, mkt, _ = G.load_crsp(db, sorted(g.permno.unique()), S3.START, S3.END)
    tab, age_tab = S3.market_table(crsp), S3.load_age(db)
    si, si1m = S3.PD.SaleIndex(sales), S3.PD.SaleIndex(sales, 1e6)
    Q, B, _ = S3.build_panels(Q0, g, ep, sales, tab, age_tab, si, si1m)
    Q = attach_T(Q, scores, idx, meta, date_col="qs", strict=False)
    B = attach_T(B, scores, idx, meta, date_col="t", strict=True)       # a guidance release is dated within the day: a 10-K filed that day is not yet known

    # T4: stayers and leavers, with their own windows
    ev_all, win, fac, H = S3.build_p3_events(db, Q, crsp, mkt)
    ev_all = attach_T(ev_all, scores, idx, meta, date_col="d", strict=True)
    return {"events": events, "ev_all": ev_all, "Q": Q, "B": B, "windows": sw["windows252"], "fac": sw["factors"], "H": len(sw["calendar"]),
            "flagged_ids": set(sw["flag_ids"]), "broad_ids": None, "t_sd": meta["sd"], "meta": meta,
            "windows_p3": win, "fac_p3": fac, "H_p3": H}


def _secondary_analyses(inp: dict, events: pd.DataFrame, Q: pd.DataFrame, B: pd.DataFrame, enabled: list, rng) -> dict:
    """The analyses fixed in advance as secondary (spec 4): T1 continuous, and T2/T3 with the complexity controls. They are
    reported beside the primary tests, outside the Holm family. A failure is recorded, not raised."""
    out = {}

    def attempt(name, fn):
        try:
            out[name] = fn()
        except Exception as e:                       # a secondary analysis must never sink the primary results
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    if "T1" in enabled:
        def t1c():
            ev = events.assign(flag=events["event_id"].isin(inp["flagged_ids"]))
            ev = ev[ev["event_id"].isin(inp["windows"].keys()) & ev["T_z"].notna()]
            return run_t1_continuous(ev, event_alphas(inp["windows"], ev["event_id"], inp["fac"], inp["H"]))
        attempt("T1_continuous", t1c)
    for name, key, df, fn in (("T2_complexity", "T2", Q, run_p1_T), ("T3_complexity", "T3", B, run_p2_T)):
        if key in enabled:
            if {"log_assets", "n_segments"} <= set(df.columns):
                attempt(name, lambda df=df, fn=fn: fn(df, complexity=True, rng=rng, t_sd=inp.get("t_sd")))
            else:
                out[name] = {"skipped": "log_assets and n_segments not in the panel"}
    return out


def _shuffle_panel(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Dry-run shuffle of T on a firm-quarter panel: blocks are filings (gvkey, accession) when known, else firm-years,
    shuffled within calendar year, so repeated quarters of one filing keep one T."""
    d = df.copy()
    d["_yr"] = d["tq"].astype(str).str[:4] if "tq" in d.columns else d["year"]
    blk = ("gvkey", "accession") if "accession" in d.columns else ("gvkey", "_yr")
    out = permute_T_blocks(d, block=blk, within="_yr", rng=rng)
    return out.drop(columns="_yr")


def main(argv=None, *, loader=None, paths=None) -> int:
    """python src/gg_stage5.py [--dry-run] [--reps N]. Refuses unless the measurement is frozen, the gate passed for that
    config, and the scores were made under it. Runs only the tests the gate enables; Holm over those, each on its primary
    p-value. --dry-run shuffles T (events across events within announcement year; panels across firm-year blocks within
    year) and writes only the DRYRUN file."""
    import argparse
    import json
    import gg_tests as GT
    import gg_transparency as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    paths = paths or default_paths()
    try:
        enabled = check_preconditions(paths)
    except RunRefused as e:
        print(f"REFUSED: {e}")
        return 2
    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    inp = (loader or load_inputs)(a.dry_run)
    rng = np.random.default_rng(a.seed)
    events, ev_all, Q, B = inp["events"], inp["ev_all"], inp["Q"], inp["B"]
    if a.dry_run:
        events = permute_T_blocks(events, block=("event_id",), within="year", rng=rng)
        ev_all = permute_T_blocks(ev_all, block=("event_id",), within="year", rng=rng)
        Q, B = _shuffle_panel(Q, rng), _shuffle_panel(B, rng)
    cuts = T.cut_points(events["T"])
    n_boot = 1000 if a.reps >= 500 else 200
    tests, primary, skipped = {}, {}, {}
    for name in ("T1", "T2", "T3", "T4"):
        if name not in enabled:
            continue
        if name == "T1":
            r = run_t1(inp["windows"], events, inp["fac"], inp["H"], inp["flagged_ids"], cuts=cuts, reps=a.reps, seed=a.seed,
                       broad_ids=inp.get("broad_ids"), n_boot=n_boot)
            p = r.get("perm_p")
        elif name == "T2":
            r = run_p1_T(Q, rng=rng, dry=False, t_sd=inp.get("t_sd"))
            p = r.get("p_one")
        elif name == "T3":
            r = run_p2_T(B, rng=rng, dry=False, t_sd=inp.get("t_sd"))
            p = r.get("p_one")
        else:
            r = run_t4(ev_all, inp.get("windows_p3", inp["windows"]), inp.get("fac_p3", inp["fac"]), inp.get("H_p3", inp["H"]), cuts=cuts, reps=a.reps, seed=a.seed, n_boot=n_boot)
            p = r.get("primary", {}).get("p_one")
        if isinstance(r, dict) and "nan_reason" in r:
            skipped[name] = r["nan_reason"]          # decided at run time from the full scores (not at the pilot)
            continue
        tests[name] = {"p_primary": p, "result": r}
        if p is not None and np.isfinite(p):
            primary[name] = float(p)
    secondary = _secondary_analyses(inp, events, Q, B, enabled, rng)
    holm = GT.holm(primary) if primary else {}
    res = {"dry_run": bool(a.dry_run), "config_hash": freeze["config_hash"], "cut_rule": T.CUT_RULE, "reps": a.reps,
           "tests_enabled": enabled, "tests": tests, "skipped": skipped, "secondary": secondary, "family_p": primary,
           "holm": holm, "family_size": len(holm), "prior_stage_tests": PRIOR_STAGE_TESTS,
           "project_wide_tests": PRIOR_STAGE_TESTS + len(holm)}
    out = paths["dryrun"] if a.dry_run else paths["results"]
    out.write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
