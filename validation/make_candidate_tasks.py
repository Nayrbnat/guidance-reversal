"""
Build blind annotation tasks for the CANDIDATE events (recent guidance + insider sale; both text-independent).
Annotators judge from the announcement text only and never see returns.

usage:
  python validation/make_candidate_tasks.py pass1 [events.csv]            -> validation/cand_tasks/pass1_k.jsonl
  python validation/make_candidate_tasks.py pass2 [events.csv]            -> second, independent read of the events that
        pass 1 called abrupt or the rule classifier called abrupt (these drive the flag, so they get two reads)
"""
import sys, json, glob
import pandas as pd
sys.path.insert(0, "src")
import gg_labels as L

mode = sys.argv[1]
events_csv = sys.argv[2] if len(sys.argv) > 2 else "outputs/data/events.csv"
ev = pd.read_csv(events_csv, parse_dates=["leftofc", "dep_ann"], dtype={"gvkey": str})
ev["gvkey"] = ev.gvkey.astype(str).str.zfill(6)
cand = ev[ev.gr_candidate == 1].copy()
print("candidate events:", len(cand))

kd_files = sorted(glob.glob("gg_cache/ciq_keydev_*.pkl"), key=lambda f: -len(pd.read_pickle(f)))
kd = pd.read_pickle(kd_files[0])
kd["gvkey"] = kd.gvkey.astype(str).str.zfill(6)
kd["announcedate"] = pd.to_datetime(kd.announcedate)
kd = kd[kd.eventtype.str.contains("Executive", case=False, na=False)].copy()
kd["txt"] = (kd.headline.fillna("") + ". " + kd.situation.fillna("")).str.strip()
by_g = dict(tuple(kd.groupby("gvkey")))


def items_for(r):
    g = by_g.get(r.gvkey)
    if g is None:
        return []
    p = L.parse_name(r.exec_name)
    w = g[(g.announcedate >= r.leftofc - pd.Timedelta(days=365)) & (g.announcedate <= r.leftofc + pd.Timedelta(days=30))].copy()
    if w.empty:
        return []
    w["named"] = w.txt.map(lambda t: L.mentions_name(t, p["last_tokens"]))
    w["dist"] = (w.announcedate - r.leftofc).abs().dt.days
    use = pd.concat([w[w.named].sort_values("dist").head(6), w[~w.named].sort_values("dist").head(2)]).sort_values("announcedate")
    return [{"i": i, "date": k.announcedate.strftime("%Y-%m-%d"), "type": k.eventtype, "mentions_exec_surname": bool(k.named),
             "text": L.name_excerpt(k.txt, p["last_tokens"], 1500)} for i, (_, k) in enumerate(use.iterrows())]


if mode == "pass1":
    sel = cand
    prefix = "pass1"
else:
    labs = []
    for f in glob.glob("validation/cand_labels_raw/pass1_*.jsonl"):
        labs += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    p1 = pd.DataFrame(labs).rename(columns={"id": "event_id"})
    a1 = set(p1[p1.abrupt.astype(str) == "True"].event_id)
    sel = cand[cand.event_id.isin(a1) | (cand.abrupt == 1)]
    prefix = "pass2"
    print(f"pass 1 called abrupt: {len(a1)} | rule abrupt among candidates: {int((cand.abrupt==1).sum())} | to re-read: {len(sel)}")

rows = [{"id": r.event_id, "company": r.conm, "exec_name": r.exec_name, "role": r.role,
         "left_office_date": r.leftofc.strftime("%Y-%m-%d"), "items": items_for(r)} for _, r in sel.iterrows()]
import os
os.makedirs("validation/cand_tasks", exist_ok=True)
os.makedirs("validation/cand_labels_raw", exist_ok=True)
for f in glob.glob(f"validation/cand_tasks/{prefix}_*.jsonl"):
    os.remove(f)                      # regenerate this pass's task files (they are derived files)
n_files = 0
for k in range(0, len(rows), 40):
    n_files += 1
    with open(f"validation/cand_tasks/{prefix}_{n_files}.jsonl", "w", encoding="utf-8") as f:
        for r in rows[k:k + 40]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{prefix}: {len(rows)} tasks in {n_files} files; tasks without any candidate item: {sum(1 for r in rows if not r['items'])}")
