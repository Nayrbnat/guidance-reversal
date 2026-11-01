"""
Build the hand-labelling tasks used to validate the departure-text classifier and the announcement matcher.

For each sampled departure we give an annotator the executive's name/role/date and every candidate
Capital IQ 'Executive' item for the firm near the departure date. They label which item (if any) announces
THIS executive leaving, how they left, and whether the destination is a government role.
Annotators never see the pipeline's own labels.

usage: python validation/make_gold_sample.py <events.csv> <ciq_keydev cache .pkl> [n_random=200]
"""
import sys, json, random
import pandas as pd
sys.path.insert(0, "src")
import gg_labels as L

events_csv, keydev_pkl = sys.argv[1], sys.argv[2]
n_random = int(sys.argv[3]) if len(sys.argv) > 3 else 200
ev = pd.read_csv(events_csv, parse_dates=["leftofc", "dep_ann"], dtype={"gvkey": str})
ev["gvkey"] = ev.gvkey.astype(str).str.zfill(6)
kd = pd.read_pickle(keydev_pkl)
kd["gvkey"] = kd.gvkey.astype(str).str.zfill(6)
kd["announcedate"] = pd.to_datetime(kd.announcedate)
kd = kd[kd.eventtype.str.contains("Executive", case=False, na=False)].copy()
kd["txt"] = (kd.headline.fillna("") + ". " + kd.situation.fillna("")).str.strip()
by_g = dict(tuple(kd.groupby("gvkey")))

rng = random.Random(7)
ev = ev[ev.leftofc.dt.year.between(2010, 2025)].reset_index(drop=True)
rand = ev.sample(n_random, random_state=7)
flag = ev[ev.guide_and_go == 1].drop(index=rand.index, errors="ignore")
flag = flag.sample(min(40, len(flag)), random_state=7)
gov = ev[(ev.label == "government")].drop(index=list(rand.index) + list(flag.index), errors="ignore")
gov = gov.sample(min(40, len(gov)), random_state=7)
sample = pd.concat([rand.assign(stratum="random"), flag.assign(stratum="old_flag"), gov.assign(stratum="old_gov")])
sample = sample.sample(frac=1, random_state=7).reset_index(drop=True)
sample["id"] = [f"g{i + 1:03d}" for i in range(len(sample))]
sample["split"] = ["dev" if (i % 2 == 0) else "test" for i in range(len(sample))]

def items_for(r):
    g = by_g.get(r.gvkey)
    if g is None:
        return []
    p = L.parse_name(r.exec_name)
    w = g[(g.announcedate >= r.leftofc - pd.Timedelta(days=365)) & (g.announcedate <= r.leftofc + pd.Timedelta(days=30))].copy()
    if w.empty:
        return []
    w["named"] = w.txt.map(lambda t: any(L.mentions_name(s, p["last_tokens"]) for s in L.split_sentences(t)))
    w["dist"] = (w.announcedate - r.leftofc).abs().dt.days
    named = w[w.named].sort_values("dist").head(6)
    near = w[~w.named].sort_values("dist").head(2)
    use = pd.concat([named, near]).sort_values("announcedate")
    out = []
    for i, (_, k) in enumerate(use.iterrows()):
        out.append({"i": i, "date": k.announcedate.strftime("%Y-%m-%d"), "type": k.eventtype,
                    "mentions_exec_surname": bool(k.named),
                    "text": L.name_excerpt(k.txt, p["last_tokens"], 1500)})
    return out

rows = []
for _, r in sample.iterrows():
    rows.append({"id": r.id, "company": r.conm, "exec_name": r.exec_name, "role": r.role,
                 "left_office_date": r.leftofc.strftime("%Y-%m-%d"), "items": items_for(r)})

key = sample[["id", "event_id", "stratum", "split", "exec_name", "conm", "leftofc", "label", "abrupt", "gov_dest", "guide_and_go"]].rename(
    columns={"label": "old_label", "abrupt": "old_abrupt", "gov_dest": "old_gov", "guide_and_go": "old_flag"})
# annotators 1-7: 40 unique tasks each; annotator 8: 40 duplicated tasks for inter-annotator agreement
ids = list(sample.id)
assign = {}
for k in range(7):
    for t in ids[k * 40:(k + 1) * 40]:
        assign[t] = k + 1
overlap = random.Random(11).sample(ids, 40)
for k in range(1, 9):
    mine = [r for r in rows if (assign.get(r["id"]) == k) or (k == 8 and r["id"] in overlap)]
    with open(f"validation/tasks/annotator_{k}.jsonl", "w", encoding="utf-8") as f:
        for r in mine:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"annotator {k}: {len(mine)} tasks")
key["annotator"] = key.id.map(assign)
key["overlap"] = key.id.isin(overlap)
key.to_csv("validation/gold_key.csv", index=False)
print("key rows:", len(key), "| strata:", key.stratum.value_counts().to_dict(), "| tasks with no candidate items:", sum(1 for r in rows if not r["items"]))
