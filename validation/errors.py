"""Show classifier errors on a gold split. usage: python validation/errors.py dev fp|fn|miss [max]"""
import sys, json
import pandas as pd
sys.path.insert(0, "src")
import gg_labels as L

split, kind = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10
g = pd.read_csv("validation/gold_labels.csv")
g = g[g.split == split] if split != "all" else g
tasks = {}
for k in range(1, 9):
    for ln in open(f"validation/tasks/annotator_{k}.jsonl", encoding="utf-8"):
        d = json.loads(ln)
        tasks.setdefault(d["id"], d)
shown = 0
for _, r in g.iterrows():
    t = tasks[r.id]
    c = L.classify_event(t["items"], t["exec_name"])
    ga, pa = int(str(r.abrupt) == "True"), int(c["abrupt"] is True)
    gi = None if pd.isna(r.ann_item) else int(r.ann_item)
    if kind == "fp" and not (pa and not ga):
        continue
    if kind == "fn" and not (ga and not pa):
        continue
    if kind == "miss" and not (gi is not None and not c["departing"] and r.dep_type not in ("other_unclear",) and ga == 0 and False):
        if not (gi is not None and not c["departing"]):
            continue
    shown += 1
    if shown > mx:
        break
    use = pd_i = c.get("item") if c.get("item") is not None else gi
    it = [k for k in t["items"] if k["i"] == use]
    print(f"\n[{r.id}] {t['exec_name']} | gold={r.dep_type}/abrupt={r.abrupt} item={gi} | pred={c['dep_type']} item={c.get('item')} eff={c['eff_days']}")
    print("   gold evidence:", str(r.evidence)[:160])
    if it:
        print("   TEXT:", it[0]["text"][:420])
