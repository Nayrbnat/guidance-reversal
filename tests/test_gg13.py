"""
Tests for Stage 13 (a wider sale condition for the Guidance Reversal flag): gg_stage13.
Run:  uv run python tests/test_gg13.py    (from the repo root; no pytest needed)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


def _sales():
    return pd.DataFrame({"trandate": pd.to_datetime(["2024-09-20", "2024-07-24", "2024-01-10", "2025-03-01", "2025-05-20"]), "val": [25.7e6, 1.0e6, 5.0e6, 2.0e6, 9.0e6]})


def test_window_sum():
    import gg_stage13 as S13
    lg, ann = pd.Timestamp("2025-04-24"), pd.Timestamp("2025-05-07")
    check("a sale 7 months before the latest guidance counts with a 9-month look-back", S13.sale_value_in_window(_sales(), lg, ann, 9) >= 25.7e6)
    got = S13.sale_value_in_window(_sales(), lg, ann, 9)
    check("window is [guidance - 9 months, announcement]: sales on 2024-09-20, 2024-07-24 and 2025-03-01 are in; Jan 2024 and after the announcement are out",
          abs(got - (25.7e6 + 1.0e6 + 2.0e6)) < 1.0, str(got))
    check("with the old rule (look-back 0) only sales from the guidance date on count", S13.sale_value_in_window(_sales(), lg, ann, 0) == 0.0)
    check("a shorter look-back of 6 months drops the September 2024 sale but keeps the 2025-03-01 one", abs(S13.sale_value_in_window(_sales(), lg, ann, 6) - 2.0e6) < 1.0)
    check("the window start is inclusive (a sale exactly 9 months before the guidance counts)",
          S13.sale_value_in_window(pd.DataFrame({"trandate": [pd.Timestamp("2024-07-24")], "val": [3.0]}), lg.replace(month=4, day=24), ann, 9) == 3.0)
    check("no guidance date gives zero, an empty sale table gives zero",
          S13.sale_value_in_window(_sales(), pd.NaT, ann, 9) == 0.0 and S13.sale_value_in_window(_sales().iloc[:0], lg, ann, 9) == 0.0)


def test_flag_columns():
    import gg_stage13 as S13
    ev = pd.DataFrame({"event_id": ["a", "b", "c", "d"], "guided_recent": [1, 1, 0, 1], "abrupt": [1, 1, 1, 0], "sell_val_win9": [5.0, 0.0, 5.0, 5.0], "sell_val_post_lastguid": [0.0, 0.0, 5.0, 5.0]})
    out = S13.add_flags(ev)
    check("new candidate = guided recently and a sale in the wider window", out.set_index("event_id").gr_candidate9.to_dict() == {"a": 1, "b": 0, "c": 0, "d": 1})
    check("new rule flag = new candidate and abrupt", out.set_index("event_id").gr_rule9.to_dict() == {"a": 1, "b": 0, "c": 0, "d": 0})
    check("the old flag columns are never overwritten by this stage", set(["gr_candidate", "gr_rule"]).isdisjoint(out.columns))
    both = out.assign(old=((ev.sell_val_post_lastguid > 0) & (ev.guided_recent == 1)).astype(int))
    check("every old candidate that is guided stays a new candidate (the new window contains the old one)", bool((both.gr_candidate9 >= both.old).all()))


def main():
    test_window_sum()
    test_flag_columns()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
