"""M2e-0 三态重跑：改进 slot 判断后，重跑三态并重算交叉表/条件概率。

读 outputs/*/m2e0_quantify.json 的 records，对 is_yesno==1 的轮重跑 slot_status，
重算 3×5 交叉表与两个条件概率，回写 JSON。

用法:
  python scripts/m2e0_reslot.py outputs/20261127_m2e0/m2e0_quantify.json \
      --truthset data/aiu_truthset_v2_20261126.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.m2e0_quantify import slot_one, bootstrap_proportions
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--truthset", required=True)
    args = ap.parse_args()

    data = json.load(open(args.json_path, encoding="utf-8"))
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}
    recs = data["records"]

    yesno = [r for r in recs if r["is_yesno"] == 1]
    print(f"重跑三态：yes/no 轮 {len(yesno)} 条")

    t0 = time.time()
    for i, r in enumerate(yesno):
        ts = truthset[r["dialogue_id"]]["truth_set"]
        ss = slot_one(r["doctor_utt"], ts)
        r["slot_status"] = ss["slot_status"]
        if ss.get("error"):
            r["slot_error"] = ss["error"]
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(yesno)}  耗时 {time.time()-t0:.0f}s")

    yesno_valid = [r for r in yesno if r["answer_class"] in ("a", "b", "c", "d", "e")]

    cross = defaultdict(lambda: defaultdict(int))
    for r in yesno_valid:
        cross[r.get("slot_status", "error")][r["answer_class"]] += 1

    def cond(slot, ans):
        denom = sum(cross[slot].values())
        return (cross[slot][ans] / denom if denom else float("nan")), denom

    p_b_denied, n_denied = cond("denied", "b")
    p_c_absent, n_absent = cond("absent", "c")

    data["cross_table"] = {k: dict(v) for k, v in cross.items()}
    data["conditional_probs"] = {
        "p_b_given_slot_denied": {"value": p_b_denied, "n": n_denied},
        "p_c_given_slot_absent": {"value": p_c_absent, "n": n_absent},
    }
    data["slot_reslot_note"] = "slot 判断已用相关性筛 top-8 + 明确 denied/absent 定义重跑"

    json.dump(data, open(args.json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n重跑后三态交叉表:")
    print(f"  {'slot':<10} {'a':>6} {'b':>6} {'c':>6} {'d':>6} {'e':>6} {'sum':>6}")
    for ss in ("affirmed", "denied", "absent", "error"):
        row = cross[ss]
        print(f"  {ss:<10} {row['a']:>6} {row['b']:>6} {row['c']:>6} "
              f"{row['d']:>6} {row['e']:>6} {sum(row.values()):>6}")
    print(f"\nP(答否定|清单有否定槽) = {p_b_denied:.1%} (n={n_denied})")
    print(f"P(答不知道|清单无该槽) = {p_c_absent:.1%} (n={n_absent})")
    print(f"\n完成 -> {args.json_path}")


if __name__ == "__main__":
    main()
