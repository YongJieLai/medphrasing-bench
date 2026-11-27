"""M3-0a 评测集构建：从 holdout 分层抽 100 条。

分层（派单 §7.10）：
  strong leading 30 / multi_barreled 30 / surface follow_up 30 / 其余随机 10

稀有档强制纳入（全库 strong 203 / multi 264 / surface 315，随机抽大概率抽不到）。
抽样固定 seed，可复现。

产出 outputs/m3_0a/eval_set.jsonl：每行含
  dialogue_id, turn_idx, doctor_utt, prev_patient_utt
  + M0 四维标签（作为真值来源 1，供后续对比，不下发进判别器）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_simulator import valid_rows
from scripts.m2d_run import split_holdout

DIM_ALIAS = {
    # 派单表格 ↔ M0 口径 的别名，仅用于文档说明，不影响抽样
    "leadingness": {"leading": "mild"},
    "follow_up": {"follow_up": "deepen"},
}


def build_eval_set(rows: list, truthset: dict, holdout_seed: int = 42,
                   sample_seed: int = 12345) -> dict:
    valid = valid_rows(rows)
    by_d, hold_ids = split_holdout(valid, truthset, holdout_seed)
    hold_rows = [r for d in hold_ids for r in by_d[d]]

    # 分组索引
    strong = [r for r in hold_rows if r["leadingness"] == "strong"]
    multi = [r for r in hold_rows if r["multi_barreled"] == "multi"]
    surface = [r for r in hold_rows if r["follow_up"] == "surface"]
    rest = [r for r in hold_rows
            if r["leadingness"] != "strong"
            and r["multi_barreled"] != "multi"
            and r["follow_up"] != "surface"]

    rng = random.Random(sample_seed)

    def sample(pool, n):
        return rng.sample(pool, min(n, len(pool)))

    picked_strong = sample(strong, 30)
    picked_multi = sample(multi, 30)
    picked_surface = sample(surface, 30)
    picked_rest = sample(rest, 10)

    # 去重合并（strong/multi/surface 可能重叠）
    seen = set()
    merged = []
    for r in picked_strong + picked_multi + picked_surface + picked_rest:
        key = (r["dialogue_id"], r["turn_idx"])
        if key not in seen:
            seen.add(key)
            merged.append(r)

    # 若去重后不足 100，从剩余未选中补足
    if len(merged) < 100:
        chosen = {(r["dialogue_id"], r["turn_idx"]) for r in merged}
        remaining = [r for r in hold_rows
                     if (r["dialogue_id"], r["turn_idx"]) not in chosen]
        rng.shuffle(remaining)
        for r in remaining:
            if len(merged) >= 100:
                break
            merged.append(r)

    # 覆盖统计
    def has(r, dim, val):
        return r[dim] == val

    coverage = {
        "strong": sum(1 for r in merged if has(r, "leadingness", "strong")),
        "multi": sum(1 for r in merged if has(r, "multi_barreled", "multi")),
        "surface": sum(1 for r in merged if has(r, "follow_up", "surface")),
    }

    return {
        "holdout_dialogues": len(hold_ids),
        "holdout_turns": len(hold_rows),
        "pool_counts": {"strong": len(strong), "multi": len(multi),
                        "surface": len(surface), "rest": len(rest)},
        "eval_n": len(merged),
        "coverage": coverage,
        "records": [{
            "dialogue_id": r["dialogue_id"],
            "turn_idx": r["turn_idx"],
            "doctor_utt": r["doctor_utt"],
            "prev_patient_utt": r.get("prev_patient_utt"),
            "m0": {
                "openness": r["openness"],
                "leadingness": r["leadingness"],
                "multi_barreled": r["multi_barreled"],
                "follow_up": r["follow_up"],
            },
        } for r in merged],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annot", help="M0 标注文件 annot_all_flash.jsonl")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-seed", type=int, default=12345)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.annot, encoding="utf-8")]
    truthset = {t["dialogue_id"]: t
                for t in json.load(open(args.truthset, encoding="utf-8"))}
    res = build_eval_set(rows, truthset, sample_seed=args.sample_seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = {k: v for k, v in res.items() if k != "records"}
    (out.parent / "eval_set_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out, "w", encoding="utf-8") as f:
        for r in res["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
