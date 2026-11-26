"""从全量标注数据估计条件概率表并保存（M2 校准，D8/D9）。

用法:
  python scripts/calibrate_simulator.py data/annot_all_flash.jsonl --out data/simulator_probs_20261126.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import (
    ConditionalSimulator, OPENNESS, LEADINGNESS, MULTI_BARRELED,
    RELEVANCE, DEFINITENESS, COMPLETENESS,
)


def valid_rows(rows: list[dict]) -> list[dict]:
    """过滤：is_question==1 且三个措辞维度、三个披露属性均合法。"""
    out = []
    for r in rows:
        if r.get("label_error"):
            continue
        if r.get("is_question") != 1:
            continue
        if (r.get("openness") in OPENNESS
                and r.get("leadingness") in LEADINGNESS
                and r.get("multi_barreled") in MULTI_BARRELED
                and r.get("relevance") in RELEVANCE
                and r.get("definiteness") in DEFINITENESS
                and r.get("completeness") in COMPLETENESS):
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    print(f"原始 {len(rows)} 条，过滤后有效 {len(valid)} 条")

    sim = ConditionalSimulator.estimate(valid, alpha=args.alpha)
    sim.save(Path(args.out))

    # 打印条件概率表供审阅
    print(f"\n已保存 -> {args.out}\n")
    print("P(relevance | openness):")
    for o in OPENNESS:
        row = ", ".join(f"{r}:{sim.probs['relevance|openness'][(o, r)]*100:.1f}%" for r in RELEVANCE)
        print(f"  {o:7s}  {row}")
    print("P(definiteness | leadingness):")
    for l in LEADINGNESS:
        row = ", ".join(f"{d}:{sim.probs['definiteness|leadingness'][(l, d)]*100:.1f}%" for d in DEFINITENESS)
        print(f"  {l:7s}  {row}")
    print("P(completeness | multi_barreled):")
    for m in MULTI_BARRELED:
        row = ", ".join(f"complete={c}:{sim.probs['completeness|multi_barreled'][(m, c)]*100:.1f}%" for c in COMPLETENESS)
        print(f"  {m:7s}  {row}")


if __name__ == "__main__":
    main()
