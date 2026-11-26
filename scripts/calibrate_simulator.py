"""从全量标注数据序贯构建样本并估计 3 维条件概率表（M2b 校准，D8/D9）。

用法:
  python scripts/calibrate_simulator.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --config config/m2b_config.json \
      --out data/simulator_probs_v2_20261126.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import (
    ConditionalSimulator, OPENNESS, LEADINGNESS, MULTI_BARRELED,
    RELEVANCE, DEFINITENESS, COMPLETENESS, INFO_UNITS, _bin_of,
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


def build_sequential_samples(valid: list[dict], truthset: dict[int, dict]) -> list[dict]:
    """对每段对话按提问序贯重编号轮次，序贯推算剩余量，产出 (措辞, remaining, turn, y) 样本。"""
    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_d[r["dialogue_id"]].append(r)

    samples = []
    skipped = 0  # 无真值集对应（D12 报数量）
    for d, prs in by_d.items():
        if d not in truthset:
            skipped += 1
            continue
        prs.sort(key=lambda x: x["turn_idx"])  # 按原始话语顺序
        tsize = truthset[d]["truth_set_size"]
        remaining = int(tsize)
        for i, p in enumerate(prs):
            iu = p.get("info_units")
            iu = int(iu) if isinstance(iu, (int, float)) else 0
            iu = max(0, min(iu, 10))  # clamp 到 INFO_UNITS 范围
            samples.append({
                "openness": p["openness"],
                "leadingness": p["leadingness"],
                "multi_barreled": p["multi_barreled"],
                "relevance": p["relevance"],
                "definiteness": p["definiteness"],
                "completeness": p["completeness"],
                "info_units": iu,
                "remaining": remaining,
                "turn": i,
            })
            remaining = max(0, remaining - iu)
    return samples, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    config = json.load(open(args.config, encoding="utf-8"))
    alpha = args.alpha if args.alpha is not None else config.get("alpha", 1.0)

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}
    print(f"原始 {len(rows)} 条，过滤后有效 {len(valid)} 条，真值集对话数 {len(truthset)}")

    samples, skipped = build_sequential_samples(valid, truthset)
    print(f"序贯样本 {len(samples)} 条；无真值集对应而跳过的对话 {skipped} 个（D12）")

    sim = ConditionalSimulator.estimate_3d(samples, config, alpha=alpha)
    sim.save(Path(args.out))
    print(f"\n已保存 -> {args.out}\n")

    # ---------- D12：各档样本量 ----------
    turn_bins = config["turn_bins"]
    rem_bins = config["remaining_bins"]
    print("=" * 78)
    print("D12 各档样本量")
    print("=" * 78)
    tcnt = Counter(_bin_of(s["turn"], turn_bins) for s in samples)
    rcnt = Counter(_bin_of(s["remaining"], rem_bins) for s in samples)
    print("轮次档:", dict(tcnt))
    print("剩余量档:", dict(rcnt))

    # ---------- 验收6：3 维单元格样本量表 ----------
    print("\n" + "=" * 78)
    print("覆盖边界报告：info_units 的 3 维单元格样本量 (openness × 剩余量档 × 轮次档)")
    print("=" * 78)
    cell_n = sim.probs["_meta"]["cell_n"]["info_units"]
    all_cells = []
    low_cells = []
    for x in OPENNESS:
        for rb in [b["name"] for b in rem_bins]:
            for tb in [b["name"] for b in turn_bins]:
                n = cell_n.get((x, rb, tb), 0)
                all_cells.append(n)
                if n < 30:
                    low_cells.append((x, rb, tb, n))
    print(f"单元格总数 {len(all_cells)}，最小 {min(all_cells)}，最大 {max(all_cells)}，"
          f"中位 {sorted(all_cells)[len(all_cells)//2]}")
    print(f"样本 < 30 的低置信单元格 {len(low_cells)} 个：")
    for x, rb, tb, n in low_cells:
        print(f"  openness={x:6s} rbin={rb:4s} tbin={tb:4s} n={n}")

    # 最大可支撑档（n ≥ 30 的最大轮次档/剩余量档，按序）
    def max_supported(bins, cell_n, x, other_bins, axis):
        """axis: 0=按轮次档变化(固定剩余档), 1=按剩余档变化(固定轮次档)。粗略报边界。"""
        ok = set()
        for b in bins:
            bname = b["name"]
            all_ok = True
            for ob in [o["name"] for o in other_bins]:
                key = (x, bname, ob) if axis == 0 else (x, ob, bname)
                if cell_n.get(key, 0) < 30:
                    all_ok = False
                    break
            if all_ok:
                ok.add(bname)
        return ok

    print("\n最大可支撑边界（openness 分档看，n≥30 才视为可支撑）：")
    for x in OPENNESS:
        sup_t = [tb for tb in [b["name"] for b in turn_bins]
                 if all(cell_n.get((x, rb, tb), 0) >= 30 for rb in [b["name"] for b in rem_bins])]
        sup_r = [rb for rb in [b["name"] for b in rem_bins]
                 if all(cell_n.get((x, rb, tb), 0) >= 30 for tb in [b["name"] for b in turn_bins])]
        print(f"  {x:6s}: 全档支撑的轮次档={sup_t or '无'}  全档支撑的剩余量档={sup_r or '无'}")

    # ---------- H3 样本分档后剩余 ----------
    print("\n" + "=" * 78)
    print("H3（复合性→漏答）可用样本分档后：multi_barreled='multi' 的单元格样本量")
    print("=" * 78)
    multi_cell = sim.probs["_meta"]["cell_n"]["completeness"]
    multi_total = sum(1 for s in samples if s["multi_barreled"] == "multi")
    print(f"multi 样本总数 {multi_total} 条；分档后（multi × rbin × tbin）单元格：")
    for rb in [b["name"] for b in rem_bins]:
        for tb in [b["name"] for b in turn_bins]:
            n = multi_cell.get(("multi", rb, tb), 0)
            if n > 0:
                flag = "⚠️空/低置信(合并档估计)" if n < 30 else ""
                print(f"  multi × {rb:4s} × {tb:4s}: n={n} {flag}")
    empty = [(rb, tb) for rb in [b["name"] for b in rem_bins]
             for tb in [b["name"] for b in turn_bins]
             if multi_cell.get(("multi", rb, tb), 0) == 0]
    print(f"multi 空单元格 {len(empty)} 个（这些格由 Laplace 平滑兜底，等价合并档估计）")


if __name__ == "__main__":
    main()
