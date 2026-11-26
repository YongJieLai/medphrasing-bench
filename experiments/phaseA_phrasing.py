"""阶段 A（核心 MVP）：措辞对照实验。

固定"想问的信息点"，只改变提问措辞（开放式 vs 封闭式），
测量患者同一轮回答中的"溢出 AIU 数"差异。

假设 H1：开放式措辞比封闭式措辞引发更多信息溢出。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cases import CASES, load_cases
from src.phrasing import phrasing_variants
from src.patient import Patient
from src.metrics import measure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def run(case_ids: list[str] | None, repeat: int, data: str | None):
    OUT.mkdir(exist_ok=True)
    cases = load_cases(ROOT / data) if data else CASES
    out_name = f"phaseA_{Path(data).stem if data else 'demo'}.csv"
    out_csv = OUT / out_name
    rows = []

    for case in cases:
        if case_ids and case.id not in case_ids:
            continue
        patient = Patient(case)
        print(f"\n===== 病例 {case.id}（{case.diagnosis}）=====")
        for v in phrasing_variants(case):
            targets = case.info_points[v["label"]]["targets"]
            for r in range(repeat):
                answer = patient.answer(v["question"])
                m = measure(answer, case, targets)
                rows.append({
                    "case_id": case.id,
                    "info_point": v["label"],
                    "style": v["style"],
                    "repeat": r,
                    "question": v["question"],
                    "answer": answer,
                    "target_hit_count": m["target_hit_count"],
                    "spillover_count": m["spillover_count"],
                    "target_hits": "; ".join(m["target_hits"]),
                    "spillover": "; ".join(m["spillover"]),
                })
                print(f"[{v['style']:6s}] {v['label']} (rep{r}) "
                      f"命中{ m['target_hit_count']} 溢出{ m['spillover_count']}")

    # 写 CSV
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 汇总：open vs closed 的平均溢出
    agg: dict[str, list[int]] = {}
    for r in rows:
        agg.setdefault(r["style"], []).append(r["spillover_count"])

    print(f"\n===== 汇总（溢出 AIU 数）=====")
    for style, vals in agg.items():
        print(f"{style:6s}: n={len(vals)}  mean={sum(vals)/len(vals):.3f}  "
              f"total={sum(vals)}")
    print(f"\n结果已写入 {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/ddxplus_50.json",
                    help="病例 JSON 路径（相对仓库根；不传则用内置样例）")
    ap.add_argument("--cases", nargs="*", default=None, help="指定病例 id（默认全部）")
    ap.add_argument("--repeat", type=int, default=1, help="每个条件重复采样次数")
    args = ap.parse_args()
    run(args.cases, args.repeat, args.data)
