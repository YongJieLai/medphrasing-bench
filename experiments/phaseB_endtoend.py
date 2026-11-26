"""阶段 B（端到端）：自由问诊下的措辞策略对照。

医生 agent 分两组（开放式 / 封闭式提问策略），自由问诊若干轮后下诊断。
测量：累计溢出 AIU、信息覆盖率、最终诊断是否命中金标准。

H2：措辞策略 → 溢出差异 → 诊断证据/准确率差异。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cases import CASES
from src.patient import Patient
from src.doctor import Doctor
from src.metrics import extract_disclosed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MAX_TURNS = 8  # 问诊轮次上限


def run(case_ids=None, repeat=1):
    OUT.mkdir(exist_ok=True)
    out_csv = OUT / "phaseB_endtoend.csv"
    rows = []

    for case in CASES:
        if case_ids and case.id not in case_ids:
            continue
        for style in ("open", "closed"):
            for r in range(repeat):
                patient = Patient(case)
                doctor = Doctor(style)
                history: list[dict] = []
                disclosed: set[str] = set()
                diagnosis = ""

                q = doctor.first_question(case)
                for turn in range(MAX_TURNS):
                    a = patient.answer(q, history)
                    disclosed |= extract_disclosed(a, case)
                    history.append({"role": "assistant", "content": q})
                    history.append({"role": "user", "content": a})
                    nxt = doctor.next_turn(case, history)
                    if nxt.strip().startswith("诊断"):
                        diagnosis = nxt.strip()
                        break
                    q = nxt

                correct = case.diagnosis.split("（")[0] in diagnosis
                coverage = len(disclosed) / len(case.aius)
                rows.append({
                    "case_id": case.id, "style": style, "repeat": r,
                    "turns": len(history) // 2,
                    "disclosed_count": len(disclosed),
                    "coverage": round(coverage, 3),
                    "correct": correct,
                    "diagnosis": diagnosis,
                    "gold": case.diagnosis,
                })
                print(f"[{case.id}][{style}] 轮次{len(history)//2} "
                      f"覆盖率{coverage:.2f} 诊断命中{correct}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已写入 {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    run(args.cases, args.repeat)
