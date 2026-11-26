"""M2a 核查 1：opening 阶段的 openness 分布（回答 R6）。

R6 风险：语料里 Cramér's V 仅 0.075–0.160，说明「开场并不更开放」，
与 Calgary-Cambridge 漏斗结构（开放开头、封闭收尾）矛盾。
主控怀疑：标注口径把「你哪里不舒服 / what brings you in」这类开场问句判成了 semi。

本脚本：
1. 按对话内 doctor 提问的相对位置三分（与 M1 同口径）得 stage。
2. 统计 opening / body / closing 各阶段的 open/semi/closed 分布。
3. 与全库分布对照。
4. 若 opening 中 open 占比 < 40%，输出 20 条被判 semi 的 opening 问句原文供主控判断。

用法:
  python scripts/check_opening_openness.py data/annot_all_flash.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OPENNESS = ["open", "semi", "closed"]


def add_stage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["dialogue_id", "turn_idx"]).reset_index(drop=True)
    df["turn_index"] = df.groupby("dialogue_id").cumcount()
    df["n_doctor_turns"] = df.groupby("dialogue_id")["dialogue_id"].transform("size")
    df["rel_pos"] = df["turn_index"] / np.maximum(1, df["n_doctor_turns"] - 1)
    df["stage"] = np.select(
        [df["rel_pos"] < 1 / 3, df["rel_pos"] <= 2 / 3],
        ["opening", "body"],
        default="closing",
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    df = pd.DataFrame(rows)
    q = df[(df["is_question"] == 1) & (df["openness"].isin(OPENNESS))].copy()
    q = add_stage(q)
    print(f"提问样本（openness 合法）: {len(q)}，对话数: {q['dialogue_id'].nunique()}")

    # 各阶段 openness 分布
    print("\n" + "=" * 66)
    print("各阶段 openness 分布")
    print("-" * 66)
    stages = ["opening", "body", "closing"]
    print(f"{'阶段':10s} {'n':>5s}  " + "  ".join(f"{o:>8s}" for o in OPENNESS))
    for s in stages:
        sub = q[q["stage"] == s]
        n = len(sub)
        dist = {o: (sub["openness"] == o).mean() * 100 for o in OPENNESS}
        line = f"{s:10s} {n:5d}  " + "  ".join(f"{dist[o]:7.1f}%" for o in OPENNESS)
        print(line)

    # 全库对照
    print("-" * 66)
    all_dist = {o: (q["openness"] == o).mean() * 100 for o in OPENNESS}
    print(f"{'全库':10s} {len(q):5d}  " + "  ".join(f"{all_dist[o]:7.1f}%" for o in OPENNESS))

    # R6 判定
    opening = q[q["stage"] == "opening"]
    open_pct = (opening["openness"] == "open").mean() * 100
    print("\n" + "=" * 66)
    print(f"opening 阶段 open 占比: {open_pct:.1f}%")
    if open_pct < 40:
        print("⚠️ 低于 40% 阈值 -> 触发 R6 核查，输出 20 条被判 semi 的 opening 问句原文：")
        semi_opening = opening[opening["openness"] == "semi"].head(20)
        for i, r in semi_opening.iterrows():
            print(f"  [{r['dialogue_id']}:{r['turn_idx']}] {r['doctor_utt']!r}")
    else:
        print("✅ opening 阶段 open 占比 ≥ 40%，R6 风险弱化")


if __name__ == "__main__":
    main()
