"""M2a 核查 2：补 C(dialogue_id) 固定效应重估四假设（M1 遗留）。

M1 用 GEE（exchangeable）混合了 within / between 变异。主控要求补对话内固定效应，
确认措辞效应不是「对话间的横截面关联」，而是「对话内、给定同一患者，措辞变化伴随披露变化」。

做法：
1. 线性概率模型 y ~ x_val + C(dialogue_id)，OLS 估计，聚类稳健 SE（按 dialogue_id）。
   -> 纯 within 估计（每个对话一个固定效应，只利用对话内变异）。
2. 辅以 mixedlm 随机截距（within+between 加权）对照。
3. 报告四假设系数符号与显著性，与 M1 GEE 结果对照；符号翻转立即标出。

用法:
  python scripts/analyze_m1_within.py data/annot_all_flash.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 假设编号（v2 章程）：H2 引导→附和 / H3 复合→漏答 / H4 追问→精确 / H5 开放→溢出
HYPOTHESES = {
    "H2 引导性→附和": {
        "x": "leadingness", "order": ["neutral", "mild", "strong"],
        "make_y": lambda d: (d["definiteness"] == "agree").astype(int),
    },
    "H3 复合性→漏答": {
        "x": "multi_barreled", "order": ["single", "multi"],
        "make_y": lambda d: (d["completeness"] == 0).astype(int),
    },
    "H4 追问→精确": {
        "x": "follow_up", "order": ["surface", "new", "deepen"],
        "make_y": lambda d: (d["definiteness"] == "exact").astype(int),
    },
    "H5 开放度→溢出": {
        "x": "openness", "order": ["closed", "semi", "open"],
        "make_y": lambda d: (d["relevance"] == "spillover").astype(int),
    },
}


def lpm_within(sub: pd.DataFrame) -> dict:
    """线性概率模型 + 对话固定效应，聚类 SE。返回 x_val 系数。"""
    from statsmodels.formula.api import ols
    try:
        model = ols("y ~ x_val + C(dialogue_id)", data=sub)
        res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["dialogue_id"]})
        return {
            "beta": res.params["x_val"],
            "se": res.bse["x_val"],
            "p": res.pvalues["x_val"],
            "t": res.tvalues["x_val"],
        }
    except Exception as e:
        return {"beta": np.nan, "se": np.nan, "p": np.nan, "t": np.nan, "error": str(e)}


def mixedlm_random(sub: pd.DataFrame) -> dict:
    """mixedlm 随机截距（对照）。"""
    from statsmodels.formula.api import mixedlm
    try:
        md = mixedlm("y ~ x_val", sub, groups=sub["dialogue_id"]).fit(reml=True)
        return {
            "beta": md.params["x_val"],
            "se": md.bse["x_val"],
            "p": md.pvalues["x_val"],
            "t": md.tvalues["x_val"],
        }
    except Exception as e:
        return {"beta": np.nan, "se": np.nan, "p": np.nan, "t": np.nan, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    df = pd.DataFrame(rows)
    q = df[df["is_question"] == 1].copy()
    print(f"提问样本: {len(q)}，对话数: {q['dialogue_id'].nunique()}\n")

    print("=" * 88)
    print(f"{'假设':16s} {'LPMβ(within)':>12s} {'t':>7s} {'p':>9s}  {'mixedlmβ':>10s} {'t':>7s}  {'符号翻转?':>8s}")
    print("-" * 88)

    results = []
    for name, h in HYPOTHESES.items():
        sub = q[q[h["x"]].isin(h["order"])].copy()
        sub["x_val"] = sub[h["x"]].map({v: i for i, v in enumerate(h["order"])})
        sub["y"] = h["make_y"](sub)
        # 只保留有对话内变异的对话（x_val 在对话内非恒定）
        g = sub.groupby("dialogue_id")["x_val"]
        keep = g.transform("nunique") > 1
        sub_within = sub[keep]
        n_drop = len(sub) - len(sub_within)

        w = lpm_within(sub_within)
        m = mixedlm_random(sub)

        flipped = ""
        if "error" in w or np.isnan(w["beta"]):
            flipped = "⚠️ 失败"
        elif "error" not in m and not np.isnan(m["beta"]):
            flipped = "✅ 一致" if (w["beta"] * m["beta"] > 0) else "❌ 翻转!"

        print(f"{name:16s} {w['beta']:+12.4f} {w.get('t',0):+7.2f} {w['p']:9.3g}  "
              f"{m['beta']:+10.4f} {m.get('t',0):+7.2f}  {flipped}")
        if n_drop:
            print(f"  ^ {name}: 丢弃 {n_drop} 条（对话内措辞无变异的样本，无法用于 within 估计）")

        results.append({
            "hypothesis": name, "lpm_beta": w.get("beta"), "lpm_p": w.get("p"),
            "mixedlm_beta": m.get("beta"), "mixedlm_p": m.get("p"),
            "n": len(sub), "n_within": len(sub_within),
        })

    print("=" * 88)
    print("\n符号说明：LPM 系数为「x 每升一档，y 概率的变化（pp）」。")
    print("H2/H3/H5 预期为正（引导/复合/开放 → 附和/漏答/溢出上升），H4 预期为正（追问→精确上升）。")
    print("若 LPM(within) 与 mixedlm 符号一致且与 M1 GEE 方向相同，则效应是对话内真效应，非横截面伪相关。")


if __name__ == "__main__":
    main()
