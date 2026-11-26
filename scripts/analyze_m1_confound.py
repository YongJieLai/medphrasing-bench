"""M1 阶段共线排查：控制对话阶段（时机）后，措辞效应是否还在。

核心问题（charter R1）：测到的「措辞 → 披露」效应，有多少其实来自「对话阶段」？
开场问题天然更开放，开场患者本来也说得多——若不控制时机，措辞效应会被高估。

做法：
1. 给每个问答对补 turn_index（对话内第几个医生提问）与 stage（opening/body/closing，按相对位置三分）。
2. 对 H2/H3/H4（+ H5/H1 顺带）各跑：
   - 未控制:   logit(y) ~ x
   - 控制:     logit(y) ~ x + turn_index + C(stage)
   （均为 GEE + exchangeable 相关结构，以对话为单位处理相关，符合 D3）
3. 对比 x 系数（log-OR）控制前后的相对变化：
   - < 50%  → 效应稳健，M1 通过
   - >= 50% → 效应主要来自时机，需重写研究问题

用法:
  python scripts/analyze_m1_confound.py data/annot_all_flash.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def add_timing_fields(df: pd.DataFrame) -> pd.DataFrame:
    """按对话计算 turn_index（第几个医生提问）、n_doctor_turns、rel_pos、stage。"""
    # 每个对话内，按 turn_idx 排序，得到 doctor 提问的序号
    df = df.sort_values(["dialogue_id", "turn_idx"]).reset_index(drop=True)
    df["turn_index"] = df.groupby("dialogue_id").cumcount()
    df["n_doctor_turns"] = df.groupby("dialogue_id")["dialogue_id"].transform("size")
    df["rel_pos"] = df["turn_index"] / np.maximum(1, df["n_doctor_turns"] - 1)
    # 三分位：opening / body / closing
    df["stage"] = np.select(
        [df["rel_pos"] < 1 / 3, df["rel_pos"] <= 2 / 3],
        ["opening", "body"],
        default="closing",
    )
    return df


def gee_logit(df: pd.DataFrame, formula: str) -> dict:
    """二值因变量 GEE（exchangeable，按 dialogue_id 聚类），返回 x_val 系数相关量。"""
    try:
        model = GEE.from_formula(
            formula,
            groups="dialogue_id",
            family=Binomial(),
            cov_struct=Exchangeable(),
            data=df,
        )
        res = model.fit()
        return {
            "beta": res.params.get("x_val", np.nan),
            "se": res.bse.get("x_val", np.nan),
            "p": res.pvalues.get("x_val", np.nan),
            "n": int(len(df)),
            "n_dialogue": int(df["dialogue_id"].nunique()),
        }
    except Exception as e:
        return {"beta": np.nan, "se": np.nan, "p": np.nan,
                "n": int(len(df)), "n_dialogue": int(df["dialogue_id"].nunique()),
                "error": str(e)}


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V：两个分类变量的关联强度。"""
    tab = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.sum().sum()
    min_dim = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0


# 每个假设：y 列、x 列、x 的合法取值（ordinal 编码）、因变量说明
HYPOTHESES = {
    "H2_引导性→附和": {
        "x": "leadingness", "order": ["neutral", "mild", "strong"],
        "y_name": "agree", "make_y": lambda d: (d["definiteness"] == "agree").astype(int),
    },
    "H3_复合性→漏答": {
        "x": "multi_barreled", "order": ["single", "multi"],
        "y_name": "漏答", "make_y": lambda d: (d["completeness"] == 0).astype(int),
    },
    "H4_追问→精确": {
        "x": "follow_up", "order": ["surface", "new", "deepen"],
        "y_name": "exact", "make_y": lambda d: (d["definiteness"] == "exact").astype(int),
    },
    "H5_开放度→溢出": {
        "x": "openness", "order": ["closed", "semi", "open"],
        "y_name": "spillover", "make_y": lambda d: (d["relevance"] == "spillover").astype(int),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="系数变化阈值（默认 0.5 = 50%）")
    args = ap.parse_args()

    rows = load(Path(args.file))
    df = pd.DataFrame(rows)
    q = df[df["is_question"] == 1].copy()
    q = add_timing_fields(q)
    print(f"提问样本: {len(q)}，对话数: {q['dialogue_id'].nunique()}")
    print(f"stage 分布: {q['stage'].value_counts().to_dict()}")

    print("\n" + "=" * 78)
    print("共线诊断：措辞变量 与 stage 的关联强度（Cramér's V，越大越共线）")
    for name, h in HYPOTHESES.items():
        sub = q[q[h["x"]].isin(h["order"])]
        v = cramers_v(sub[h["x"]], sub["stage"])
        print(f"  {name:18s}  x={h['x']:14s}  V={v:.3f}")

    results = []
    print("\n" + "=" * 78)
    for name, h in HYPOTHESES.items():
        sub = q[q[h["x"]].isin(h["order"])].copy()
        sub["x_val"] = sub[h["x"]].map({v: i for i, v in enumerate(h["order"])})
        sub["y"] = h["make_y"](sub)

        print(f"\n{'='*78}\n【{name}】x={h['x']}，因变量={h['y_name']}")
        # 描述性：分档阳性率
        desc = sub.groupby(h["x"])["y"].agg(["count", "mean"])
        for idx, r in desc.iterrows():
            print(f"    {idx:8s}: n={int(r['count']):4d}  阳性率={r['mean']*100:5.1f}%")

        base = gee_logit(sub, "y ~ x_val")
        ctrl_t = gee_logit(sub, "y ~ x_val + turn_index")
        ctrl_s = gee_logit(sub, "y ~ x_val + C(stage)")
        ctrl_both = gee_logit(sub, "y ~ x_val + turn_index + C(stage)")

        b0, b1 = base["beta"], ctrl_both["beta"]
        if "error" in base or "error" in ctrl_both:
            print(f"    [GEE 失败] base={base.get('error')} ctrl={ctrl_both.get('error')}")
            continue

        z0 = b0 / base["se"] if base["se"] else np.nan
        rel_change = abs(b1 - b0) / abs(b0) * 100 if abs(b0) > 1e-9 else np.nan
        verdict = ""
        if abs(z0) < 1.96:
            verdict = "⚪ 未控制前本就不显著"
        elif rel_change < args.threshold * 100:
            verdict = "✅ 通过（控制后效应稳健）"
        else:
            verdict = "❌ 不通过（效应主要来自时机）"

        print(f"    未控制   : β={b0:+.4f}  se={base['se']:.4f}  z={z0:+.2f}  p={base['p']:.3g}")
        print(f"    只控turn : β={ctrl_t['beta']:+.4f}  p={ctrl_t['p']:.3g}")
        print(f"    只控stage: β={ctrl_s['beta']:+.4f}  p={ctrl_s['p']:.3g}")
        print(f"    联合控制 : β={b1:+.4f}  se={ctrl_both['se']:.4f}  p={ctrl_both['p']:.3g}")
        print(f"    系数变化 : {rel_change:.1f}%  ->  {verdict}")
        print(f"    (n={base['n']}, n_dialogue={base['n_dialogue']})")

        results.append({
            "hypothesis": name, "x": h["x"], "y": h["y_name"],
            "beta_uncontrolled": b0, "beta_controlled": b1,
            "z_uncontrolled": z0, "p_uncontrolled": base["p"],
            "p_controlled": ctrl_both["p"],
            "rel_change_pct": rel_change,
            "verdict": verdict,
            "n": base["n"], "n_dialogue": base["n_dialogue"],
        })

    # 汇总判定表
    print("\n" + "=" * 78)
    print("M1 判定汇总")
    print("-" * 78)
    print(f"{'假设':18s} {'未控β':>8s} {'控后β':>8s} {'变化%':>7s} {'未控p':>8s} {'控后p':>8s}  判定")
    for r in results:
        print(f"{r['hypothesis']:18s} {r['beta_uncontrolled']:+8.4f} "
              f"{r['beta_controlled']:+8.4f} {r['rel_change_pct']:6.1f}% "
              f"{r['p_uncontrolled']:8.3g} {r['p_controlled']:8.3g}  {r['verdict']}")


if __name__ == "__main__":
    main()
