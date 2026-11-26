"""效应量测量：四维提问措辞 → 四属性患者披露。

用法:
  python scripts/analyze_effects.py data/annot_all_flash.jsonl

在标注数据上测量：
  1. 开放度 → 披露量 (info_units / word_count)
  2. 开放度 → 披露相关性 (spillover 率)
  3. 引导性 → 披露准确性 (agree 率)
  4. 复合性 → 披露完整性 (漏答率)
  5. 追问深度 → 披露准确性 (exact 率)

统计：组间 Cohen's d / 风险差 + 混合效应模型（dialogue_id 随机截距处理对话内相关）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def cohens_d(a: list, b: list) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def group_mean_ci(x: list) -> tuple[float, float]:
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
    return x.mean(), 1.96 * se


def mixed_linear(df: pd.DataFrame, y: str, x: str):
    """混合线性模型 y ~ x + (1|dialogue_id)，返回系数的 z/p。"""
    try:
        md = mixedlm(f"{y} ~ C({x})", df, groups=df["dialogue_id"]).fit(reml=True)
        return md
    except Exception as e:
        print(f"    [mixedlm 失败] {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    args = ap.parse_args()

    rows = load(Path(args.file))
    df = pd.DataFrame(rows)
    # 只看提问
    q = df[df["is_question"] == 1].copy()
    # info_units 转数值
    q["info_units"] = pd.to_numeric(q["info_units"], errors="coerce")
    q["word_count"] = pd.to_numeric(q["word_count"], errors="coerce")
    print(f"提问样本: {len(q)}")

    def report_cont(name, y, groups, order):
        print(f"\n{'='*60}\n【{name}】(因变量={y})")
        vals = {g: q[q[groups] == g][y].dropna().tolist() for g in order}
        for g in order:
            v = vals[g]
            m, ci = group_mean_ci(v)
            print(f"  {g:8s}: n={len(v):4d}  mean={m:6.2f}  ±{ci:5.2f}")
        # 首尾两组 Cohen's d
        if len(vals[order[0]]) > 1 and len(vals[order[-1]]) > 1:
            d = cohens_d(vals[order[0]], vals[order[-1]])
            print(f"  Cohen's d ({order[0]} vs {order[-1]}): {d:.3f}")
        # 混合模型
        sub = q[q[groups].isin(order)]
        if sub[groups].nunique() > 1:
            md = mixed_linear(sub, y, groups)
            if md is not None:
                print(f"  混合模型 p值(F): {md.pvalues[0]:.4f}" if False else "")
                # 打印各水平系数
                print(f"  [mixedlm 组间差异 p 值]")
                for k, v in md.pvalues.items():
                    if k != "Group Var":
                        print(f"    {k}: p={v:.4g}")

    def report_binary(name, target, groups, order, pos_val):
        print(f"\n{'='*60}\n【{name}】(阳性率={target}=={pos_val})")
        for g in order:
            sub = q[q[groups] == g]
            if len(sub) == 0:
                print(f"  {g:8s}: n=0")
                continue
            rate = (sub[target] == pos_val).mean()
            print(f"  {g:8s}: n={len(sub):4d}  阳性率={rate*100:5.1f}%")
        # 首尾两组风险差 + χ²
        a = q[q[groups] == order[0]]
        b = q[q[groups] == order[-1]]
        if len(a) and len(b):
            ra = (a[target] == pos_val).mean()
            rb = (b[target] == pos_val).mean()
            tab = pd.crosstab(q[q[groups].isin([order[0], order[-1]])][groups],
                              q[q[groups].isin([order[0], order[-1]])][target] == pos_val)
            chi2, p, dof, _ = stats.chi2_contingency(tab)
            print(f"  风险差 ({order[0]} - {order[-1]}): {(ra-rb)*100:+.1f}pp  χ² p={p:.4g}")

    # 1. 开放度 → 披露量
    report_cont("开放度 → 披露量", "info_units", "openness", ["closed", "semi", "open"])
    report_cont("开放度 → 披露量(词数)", "word_count", "openness", ["closed", "semi", "open"])

    # 2. 开放度 → spillover 率
    report_binary("开放度 → 披露相关性(spillover)", "relevance", "openness",
                  ["closed", "semi", "open"], "spillover")

    # 3. 引导性 → agree 率
    report_binary("引导性 → 披露准确性(agree附和)", "definiteness", "leadingness",
                  ["neutral", "mild", "strong"], "agree")

    # 4. 复合性 → 漏答率
    report_binary("复合性 → 披露完整性(漏答)", "completeness", "multi_barreled",
                  ["single", "multi"], 0)

    # 5. 追问深度 → exact 率
    report_binary("追问深度 → 披露准确性(exact)", "definiteness", "follow_up",
                  ["surface", "new", "deepen"], "exact")

    # 附加：spillover 时 info_units 是否更高（溢出=多披露的直接验证）
    print(f"\n{'='*60}\n【spillover 是否伴随更多披露】(info_units)")
    on = q[q["relevance"] == "on-topic"]["info_units"].dropna()
    sp = q[q["relevance"] == "spillover"]["info_units"].dropna()
    print(f"  on-topic: n={len(on)} mean={on.mean():.2f}")
    print(f"  spillover: n={len(sp)} mean={sp.mean():.2f}")
    if len(on) > 1 and len(sp) > 1:
        t, p = stats.ttest_ind(on, sp, equal_var=False)
        print(f"  t={t:.2f} p={p:.4g}  Cohen's d={cohens_d(on.tolist(), sp.tolist()):.3f}")


if __name__ == "__main__":
    main()
