"""计算两个标注者（如 flash vs plus）的逐维度一致性。

用法:
  python scripts/annotate_agreement.py data/annot_200_flash.jsonl data/annot_200_plus.jsonl

输出每个维度的: 一致率 + Cohen's κ（名义/线性加权）+ info_units 的相关。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def align(a: list[dict], b: list[dict]) -> tuple[list, list]:
    """按 (dialogue_id, turn_idx) 对齐，返回两个值列表。"""
    bmap = {(x["dialogue_id"], x["turn_idx"]): x for x in b}
    va, vb = [], []
    for x in a:
        y = bmap.get((x["dialogue_id"], x["turn_idx"]))
        if y is not None:
            va.append(x)
            vb.append(y)
    return va, vb


def cohen_kappa(va: list, vb: list, ordered: bool = False, labels: list[str] | None = None) -> float:
    """Cohen's κ；ordered=True 时用线性加权 κ。"""
    if labels is None:
        labels = sorted(set(va) | set(vb))
    n = len(va)
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    # 混淆矩阵
    O = [[0] * k for _ in range(k)]
    for x, y in zip(va, vb):
        O[idx[x]][idx[y]] += 1
    # 权重矩阵：非序数 → 对角一致；序数 → 线性加权
    if ordered:
        W = [[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    else:
        W = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
    # 边际
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    po = sum(O[i][j] * W[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(row[i] * col[j] * W[i][j] for i in range(k) for j in range(k)) / (n * n)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def gwet_ac1(va: list, vb: list, labels: list[str] | None = None) -> float:
    """Gwet's AC1：对类别不平衡分布比 Cohen's κ 更稳健。"""
    if labels is None:
        labels = sorted(set(va) | set(vb))
    n = len(va)
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    row = [0] * k
    col = [0] * k
    agree = 0
    for x, y in zip(va, vb):
        row[idx[x]] += 1
        col[idx[y]] += 1
        if x == y:
            agree += 1
    pa = agree / n
    # 边际占比
    pi = [(row[i] + col[i]) / (2 * n) for i in range(k)]
    pe = sum(p * (1 - p) for p in pi) / (k - 1) if k > 1 else 0.0
    if pe == 1:
        return 1.0
    return (pa - pe) / (1 - pe)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    args = ap.parse_args()

    a = load(Path(args.file_a))
    b = load(Path(args.file_b))
    va, vb = align(a, b)
    print(f"对齐样本数: {len(va)}")

    # 维度定义: (字段名, 是否序数, 固定标签)
    dims = [
        ("is_question", False, [0, 1]),
        ("openness", True, ["n/a", "closed", "semi", "open"]),
        ("leadingness", True, ["n/a", "neutral", "mild", "strong"]),
        ("multi_barreled", False, ["n/a", "single", "multi"]),
        ("follow_up", False, ["n/a", "surface", "deepen", "new"]),
        ("definiteness", True, ["agree", "vague", "exact"]),
        ("completeness", False, [0, 1]),
        ("relevance", False, ["on-topic", "spillover", "off-topic"]),
    ]

    print(f"\n{'维度':16s} {'一致率':>8s} {'κ':>7s} {'AC1':>7s}   说明")
    for field, ordered, labels in dims:
        xs = [x.get(field) for x in va]
        ys = [x.get(field) for x in vb]
        # 只保留双方都有值的
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if not pairs:
            continue
        xs2, ys2 = zip(*pairs)
        agree = sum(1 for x, y in zip(xs2, ys2) if x == y) / len(xs2)
        kappa = cohen_kappa(list(xs2), list(ys2), ordered=ordered,
                            labels=sorted(set(xs2) | set(ys2)))
        ac1 = gwet_ac1(list(xs2), list(ys2), labels=sorted(set(xs2) | set(ys2)))
        # 分布（flash 侧）
        dist = Counter(xs2)
        dist_s = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items(), key=lambda t: str(t[0])))
        flag = "✅" if ac1 >= 0.7 else ("⚠️" if ac1 >= 0.4 else "❌")
        print(f"{field:16s} {agree*100:7.1f}% {kappa:7.3f} {ac1:7.3f} {flag}  分布[{dist_s}]")

    # info_units（连续）
    xs = [x.get("info_units") for x in va]
    ys = [x.get("info_units") for x in vb]
    pairs = [(x, y) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if pairs:
        xs2, ys2 = zip(*pairs)
        r = pearson(xs2, ys2)
        mae = sum(abs(x - y) for x, y in zip(xs2, ys2)) / len(xs2)
        print(f"\ninfo_units: Pearson r={r:.3f}, MAE={mae:.2f}, flash均值={sum(xs2)/len(xs2):.2f}, plus均值={sum(ys2)/len(ys2):.2f}")


if __name__ == "__main__":
    main()
