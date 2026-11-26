"""阶段 A 结果统计：配对检验 open vs closed 的溢出 AIU 数。

输出：描述统计 + 配对 t 检验 + Wilcoxon 符号秩检验。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze(csv_path: str):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    # 按 (case, info_point, repeat) 配对 open 与 closed
    paired = defaultdict(dict)
    for r in rows:
        key = (r["case_id"], r["info_point"], r["repeat"])
        paired[key][r["style"]] = int(r["spillover_count"])

    diffs = []
    open_vals, closed_vals = [], []
    for k, v in paired.items():
        if "open" in v and "closed" in v:
            diffs.append(v["open"] - v["closed"])
            open_vals.append(v["open"])
            closed_vals.append(v["closed"])

    n = len(diffs)
    import statistics
    mean_d = statistics.mean(diffs)
    sd_d = statistics.stdev(diffs) if n > 1 else 0
    t = mean_d / (sd_d / (n ** 0.5)) if sd_d > 0 else float("inf")

    print(f"配对样本数 n = {n}")
    print(f"open  溢出: mean={statistics.mean(open_vals):.3f}, median={statistics.median(open_vals)}")
    print(f"closed 溢出: mean={statistics.mean(closed_vals):.3f}, median={statistics.median(closed_vals)}")
    print(f"差值(open-closed): mean={mean_d:.3f}, sd={sd_d:.3f}")
    print(f"配对 t 统计量 = {t:.2f}")
    print(f"open>closed 的配对占比 = {sum(1 for d in diffs if d > 0)}/{n} "
          f"({sum(1 for d in diffs if d > 0)/n:.0%})")

    # Wilcoxon 符号秩检验（scipy 可用时）
    try:
        from scipy import stats
        t_stat, t_p = stats.ttest_rel(open_vals, closed_vals)
        w_stat, w_p = stats.wilcoxon(open_vals, closed_vals)
        print(f"\n配对 t 检验: t={t_stat:.3f}, p={t_p:.3e}")
        print(f"Wilcoxon 符号秩: W={w_stat:.0f}, p={w_p:.3e}")
    except ImportError:
        print("\n(scipy 未安装，跳过精确 p 值；已给出 t 统计量与占比)")

    # 每个病例的效应（case 级聚合）
    print("\n===== 病例级汇总（open均值 vs closed均值）=====")
    by_case_open = defaultdict(list)
    by_case_closed = defaultdict(list)
    for k, v in paired.items():
        if "open" in v and "closed" in v:
            by_case_open[k[0]].append(v["open"])
            by_case_closed[k[0]].append(v["closed"])
    n_pos = 0
    for cid in sorted(by_case_open):
        mo = statistics.mean(by_case_open[cid])
        mc = statistics.mean(by_case_closed[cid])
        flag = "+" if mo > mc else ("-" if mo < mc else "=")
        if mo > mc:
            n_pos += 1
        print(f"  {cid}: open={mo:.2f} closed={mc:.2f} {flag}")
    print(f"\nopen>closed 的病例数 = {n_pos}/{len(by_case_open)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str)
    args = ap.parse_args()
    analyze(args.csv)
