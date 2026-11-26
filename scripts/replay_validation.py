"""M2 回放检验：条件概率表驱动的模拟器能否复现真实患者的披露分布。

做法（符合 D6「校准只认分布匹配」）：
  1. 按对话划分 holdout：随机取 200 段对话作回放集，其余作训练集（估计条件概率表）。
  2. 对回放集每条真实问句（用其措辞标注），条件概率表采样披露行为标签。
  3. bootstrap 得到模拟器三率分布；真实三率用 cluster bootstrap（按对话重采样）得 CI。
  4. 判定：三率差 ≤ 3pp 且 95% CI 重叠。

用法:
  python scripts/replay_validation.py data/annot_all_flash.jsonl --replay-dialogues 200 --seed 42 --bootstrap 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import ConditionalSimulator
from scripts.calibrate_simulator import valid_rows

RATE_KEYS = ("spillover_rate", "agree_rate", "incomplete_rate")


def empirical_rates(pairs: list[dict]) -> dict:
    n = len(pairs)
    return {
        "spillover_rate": sum(1 for p in pairs if p["relevance"] == "spillover") / n,
        "agree_rate": sum(1 for p in pairs if p["definiteness"] == "agree") / n,
        "incomplete_rate": sum(1 for p in pairs if p["completeness"] == 0) / n,
    }


def cluster_bootstrap_ci(dialogues: list[list[dict]], rng, B: int = 2000) -> dict:
    """按对话重采样（cluster bootstrap），返回三率的 2.5/50/97.5 分位。"""
    D = len(dialogues)
    rates = {k: [] for k in RATE_KEYS}
    for _ in range(B):
        idx = rng.integers(0, D, size=D)
        pairs = [p for i in idx for p in dialogues[i]]
        r = empirical_rates(pairs)
        for k in RATE_KEYS:
            rates[k].append(r[k])
    out = {}
    for k in RATE_KEYS:
        arr = np.asarray(rates[k])
        out[k] = {
            "ci_lo": np.percentile(arr, 2.5),
            "median": np.percentile(arr, 50),
            "ci_hi": np.percentile(arr, 97.5),
        }
    return out


def simulator_bootstrap(sim: ConditionalSimulator, replay_pairs: list[dict],
                        rng, B: int = 2000) -> dict:
    """模拟器逐条采样，B 次全量重放，得三率分布。"""
    rates = {k: [] for k in RATE_KEYS}
    for _ in range(B):
        r = sim.sample_rates(replay_pairs, rng)
        for k in RATE_KEYS:
            rates[k].append(r[k])
    out = {}
    for k in RATE_KEYS:
        arr = np.asarray(rates[k])
        out[k] = {
            "ci_lo": np.percentile(arr, 2.5),
            "median": np.percentile(arr, 50),
            "ci_hi": np.percentile(arr, 97.5),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--replay-dialogues", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    print(f"有效标注: {len(valid)} 条")

    # 按对话分组
    by_dialogue: dict[int, list[dict]] = {}
    for r in valid:
        by_dialogue.setdefault(r["dialogue_id"], []).append(r)
    dial_ids = list(by_dialogue.keys())
    print(f"对话数: {len(dial_ids)}")

    # 划分 holdout（200 段对话）
    rng.shuffle(dial_ids)
    replay_ids = set(dial_ids[: args.replay_dialogues])
    train_ids = set(dial_ids[args.replay_dialogues:])
    train_pairs = [r for i in train_ids for r in by_dialogue[i]]
    replay_pairs = [r for i in replay_ids for r in by_dialogue[i]]
    replay_dialogues = [by_dialogue[i] for i in dial_ids if i in replay_ids]
    print(f"训练集: {len(train_pairs)} 条 ({len(train_ids)} 对话) | "
          f"回放集: {len(replay_pairs)} 条 ({len(replay_ids)} 对话)")

    # 估计条件概率表（训练集）
    sim = ConditionalSimulator.estimate(train_pairs)

    # 真实三率（回放集 empirical + cluster bootstrap CI）
    real_emp = empirical_rates(replay_pairs)
    real_ci = cluster_bootstrap_ci(replay_dialogues, rng, args.bootstrap)

    # 模拟器三率（bootstrap）
    sim_ci = simulator_bootstrap(sim, replay_pairs, rng, args.bootstrap)

    # 判定
    print(f"\n{'='*72}")
    print(f"{'指标':16s} {'真实率':>8s} {'真实CI(95%)':>22s} {'模拟器率':>8s} {'模拟器CI':>22s} {'差(pp)':>8s}  判定")
    print(f"{'='*72}")
    labels = {"spillover_rate": "溢出率", "agree_rate": "附和率", "incomplete_rate": "漏答率"}
    all_pass = True
    for k in RATE_KEYS:
        re = real_emp[k]
        sm = sim_ci[k]["median"]
        diff = abs(sm - re) * 100
        # CI 重叠判定：模拟器 CI 与真实 CI 有交集
        overlap = not (sim_ci[k]["ci_hi"] < real_ci[k]["ci_lo"] or sim_ci[k]["ci_lo"] > real_ci[k]["ci_hi"])
        passed = (diff <= 3.0) and overlap
        all_pass = all_pass and passed
        rci = f"[{real_ci[k]['ci_lo']*100:.2f}, {real_ci[k]['ci_hi']*100:.2f}]"
        sci = f"[{sim_ci[k]['ci_lo']*100:.2f}, {sim_ci[k]['ci_hi']*100:.2f}]"
        print(f"{labels[k]:14s} {re*100:7.2f}% {rci:>20s} {sm*100:7.2f}% {sci:>20s} {diff:7.2f}  {'✅' if passed else '❌'}")

    print(f"{'='*72}")
    print(f"\nM2 回放检验{'通过' if all_pass else '未通过'}: 三率差 ≤3pp 且 95% CI 重叠" +
          ("（全部满足）" if all_pass else "（存在不满足项）"))


if __name__ == "__main__":
    main()
