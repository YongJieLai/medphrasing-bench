"""M2b 回放检验：逐轮回放（状态依赖）测 B1/B2 结构判据 + A 组三率 + 阴性对照。

与旧版（逐条独立采样）的本质区别：逐轮维护双状态账本，
每轮采样吃（措辞, remaining_n, turn_index），因此能测「状态依赖」类判据。

判据（只测 B1/B2，B3/B4/C 组留 M2c）：
  B1 患者间异质性：模拟的患者间披露量方差 / 真人方差 ≥ 0.8
  B2 残量效应：对话内相对披露量随轮次的斜率，符号一致、相对差 ≤ 30%
  A 组三率不倒退（溢出/附和/漏答）
  阴性对照：1 维模拟器（无状态）必须在 B1/B2 上失败（D11 失败路径）

用法:
  python scripts/replay_validation.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --config config/m2b_config.json \
      --out outputs/20261126_m2b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import ConditionalSimulator, _bin_of
from src.disclosure_ledger import DisclosureLedger
from scripts.calibrate_simulator import valid_rows, build_sequential_samples

RATE_KEYS = ("spillover_rate", "agree_rate", "incomplete_rate")


# ----------------------------------------------------------------------
# 真实披露量（info_units）序列
# ----------------------------------------------------------------------
def real_sequences(valid: list[dict], truthset: dict) -> dict[int, list[tuple]]:
    by_d = defaultdict(list)
    for r in valid:
        by_d[r["dialogue_id"]].append(r)
    seqs = {}
    for d, prs in by_d.items():
        if d not in truthset:
            continue
        prs.sort(key=lambda x: x["turn_idx"])
        seqs[d] = [(i, p.get("info_units") or 0) for i, p in enumerate(prs)]
    return seqs


def replay_3d(sim: ConditionalSimulator, dialogues: dict, truthset: dict,
              rng: np.random.Generator) -> dict[int, list[tuple]]:
    """逐轮回放：每轮更新账本，采样 (措辞, remaining, turn)。"""
    seqs = {}
    for d, prs in dialogues.items():
        tsize = truthset[d]["truth_set_size"]
        ledger = DisclosureLedger(tsize)
        seq = []
        for i, p in enumerate(prs):
            b = sim.sample_behavior(
                p["openness"], p["leadingness"], p["multi_barreled"], rng,
                remaining_n=ledger.remaining(), turn_index=i, dialogue_baseline=tsize)
            iu = b.get("info_units") or 0
            ledger.record(iu)
            seq.append((i, iu))
        seqs[d] = seq
    return seqs


def replay_1d(sim_iu_by_openness: dict, dialogues: dict,
              rng: np.random.Generator) -> dict[int, list[tuple]]:
    """阴性对照：1 维模拟器，披露量只由 openness 决定（无状态、无患者异质、无轮次效应）。"""
    seqs = {}
    for d, prs in dialogues.items():
        seq = []
        for i, p in enumerate(prs):
            iu = sim_iu_by_openness.get(p["openness"], 1.63)
            seq.append((i, iu))
        seqs[d] = seq
    return seqs


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def totals(seqs: dict) -> dict[int, float]:
    """每对话披露总量。"""
    return {d: sum(iu for _, iu in seq) for d, seq in seqs.items()}


def within_dialogue_slope(seqs: dict) -> dict:
    """对话内相对披露量随轮次的斜率（对话内中心化后线性回归）。"""
    xs, ys = [], []
    for d, seq in seqs.items():
        if len(seq) < 2:
            continue
        ts = np.array([t for t, _ in seq], dtype=float)
        ius = np.array([iu for _, iu in seq], dtype=float)
        centered = ius - ius.mean()
        xs.extend(ts.tolist())
        ys.extend(centered.tolist())
    if len(xs) < 2:
        return {"slope": float("nan"), "n": len(xs)}
    xs = np.array(xs)
    ys = np.array(ys)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return {"slope": slope, "n": len(xs), "intercept": float(np.polyfit(xs, ys, 1)[1])}


def empirical_rates(pairs: list[dict]) -> dict:
    n = len(pairs)
    return {
        "spillover_rate": sum(1 for p in pairs if p["relevance"] == "spillover") / n,
        "agree_rate": sum(1 for p in pairs if p["definiteness"] == "agree") / n,
        "incomplete_rate": sum(1 for p in pairs if p["completeness"] == 0) / n,
    }


def simulated_rates_3d(sim: ConditionalSimulator, dialogues: dict, truthset: dict,
                       rng: np.random.Generator, R: int) -> dict:
    """3 维模拟器回放 R 次，三率的均值。"""
    acc = {k: [] for k in RATE_KEYS}
    for _ in range(R):
        spill = agree = incomplete = n = 0
        for d, prs in dialogues.items():
            ledger = DisclosureLedger(truthset[d]["truth_set_size"])
            for i, p in enumerate(prs):
                b = sim.sample_behavior(
                    p["openness"], p["leadingness"], p["multi_barreled"], rng,
                    remaining_n=ledger.remaining(), turn_index=i,
                    dialogue_baseline=truthset[d]["truth_set_size"])
                ledger.record(b.get("info_units") or 0)
                spill += (b["relevance"] == "spillover")
                agree += (b["definiteness"] == "agree")
                incomplete += (b["completeness"] == 0)
                n += 1
        acc["spillover_rate"].append(spill / n)
        acc["agree_rate"].append(agree / n)
        acc["incomplete_rate"].append(incomplete / n)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def simulated_rates_1d(sim: ConditionalSimulator, dialogues: dict,
                       rng: np.random.Generator, R: int) -> dict:
    acc = {k: [] for k in RATE_KEYS}
    for _ in range(R):
        spill = agree = incomplete = n = 0
        for d, prs in dialogues.items():
            for i, p in enumerate(prs):
                b = sim.sample_behavior(p["openness"], p["leadingness"], p["multi_barreled"], rng)
                spill += (b["relevance"] == "spillover")
                agree += (b["definiteness"] == "agree")
                incomplete += (b["completeness"] == 0)
                n += 1
        acc["spillover_rate"].append(spill / n)
        acc["agree_rate"].append(agree / n)
        acc["incomplete_rate"].append(incomplete / n)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = json.load(open(args.config, encoding="utf-8"))
    seed = config["seed"]
    R = config["n_replay_runs"]
    holdout_frac = config["holdout_frac"]

    rng = np.random.default_rng(seed)

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}

    # 划分 train / holdout（按对话，seed 固定）
    by_d = defaultdict(list)
    for r in valid:
        by_d[r["dialogue_id"]].append(r)
    dial_ids = sorted(by_d.keys())
    rng.shuffle(dial_ids)
    n_hold = int(len(dial_ids) * holdout_frac)
    hold_ids = set(dial_ids[:n_hold])
    train_ids = set(dial_ids[n_hold:])

    train_pairs = [r for d in train_ids for r in by_d[d]]
    hold_pairs = [r for d in hold_ids for r in by_d[d]]
    hold_dialogues = {d: sorted(by_d[d], key=lambda x: x["turn_idx"]) for d in hold_ids}
    print(f"训练集 {len(train_ids)} 对话 / {len(train_pairs)} 条 | "
          f"回放集 {len(hold_ids)} 对话 / {len(hold_pairs)} 条")

    # 估 3 维表（训练集）
    train_valid = [r for r in valid if r["dialogue_id"] in train_ids]
    train_samples, _ = build_sequential_samples(train_valid, truthset)
    sim3 = ConditionalSimulator.estimate_3d(train_samples, config)
    print(f"3 维表估计完成（训练样本 {len(train_samples)} 条）")

    # 1 维表（阴性对照，训练集）
    sim1 = ConditionalSimulator.estimate(train_pairs)

    # 1 维对照的披露量：E[iu | openness]（训练集）
    iu_by_open = defaultdict(list)
    for s in train_samples:
        iu_by_open[s["openness"]].append(s["info_units"])
    sim1_iu = {o: float(np.mean(v)) for o, v in iu_by_open.items()}
    print(f"1 维对照披露量（E[iu|openness]）: { {k: round(v,3) for k,v in sim1_iu.items()} }")

    # ---------- 真实序列 ----------
    real_seqs = real_sequences([r for r in valid if r["dialogue_id"] in hold_ids], truthset)
    real_totals = totals(real_seqs)
    real_slope = within_dialogue_slope(real_seqs)
    real_rates = empirical_rates(hold_pairs)

    # ---------- 3 维回放（R 次取均值） ----------
    sim3_totals_mean = defaultdict(list)
    sim3_slopes = []
    for _ in range(R):
        seqs = replay_3d(sim3, hold_dialogues, truthset, rng)
        for d, tot in totals(seqs).items():
            sim3_totals_mean[d].append(tot)
        sim3_slopes.append(within_dialogue_slope(seqs)["slope"])
    sim3_totals_mean = {d: float(np.mean(v)) for d, v in sim3_totals_mean.items()}
    sim3_slope = float(np.mean([s for s in sim3_slopes if not np.isnan(s)]))
    sim3_rates = simulated_rates_3d(sim3, hold_dialogues, truthset, rng, R)

    # ---------- 1 维回放（阴性对照） ----------
    sim1_totals_mean = defaultdict(list)
    sim1_slopes = []
    for _ in range(R):
        seqs = replay_1d(sim1_iu, hold_dialogues, rng)
        for d, tot in totals(seqs).items():
            sim1_totals_mean[d].append(tot)
        sim1_slopes.append(within_dialogue_slope(seqs)["slope"])
    sim1_totals_mean = {d: float(np.mean(v)) for d, v in sim1_totals_mean.items()}
    sim1_slope = float(np.mean([s for s in sim1_slopes if not np.isnan(s)]))
    sim1_rates = simulated_rates_1d(sim1, hold_dialogues, rng, R)

    # ---------- 判定 ----------
    def var_of(d: dict) -> float:
        vals = list(d.values())
        return float(np.var(vals))

    var_real = var_of(real_totals)
    var_sim3 = var_of(sim3_totals_mean)
    var_sim1 = var_of(sim1_totals_mean)
    b1_3d = var_sim3 / var_real if var_real > 0 else float("nan")
    b1_1d = var_sim1 / var_real if var_real > 0 else float("nan")

    def slope_check(s_sim, s_real):
        if np.isnan(s_sim) or np.isnan(s_real) or abs(s_real) < 1e-9:
            return {"same_sign": False, "rel_diff": float("nan"), "pass": False}
        same_sign = (s_sim * s_real) > 0
        rel_diff = abs(s_sim - s_real) / abs(s_real)
        return {"same_sign": bool(same_sign), "rel_diff": rel_diff,
                "pass": bool(same_sign) and rel_diff <= 0.30}

    b2_3d = slope_check(sim3_slope, real_slope["slope"])
    b2_1d = slope_check(sim1_slope, real_slope["slope"])

    # A 组：三率差 ≤ 3pp
    a_pass = {}
    for k in RATE_KEYS:
        diff = abs(sim3_rates[k] - real_rates[k]) * 100
        a_pass[k] = {"real": real_rates[k], "sim3": sim3_rates[k], "sim1": sim1_rates[k],
                     "diff_pp": diff, "pass": diff <= 3.0}

    # ---------- 输出 ----------
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "holdout": {"n_dialogues": len(hold_ids), "n_pairs": len(hold_pairs)},
        "B1": {
            "var_real": var_real, "var_sim3": var_sim3, "var_sim1": var_sim1,
            "ratio_3d": b1_3d, "ratio_1d": b1_1d,
            "pass_3d": b1_3d >= 0.8, "pass_1d": b1_1d >= 0.8,
            "threshold": 0.8,
        },
        "B2": {
            "slope_real": real_slope, "slope_sim3": sim3_slope, "slope_sim1": sim1_slope,
            "check_3d": b2_3d, "check_1d": b2_1d,
            "threshold": "符号一致 且 相对差 ≤ 30%",
        },
        "A_group": a_pass,
        "negative_control": {
            "B1_fails": b1_1d < 0.8,
            "B2_fails": not b2_1d["pass"],
            "note": "1 维模拟器必须 B1/B2 都失败（D11 失败路径）",
        },
        "rates_full": {
            "real": real_rates, "sim3": sim3_rates, "sim1": sim1_rates,
        },
    }
    json.dump(result, open(outdir / "replay_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)

    # 打印
    print("\n" + "=" * 78)
    print("B1 患者间异质性（披露总量方差比，阈值 ≥ 0.8）")
    print("=" * 78)
    print(f"  真人方差      = {var_real:.3f}")
    print(f"  3维模拟方差   = {var_sim3:.3f}  比值 = {b1_3d:.3f}  {'✅' if b1_3d>=0.8 else '❌'}")
    print(f"  1维对照方差   = {var_sim1:.3f}  比值 = {b1_1d:.3f}  {'✅(异常)' if b1_1d>=0.8 else '❌(应失败)'}")

    print("\n" + "=" * 78)
    print("B2 残量效应（对话内相对披露量随轮次斜率，符号一致 + 相对差 ≤ 30%）")
    print("=" * 78)
    print(f"  真人斜率      = {real_slope['slope']:.4f}  (n={real_slope['n']})")
    print(f"  3维模拟斜率   = {sim3_slope:.4f}  相对差={b2_3d['rel_diff']:.1%}  "
          f"{'✅' if b2_3d['pass'] else '❌'}")
    print(f"  1维对照斜率   = {sim1_slope:.4f}  相对差={b2_1d['rel_diff']:.1%}  "
          f"{'✅(异常)' if b2_1d['pass'] else '❌(应失败)'}")

    print("\n" + "=" * 78)
    print("A 组三率（模拟 vs 真实，差 ≤ 3pp）")
    print("=" * 78)
    labels = {"spillover_rate": "溢出率", "agree_rate": "附和率", "incomplete_rate": "漏答率"}
    for k in RATE_KEYS:
        print(f"  {labels[k]:6s}: 真实 {a_pass[k]['real']*100:5.2f}%  "
              f"3维 {a_pass[k]['sim3']*100:5.2f}%  1维 {a_pass[k]['sim1']*100:5.2f}%  "
              f"3维差 {a_pass[k]['diff_pp']:4.2f}pp {'✅' if a_pass[k]['pass'] else '❌'}")

    print("\n" + "=" * 78)
    print("阴性对照（D11）：1 维模拟器必须在 B1/B2 失败")
    print("=" * 78)
    print(f"  B1 失败: {result['negative_control']['B1_fails']}  "
          f"B2 失败: {result['negative_control']['B2_fails']}")

    overall_b = b1_3d >= 0.8 and b2_3d["pass"]
    overall_a = all(a_pass[k]["pass"] for k in RATE_KEYS)
    print(f"\n>>> M2b 判定：B 组 {'通过' if overall_b else '未通过'}，"
          f"A 组 {'通过(不倒退)' if overall_a else '倒退'}")

    # run.log + config 快照（D8 四件套）
    (outdir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写出 -> {outdir}/replay_result.json")
    print(f"config 快照 -> {outdir}/config.json")


if __name__ == "__main__":
    main()
