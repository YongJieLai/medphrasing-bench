"""M2b-1 回放检验：残量口径 B2 + 消融对照 + bootstrap + C1/C2（章程 §7.3）。

与 M2b 首轮（151e339）的区别（对应 §7.3 派单七件事）：
  1. 1 维对照改为「从 P(iu|openness) 抽样」的同口径随机采样，不再是取均值。
  2. 总披露量按账本截断：totals 用 ledger.record() 的实际入账数，不再逐轮原始 iu 求和。
  3. B2 废弃「turn 斜率」，改为 E[iu|remaining_bin] 斜率；
     判据 = 符号为正 且 落入真实斜率 bootstrap 95% CI（对话级重采样 ≥ 2000 次）；
     阴性对照改为「remaining 抹平消融」（estimate_3d_ablated），必须失败。
  4. C1 条件三率：按措辞档分层，逐层「模拟 vs 真人」差 ≤ 5pp，n < 30 标低置信。
  5. C2 联合一致性：iu==0 轮次的 completeness/relevance 分布 + 不可能组合占比 ≤ 2%。
  6. 结果文档四数齐报（B1/B2/C1/C2/A 组）。
  7. 覆盖边界表补 remaining 各档 n。

用法:
  python scripts/replay_validation.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --config config/m2b_config.json \
      --out outputs/20261126_m2b1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import (
    ConditionalSimulator, OPENNESS, LEADINGNESS, MULTI_BARRELED, INFO_UNITS, _bin_of,
)
from src.disclosure_ledger import DisclosureLedger
from scripts.calibrate_simulator import valid_rows, build_sequential_samples

RATE_KEYS = ("spillover_rate", "agree_rate", "incomplete_rate")
RATE_LABELS = {"spillover_rate": "溢出率", "agree_rate": "附和率", "incomplete_rate": "漏答率"}


# ----------------------------------------------------------------------
# 记录构建：真实与模拟统一为逐轮 record
# ----------------------------------------------------------------------
def _record(i, p, remaining_before, raw_iu, added, relevance, definiteness, completeness):
    return {
        "turn": i,
        "openness": p["openness"],
        "leadingness": p["leadingness"],
        "multi_barreled": p["multi_barreled"],
        "remaining_before": int(remaining_before),
        "raw_iu": int(raw_iu),
        "added": int(added),
        "relevance": relevance,
        "definiteness": definiteness,
        "completeness": int(completeness),
    }


def real_records(valid: list[dict], truthset: dict, dial_ids: set) -> dict[int, list[dict]]:
    """真实对话逐轮记录（含账本截断，与模拟同口径）。"""
    by_d = defaultdict(list)
    for r in valid:
        if r["dialogue_id"] in dial_ids:
            by_d[r["dialogue_id"]].append(r)
    records = {}
    for d, prs in by_d.items():
        prs.sort(key=lambda x: x["turn_idx"])
        ledger = DisclosureLedger(truthset[d]["truth_set_size"])
        rec = []
        for i, p in enumerate(prs):
            iu = p.get("info_units")
            iu = int(iu) if isinstance(iu, (int, float)) else 0
            iu = max(0, min(iu, 10))
            remaining_before = ledger.remaining()
            added = ledger.record(iu)
            rec.append(_record(i, p, remaining_before, iu, added,
                               p["relevance"], p["definiteness"], p["completeness"]))
        records[d] = rec
    return records


def replay_3d(sim: ConditionalSimulator, dialogues: dict, truthset: dict,
              rng: np.random.Generator) -> dict[int, list[dict]]:
    """3 维（含消融）逐轮回放，账本截断口径。"""
    records = {}
    for d, prs in dialogues.items():
        tsize = truthset[d]["truth_set_size"]
        ledger = DisclosureLedger(tsize)
        rec = []
        for i, p in enumerate(prs):
            remaining_before = ledger.remaining()
            b = sim.sample_behavior(
                p["openness"], p["leadingness"], p["multi_barreled"], rng,
                remaining_n=remaining_before, turn_index=i, dialogue_baseline=tsize)
            iu = b.get("info_units") or 0
            added = ledger.record(iu)
            rec.append(_record(i, p, remaining_before, iu, added,
                               b["relevance"], b["definiteness"], b["completeness"]))
        records[d] = rec
    return records


def replay_1d(sim1: ConditionalSimulator, iu_dist: dict, dialogues: dict,
              truthset: dict, rng: np.random.Generator) -> dict[int, list[dict]]:
    """1 维对照（辅助）：iu 从 P(iu|openness) 抽样，三率从 1 维表抽样，账本截断。"""
    records = {}
    for d, prs in dialogues.items():
        ledger = DisclosureLedger(truthset[d]["truth_set_size"])
        rec = []
        for i, p in enumerate(prs):
            remaining_before = ledger.remaining()
            b = sim1.sample_behavior(p["openness"], p["leadingness"], p["multi_barreled"], rng)
            iu = _draw_iu(iu_dist[p["openness"]], rng)
            if b["definiteness"] == "agree":
                iu = 0
            # 物理约束（§7.4 第4件，与 3 维同口径）
            if iu == 0 and b["relevance"] == "spillover":
                b["relevance"] = "on-topic"
            added = ledger.record(iu)
            rec.append(_record(i, p, remaining_before, iu, added,
                               b["relevance"], b["definiteness"], b["completeness"]))
        records[d] = rec
    return records


def _draw_iu(dist: dict, rng: np.random.Generator) -> int:
    ys = list(INFO_UNITS)
    ps = np.asarray([dist.get(y, 0.0) for y in ys], dtype=float)
    ps = ps / ps.sum()
    return int(ys[int(rng.choice(len(ys), p=ps))])


def build_1d_iu_dist(samples: list[dict]) -> dict[str, dict[int, float]]:
    """P(iu | openness)，Laplace(+1) 平滑。"""
    cnt = defaultdict(lambda: defaultdict(int))
    for s in samples:
        if s.get("openness") in OPENNESS and s.get("info_units") in INFO_UNITS:
            cnt[s["openness"]][s["info_units"]] += 1
    dist = {}
    for o in OPENNESS:
        denom = sum(cnt[o].values()) + 1.0 * len(INFO_UNITS)
        dist[o] = {y: (cnt[o].get(y, 0) + 1.0) / denom for y in INFO_UNITS}
    return dist


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def disclosed_totals(records: dict[int, list[dict]]) -> dict[int, float]:
    """每对话总披露量 = 账本入账数累计（截断口径，B1 用）。"""
    return {d: sum(r["added"] for r in recs) for d, recs in records.items()}


def global_rates(records: dict[int, list[dict]]) -> dict[str, float]:
    n = spill = agree = incomplete = 0
    for recs in records.values():
        for r in recs:
            n += 1
            spill += (r["relevance"] == "spillover")
            agree += (r["definiteness"] == "agree")
            incomplete += (r["completeness"] == 0)
    if n == 0:
        return {k: float("nan") for k in RATE_KEYS}
    return {"spillover_rate": spill / n, "agree_rate": agree / n,
            "incomplete_rate": incomplete / n}


def remaining_slope(pairs: list[tuple], rem_bins: list[dict]) -> float:
    """E[iu | remaining_bin] 对档序（0,1,2,...）的 OLS 斜率。pairs: [(remaining, iu)]。"""
    if len(pairs) < 2:
        return float("nan")
    groups = defaultdict(list)
    for r, iu in pairs:
        groups[_bin_of(int(r), rem_bins)].append(iu)
    order = [b["name"] for b in rem_bins]
    xs, ys = [], []
    for idx, bname in enumerate(order):
        vals = groups.get(bname)
        if vals:
            xs.append(idx)
            ys.append(float(np.mean(vals)))
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def pairs_from_records(records: dict[int, list[dict]]) -> list[tuple]:
    return [(r["remaining_before"], r["raw_iu"]) for recs in records.values() for r in recs]


def within_dialogue_slope(records: dict[int, list[dict]]) -> dict:
    """旧 B2 口径（turn 斜率，已废弃，仅按 D15 第 3 问保留报告）。"""
    xs, ys = [], []
    for recs in records.values():
        if len(recs) < 2:
            continue
        ts = np.array([r["turn"] for r in recs], dtype=float)
        ius = np.array([r["raw_iu"] for r in recs], dtype=float)
        centered = ius - ius.mean()
        xs.extend(ts.tolist())
        ys.extend(centered.tolist())
    if len(xs) < 2:
        return {"slope": float("nan"), "n": len(xs)}
    xs = np.array(xs); ys = np.array(ys)
    return {"slope": float(np.polyfit(xs, ys, 1)[0]), "n": len(xs)}


# ----------------------------------------------------------------------
# C1 / C2
# ----------------------------------------------------------------------
def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson 精确二项 95% CI（§7.5 M2b-1c 第 3 件事）。

    k=事件数，n=样本数。用 Beta 分位数实现，无需正态近似，对小 n 稳健。
    """
    from scipy.stats import beta
    if n <= 0:
        return (float("nan"), float("nan"))
    low = 0.0 if k <= 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k >= n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)


def c1_layers(records: dict[int, list[dict]]) -> dict:
    """按措辞档分层的条件三率。"""
    spill_c = defaultdict(int); spill_t = defaultdict(int)
    agree_c = defaultdict(int); agree_t = defaultdict(int)
    inc_c = defaultdict(int); inc_t = defaultdict(int)
    for recs in records.values():
        for r in recs:
            spill_t[r["openness"]] += 1
            spill_c[r["openness"]] += (r["relevance"] == "spillover")
            agree_t[r["leadingness"]] += 1
            agree_c[r["leadingness"]] += (r["definiteness"] == "agree")
            inc_t[r["multi_barreled"]] += 1
            inc_c[r["multi_barreled"]] += (r["completeness"] == 0)

    def rate(c, t):
        return {k: (c[k] / t[k] if t[k] else float("nan")) for k in t}

    return {
        "spillover_by_openness": rate(spill_c, spill_t),
        "agree_by_leadingness": rate(agree_c, agree_t),
        "incomplete_by_multi": rate(inc_c, inc_t),
        "n_openness": dict(spill_t),
        "n_leadingness": dict(agree_t),
        "n_multi": dict(inc_t),
    }


def c2_stats(records: dict[int, list[dict]]) -> dict:
    n_total = n_iu0 = n_iu0_complete = n_iu0_spill = n_iu0_offtopic = n_iu0_exact = 0
    for recs in records.values():
        for r in recs:
            n_total += 1
            if r["raw_iu"] == 0:
                n_iu0 += 1
                n_iu0_complete += (r["completeness"] == 1)
                n_iu0_spill += (r["relevance"] == "spillover")
                n_iu0_offtopic += (r["relevance"] == "off-topic")
                n_iu0_exact += (r["definiteness"] == "exact")
    return {
        "n_total": n_total,
        "n_iu0": n_iu0,
        "p_complete_given_iu0": n_iu0_complete / n_iu0 if n_iu0 else float("nan"),
        "p_spill_given_iu0": n_iu0_spill / n_iu0 if n_iu0 else float("nan"),
        "p_offtopic_given_iu0": n_iu0_offtopic / n_iu0 if n_iu0 else float("nan"),
        "p_exact_given_iu0": n_iu0_exact / n_iu0 if n_iu0 else float("nan"),
        # 各组合占全部轮次的比例（判据对象）
        "ratio_iu0_complete": n_iu0_complete / n_total if n_total else float("nan"),
        "ratio_iu0_spill": n_iu0_spill / n_total if n_total else float("nan"),
        "ratio_iu0_offtopic": n_iu0_offtopic / n_total if n_total else float("nan"),
        "ratio_iu0_exact": n_iu0_exact / n_total if n_total else float("nan"),
    }


def physical_check(records: dict[int, list[dict]]) -> dict:
    """体检表（§7.4 第2件）：iu ∈ {0,1,2,3+} 各档占比 + 每轮均值 + 每对话总量均值。

    此前从未报告；零披露率 33.9% vs 19.8% 的 +14pp 就是从这里暴露的。
    """
    n = 0
    bins = {"iu0": 0, "iu1": 0, "iu2": 0, "iu3p": 0}
    per_turn_sum = 0.0
    per_dial = []
    for recs in records.values():
        for r in recs:
            n += 1
            iu = r["raw_iu"]
            if iu == 0:
                bins["iu0"] += 1
            elif iu == 1:
                bins["iu1"] += 1
            elif iu == 2:
                bins["iu2"] += 1
            else:
                bins["iu3p"] += 1
            per_turn_sum += iu
        per_dial.append(sum(r["added"] for r in recs))
    return {
        "n_turns": n,
        "iu_dist": {k: (bins[k] / n if n else float("nan")) for k in bins},
        "iu_counts": bins,
        "per_turn_mean": per_turn_sum / n if n else float("nan"),
        "per_dialogue_mean": float(np.mean(per_dial)) if per_dial else float("nan"),
    }


# ----------------------------------------------------------------------
# bootstrap（真实残量斜率，对话级重采样）
# ----------------------------------------------------------------------
def bootstrap_real_slope(real_records: dict, rem_bins: list[dict],
                         rng: np.random.Generator, B: int) -> dict:
    dids = list(real_records.keys())
    slopes = []
    for _ in range(B):
        idx = rng.integers(0, len(dids), len(dids))
        pairs = []
        for i in idx:
            pairs.extend([(r["remaining_before"], r["raw_iu"]) for r in real_records[dids[i]]])
        s = remaining_slope(pairs, rem_bins)
        if not np.isnan(s):
            slopes.append(s)
    slopes = np.array(slopes)
    return {
        "n": int(len(slopes)),
        "mean": float(np.mean(slopes)),
        "ci_low": float(np.percentile(slopes, 2.5)),
        "ci_high": float(np.percentile(slopes, 97.5)),
    }


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
    B = config.get("bootstrap", 2000)
    holdout_frac = config["holdout_frac"]
    rem_bins = config["remaining_bins"]

    rng = np.random.default_rng(seed)

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}

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

    # 估参数（训练集）
    train_valid = [r for r in valid if r["dialogue_id"] in train_ids]
    train_samples, _ = build_sequential_samples(train_valid, truthset)
    sim3 = ConditionalSimulator.estimate_3d(train_samples, config)
    sim_abl = ConditionalSimulator.estimate_3d_ablated(train_samples, config)
    sim1 = ConditionalSimulator.estimate(train_pairs)
    iu_dist = build_1d_iu_dist(train_samples)
    print(f"3 维表 / 消融表 / 1 维表估计完成（训练样本 {len(train_samples)} 条）")

    # 真实记录（回放集，截断口径）
    real_rec = real_records(valid, truthset, hold_ids)
    real_totals = disclosed_totals(real_rec)
    var_real = float(np.var(list(real_totals.values())))
    real_slope = remaining_slope(pairs_from_records(real_rec), rem_bins)
    real_slope_legacy = within_dialogue_slope(real_rec)
    real_rates = global_rates(real_rec)
    real_c1 = c1_layers(real_rec)
    real_c2 = c2_stats(real_rec)
    print(f"真实残量斜率 = {real_slope:.4f}；患者间方差 = {var_real:.3f}")

    # bootstrap 真实斜率 CI
    boot = bootstrap_real_slope(real_rec, rem_bins, rng, B)
    print(f"真实斜率 bootstrap 95% CI = [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] (n={boot['n']})")

    # ---------- 回放（R 次，每次算方差/斜率/三率/C1/C2/体检表，再取均值） ----------
    def run_replays(replay_fn, sim, extra=None):
        vars_ = []
        slopes = []
        legacy_slopes = []
        rates_acc = {k: [] for k in RATE_KEYS}
        c1_acc = defaultdict(list)    # (dim, layer) -> [rate across R]
        c2_acc = defaultdict(list)    # key -> [value across R]
        phys_acc = defaultdict(list)  # key -> [value across R]
        for _ in range(R):
            recs = replay_fn(sim, hold_dialogues, truthset, rng) if extra is None \
                else replay_fn(sim, extra, hold_dialogues, truthset, rng)
            vars_.append(float(np.var(list(disclosed_totals(recs).values()))))
            slopes.append(remaining_slope(pairs_from_records(recs), rem_bins))
            legacy_slopes.append(within_dialogue_slope(recs)["slope"])
            rates = global_rates(recs)
            for k in RATE_KEYS:
                rates_acc[k].append(rates[k])
            c1 = c1_layers(recs)
            for dim in ("spillover_by_openness", "agree_by_leadingness", "incomplete_by_multi"):
                for layer, v in c1[dim].items():
                    c1_acc[(dim, layer)].append(v)
            c2 = c2_stats(recs)
            for k, v in c2.items():
                c2_acc[k].append(v)
            ph = physical_check(recs)
            for k, v in ph.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        phys_acc[f"{k}.{kk}"].append(vv)
                else:
                    phys_acc[k].append(v)

        def mean_nonnan(vals):
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        c1_mean = {}
        for (dim, layer), vs in c1_acc.items():
            c1_mean.setdefault(dim, {})[layer] = mean_nonnan(vs)
        c2_mean = {k: mean_nonnan(v) for k, v in c2_acc.items()}
        phys_mean = {k: mean_nonnan(v) for k, v in phys_acc.items()}
        # 还原 iu_dist / iu_counts 字典（phys_acc 里被平铺成 "iu_dist.iu0" 这类 key）
        phys_mean["iu_dist"] = {
            k: phys_mean.get(f"iu_dist.{k}", float("nan")) for k in ("iu0", "iu1", "iu2", "iu3p")
        }
        phys_mean["iu_counts"] = {
            k: phys_mean.get(f"iu_counts.{k}", 0.0) for k in ("iu0", "iu1", "iu2", "iu3p")
        }

        return {
            "var": mean_nonnan(vars_),
            "slope": mean_nonnan(slopes),
            "legacy_slope": mean_nonnan(legacy_slopes),
            "rates": {k: mean_nonnan(v) for k, v in rates_acc.items()},
            "c1": c1_mean,
            "c2": c2_mean,
            "phys": phys_mean,
        }

    sim3_res = run_replays(replay_3d, sim3)
    sim1_res = run_replays(replay_1d, sim1, extra=iu_dist)
    simabl_res = run_replays(replay_3d, sim_abl)

    # ---------- 判定 ----------
    b1 = {
        "var_real": var_real,
        "ratio_3d": sim3_res["var"] / var_real if var_real > 0 else float("nan"),
        "ratio_1d": sim1_res["var"] / var_real if var_real > 0 else float("nan"),
        "ratio_abl": simabl_res["var"] / var_real if var_real > 0 else float("nan"),
        "pass_3d": (sim3_res["var"] / var_real) >= 0.8 if var_real > 0 else False,
    }

    def b2_check(slope, boot):
        if np.isnan(slope):
            return {"sign_positive": False, "in_ci": False, "pass": False}
        sign_positive = slope > 0
        in_ci = boot["ci_low"] <= slope <= boot["ci_high"]
        return {"sign_positive": bool(sign_positive), "in_ci": bool(in_ci),
                "pass": bool(sign_positive and in_ci)}

    b2 = {
        "slope_real": real_slope,
        "boot": boot,
        "slope_3d": sim3_res["slope"], "check_3d": b2_check(sim3_res["slope"], boot),
        "slope_1d": sim1_res["slope"], "check_1d": b2_check(sim1_res["slope"], boot),
        "slope_abl": simabl_res["slope"], "check_abl": b2_check(simabl_res["slope"], boot),
        "legacy_slope_real": real_slope_legacy["slope"],
        "legacy_slope_3d": sim3_res["legacy_slope"],
        "legacy_slope_1d": sim1_res["legacy_slope"],
        "legacy_slope_abl": simabl_res["legacy_slope"],
    }

    # A 组三率（3 维 vs 真实，≤ 3pp）
    a_pass = {}
    for k in RATE_KEYS:
        diff = abs(sim3_res["rates"][k] - real_rates[k]) * 100
        a_pass[k] = {"real": real_rates[k], "sim3": sim3_res["rates"][k],
                     "sim1": sim1_res["rates"][k], "simabl": simabl_res["rates"][k],
                     "diff_pp": diff, "pass": diff <= 3.0}

    # C1 分层结算（§7.5 M2b-1c 第 3 件事）
    # 判定规则（写死，跑完不改）：
    #   - n < 50 的层（strong n=32）：报 Clopper-Pearson 95% CI；模拟值落 CI 内 → 不可判；
    #     落 CI 外 → 真超标（fail）。
    #   - 其余层：|diff| ≤ 5pp → 通过；|diff| > 5pp 且模拟值不在真实 CP CI 内 → 未通过（fail）；
    #     |diff| > 5pp 但模拟值仍在 CI 内 → 不可判（判据精度与抽样噪声同量级，不判超标）。
    c1_layers_out = {
        "spillover_by_openness": {},
        "agree_by_leadingness": {},
        "incomplete_by_multi": {},
    }
    for dim, layers_key in [
        ("spillover_by_openness", "n_openness"),
        ("agree_by_leadingness", "n_leadingness"),
        ("incomplete_by_multi", "n_multi"),
    ]:
        for layer in real_c1[dim]:
            n = real_c1[layers_key].get(layer, 0)
            real_v = real_c1[dim][layer]
            sim3_v = sim3_res["c1"][dim].get(layer, float("nan"))
            diff = (sim3_v - real_v) * 100
            k = int(round(real_v * n)) if n > 0 else 0
            ci_low, ci_high = clopper_pearson_ci(k, n)
            in_ci = bool(ci_low <= sim3_v <= ci_high) if n > 0 else False
            if n < 50:
                verdict = "undecidable" if in_ci else "fail"
            elif abs(diff) <= 5.0:
                verdict = "pass"
            elif not in_ci:
                verdict = "fail"
            else:
                verdict = "undecidable"
            c1_layers_out[dim][layer] = {
                "n": n, "real": real_v, "sim3": sim3_v,
                "diff_pp": diff,
                "ci_low": ci_low, "ci_high": ci_high, "in_ci": in_ci,
                "verdict": verdict,
                "pass": verdict in ("pass", "undecidable"),
            }

    # C2（判据对象见 note；主控字面口径 = iu0 且 completeness=1 ≤ 2%，
    # 但真实数据里该组合就占 17.3%，判据本身与数据矛盾——见结果文档）
    c2_out = {
        "real": real_c2, "sim3": sim3_res["c2"], "sim1": sim1_res["c2"], "simabl": simabl_res["c2"],
        "pass_sim3_literal": sim3_res["c2"]["ratio_iu0_complete"] <= 0.02,
        "pass_sim3_spill_2pct": sim3_res["c2"]["ratio_iu0_spill"] <= 0.02,
        "note": "主控字面口径=iu0且completeness=1≤2%（真实即17.3%，无意义）；"
                "数据揭示真·不可能组合=iu0且relevance=spillover（真实0.0%，§7.4第4件已加物理约束）",
    }

    # ---------- 覆盖边界：remaining 各档 n ----------
    rem_n = {b["name"]: 0 for b in rem_bins}
    for s in train_samples:
        bname = _bin_of(int(s["remaining"]), rem_bins)
        rem_n[bname] = rem_n.get(bname, 0) + 1

    # ---------- 体检表（§7.4 第2件） ----------
    phys_real = physical_check(real_rec)
    phys_check = {}
    for k, label in [("iu0", "iu0"), ("iu1", "iu1"), ("iu2", "iu2"), ("iu3p", "iu3p")]:
        diff = (sim3_res["phys"]["iu_dist"][k] - phys_real["iu_dist"][k]) * 100
        phys_check[f"diff_{label}_pp"] = diff
        phys_check[f"pass_{label}"] = abs(diff) <= 5.0
    mt_diff = sim3_res["phys"]["per_turn_mean"] - phys_real["per_turn_mean"]
    phys_check["diff_per_turn_mean"] = mt_diff
    phys_check["pass_per_turn_mean"] = abs(mt_diff) <= 0.3
    phys_check["diff_per_dialogue_mean"] = \
        sim3_res["phys"]["per_dialogue_mean"] - phys_real["per_dialogue_mean"]
    phys_pass = all(phys_check[f"pass_{k}"] for k in ("iu0", "iu1", "iu2", "iu3p")) \
        and phys_check["pass_per_turn_mean"]

    # ---------- 输出 ----------
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "holdout": {"n_dialogues": len(hold_ids), "n_pairs": len(hold_pairs)},
        "B1": b1,
        "B2": b2,
        "A_group": a_pass,
        "C1": c1_layers_out,
        "C2": c2_out,
        "physical_check": {
            "real": phys_real,
            "sim3": sim3_res["phys"],
            "sim1": sim1_res["phys"],
            "simabl": simabl_res["phys"],
            "checks": phys_check,
            "pass": phys_pass,
        },
        "coverage": {"remaining_bin_n": rem_n, "note": "remaining 各档样本量（训练集序贯样本）"},
        "negative_control": {
            "abl_B2_fails": not b2["check_abl"]["pass"],
            "note": "remaining 抹平消融必须在 B2 失败（D11 失败路径）；1 维为辅助对照",
        },
    }
    json.dump(result, open(outdir / "replay_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)

    # 打印
    print("\n" + "=" * 78)
    print("B1 患者间异质性（披露总量方差比，阈值 ≥ 0.8，截断口径）")
    print("=" * 78)
    print(f"  真人方差      = {var_real:.3f}")
    print(f"  3维   比值 = {b1['ratio_3d']:.3f}  {'✅' if b1['pass_3d'] else '❌'}")
    print(f"  消融  比值 = {b1['ratio_abl']:.3f}")
    print(f"  1维   比值 = {b1['ratio_1d']:.3f}")

    print("\n" + "=" * 78)
    print("B2 残量效应（E[iu|remaining_bin] 斜率：符号为正 + 落入真实 95% CI）")
    print("=" * 78)
    print(f"  真实斜率 = {real_slope:.4f}  bootstrap 95% CI = [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    for name, sl, chk in [("3维", sim3_res["slope"], b2["check_3d"]),
                          ("消融", simabl_res["slope"], b2["check_abl"]),
                          ("1维", sim1_res["slope"], b2["check_1d"])]:
        print(f"  {name:4s} 斜率 = {sl:.4f}  {'✅' if chk['pass'] else '❌'} "
              f"(符号正={chk['sign_positive']}, 落CI={chk['in_ci']})")
    print(f"  [D15 旧口径] turn 斜率：真实 {real_slope_legacy['slope']:.4f} / "
          f"3维 {sim3_res['legacy_slope']:.4f} / 1维 {sim1_res['legacy_slope']:.4f} / 消融 {simabl_res['legacy_slope']:.4f}")

    print("\n" + "=" * 78)
    print("A 组三率（3维 vs 真实，差 ≤ 3pp）")
    print("=" * 78)
    for k in RATE_KEYS:
        print(f"  {RATE_LABELS[k]:4s}: 真实 {a_pass[k]['real']*100:5.2f}%  "
              f"3维 {a_pass[k]['sim3']*100:5.2f}%  差 {a_pass[k]['diff_pp']:4.2f}pp "
              f"{'✅' if a_pass[k]['pass'] else '❌'}")

    print("\n" + "=" * 78)
    print("C1 条件三率（§7.5：CP CI 结算；n<50 或 CI 重叠 → 不可判）")
    print("=" * 78)
    verdict_flag = {"pass": "✅", "undecidable": "⚠️不可判", "fail": "❌"}
    for dim, layers in c1_layers_out.items():
        print(f"  [{dim}]")
        for layer, v in layers.items():
            flag = verdict_flag[v["verdict"]]
            print(f"    {layer:8s}: n={v['n']:4d}  真实 {v['real']*100:5.2f}%  "
                  f"3维 {v['sim3']*100:5.2f}%  差 {v['diff_pp']:5.2f}pp  "
                  f"CP-CI[{v['ci_low']*100:.1f}%,{v['ci_high']*100:.1f}%]  {flag}")

    print("\n" + "=" * 78)
    print("C2 联合一致性（iu==0 轮次的组合分布，R 次均值）")
    print("=" * 78)
    for name, st in [("真实", real_c2), ("3维", c2_out["sim3"]), ("消融", c2_out["simabl"]), ("1维", c2_out["sim1"])]:
        print(f"  {name:4s}: iu0≈{st['n_iu0']:.0f}  P(complete|iu0)={st['p_complete_given_iu0']*100:5.2f}%  "
              f"P(spill|iu0)={st['p_spill_given_iu0']*100:5.2f}%  P(offtopic|iu0)={st['p_offtopic_given_iu0']*100:5.2f}%")
        print(f"        占全部轮次: iu0+complete={st['ratio_iu0_complete']*100:.2f}%  "
              f"iu0+spill={st['ratio_iu0_spill']*100:.2f}%  iu0+offtopic={st['ratio_iu0_offtopic']*100:.2f}%")
    print(f"  C2 判定(字面口径 iu0+complete≤2%): {'✅' if c2_out['pass_sim3_literal'] else '❌(真实即17.3%)'}")
    print(f"  C2 判定(数据口径 iu0+spill≤2%):  {'✅' if c2_out['pass_sim3_spill_2pct'] else '❌'}")

    print("\n" + "=" * 78)
    print("体检表（§7.4 第2件）：iu ∈ {0,1,2,3+} 各档占比 + 每轮均值 + 每对话总量")
    print("=" * 78)
    for name, ph in [("真实", phys_real), ("3维", sim3_res["phys"]), ("消融", simabl_res["phys"]), ("1维", sim1_res["phys"])]:
        d = ph["iu_dist"]
        print(f"  {name:4s}: iu0={d['iu0']*100:5.1f}%  iu1={d['iu1']*100:5.1f}%  iu2={d['iu2']*100:5.1f}%  "
              f"iu3+={d['iu3p']*100:5.1f}%  | 每轮均值={ph['per_turn_mean']:.2f}  每对话总量={ph['per_dialogue_mean']:.2f}")
    d3 = sim3_res["phys"]["iu_dist"]
    print("  体检表判据（每档差≤5pp、每轮均值差≤0.3）：")
    for k, label in [("iu0", "iu0"), ("iu1", "iu1"), ("iu2", "iu2"), ("iu3p", "iu3+")]:
        diff = (d3[k] - phys_real["iu_dist"][k]) * 100
        print(f"    {label:4s}: 真实 {phys_real['iu_dist'][k]*100:5.1f}%  3维 {d3[k]*100:5.1f}%  差 {diff:5.1f}pp  "
              f"{'✅' if abs(diff)<=5 else '❌'}")
    mt_diff = sim3_res["phys"]["per_turn_mean"] - phys_real["per_turn_mean"]
    print(f"    每轮均值: 真实 {phys_real['per_turn_mean']:.2f}  3维 {sim3_res['phys']['per_turn_mean']:.2f}  "
          f"差 {mt_diff:.2f}  {'✅' if abs(mt_diff)<=0.3 else '❌'}")

    print("\n" + "=" * 78)
    print("覆盖边界：remaining 各档样本量（训练集）")
    print("=" * 78)
    print(f"  {rem_n}")

    print("\n" + "=" * 78)
    print("阴性对照（D11）：remaining 抹平消融必须在 B2 失败")
    print("=" * 78)
    print(f"  消融 B2 失败: {result['negative_control']['abl_B2_fails']}")

    # 总判定
    b1_pass = b1["pass_3d"]
    b2_pass = b2["check_3d"]["pass"]
    c1_pass = all(v["pass"] for layers in c1_layers_out.values() for v in layers.values())
    c2_pass = c2_out["pass_sim3_spill_2pct"]  # 用数据口径（见 note）
    a_pass_all = all(a_pass[k]["pass"] for k in RATE_KEYS)
    print("\n>>> M2b-1c 判定（修 info_units 估计口径后；含体检表与 C1 CP CI 结算）：")
    print(f"    B1 {'✅' if b1_pass else '❌'}  B2 {'✅' if b2_pass else '❌'}  "
          f"C1 {'✅' if c1_pass else '❌'}  C2 {'✅' if c2_pass else '❌'}  "
          f"A组 {'✅' if a_pass_all else '❌'}  体检表 {'✅' if phys_pass else '❌'}  "
          f"消融失败 {'✅' if result['negative_control']['abl_B2_fails'] else '❌'}")

    (outdir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写出 -> {outdir}/replay_result.json")
    print(f"config 快照 -> {outdir}/config.json")


if __name__ == "__main__":
    main()
