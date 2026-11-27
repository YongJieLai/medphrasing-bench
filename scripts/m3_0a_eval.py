"""M3-0a 判别器验收：评 100 条评测集，报四项数字。

  ① 四维逐条一致率（pred vs 仲裁后真值）≥ 0.90
  ② 四维混淆矩阵（gold × pred）
  ③ 四维边缘分布对照（pred vs M0 各维差 ≤ 5pp，防漂移）
  ④ 置信度-准确率校准曲线（≥ 5 分箱）

评测设置（对应 §7.10「核心判据无上下文下成立」）：
  - openness / leadingness / multi_barreled：无上下文判别（prev=None）
  - follow_up：附上一患者发言判别（其定义依赖 prev）

产出 outputs/m3_0a/eval_result.json + 打印关键数字。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.discriminator import Discriminator, DIMS

CTX_DIMS = {"follow_up"}  # 需上下文的维度
NOC_DIMS = ["openness", "leadingness", "multi_barreled"]  # 无上下文维度


def confusion(gold, pred, vals):
    # 返回 dict[gold_val][pred_val] = count
    m = {v: Counter() for v in vals}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return {k: dict(v) for k, v in m.items()}


def marginal_dist(labels, vals):
    c = Counter(labels)
    n = sum(c.values()) or 1
    return {v: c.get(v, 0) / n for v in vals}


def calibration(golds, preds, confs, bins=5):
    """置信度分箱 → 准确率。返回 (bin_edges, bin_centers, accs, counts)。"""
    pairs = [(c, (g == p)) for g, p, c in zip(golds, preds, confs)]
    if not pairs:
        return None
    cs = np.array([c for c, _ in pairs])
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (cs >= lo) & (cs <= hi + 1e-9)
        else:
            mask = (cs >= lo) & (cs < hi)
        accs = [a for (c, a), m in zip(pairs, mask) if m]
        if accs:
            out.append({"bin": f"[{lo:.1f},{hi:.1f})", "center": (lo + hi) / 2,
                        "n": len(accs), "accuracy": sum(accs) / len(accs)})
        else:
            out.append({"bin": f"[{lo:.1f},{hi:.1f})", "center": (lo + hi) / 2,
                        "n": 0, "accuracy": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set", help="outputs/m3_0a/eval_set.jsonl（含 M0）")
    ap.add_argument("gold", help="outputs/m3_0a/gold_labels.jsonl")
    ap.add_argument("--cache", default="outputs/m3_0a/disc_cache.jsonl")
    ap.add_argument("--out", default="outputs/m3_0a/eval_result.json")
    args = ap.parse_args()

    ev = { (r["dialogue_id"], r["turn_idx"]): r
           for r in (json.loads(l) for l in open(args.eval_set, encoding="utf-8")) }
    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]

    disc = Discriminator(cache_path=args.cache)

    result = {"records": [], "per_dim": {}, "latency_ms": []}

    for g in gold:
        k = (g["dialogue_id"], g["turn_idx"])
        m0 = ev[k]["m0"]
        doctor = g["doctor_utt"]
        prev = g["prev_patient_utt"]

        # 无上下文判别三维
        out_noc = disc.discriminate(doctor, prev=None)
        # 附上下文判别 follow_up
        out_ctx = disc.discriminate(doctor, prev=prev)

        pred = {d: out_noc[d]["label"] for d in NOC_DIMS}
        pred["follow_up"] = out_ctx["follow_up"]["label"]
        conf = {d: out_noc[d]["confidence"] for d in NOC_DIMS}
        conf["follow_up"] = out_ctx["follow_up"]["confidence"]

        rec = {"dialogue_id": k[0], "turn_idx": k[1],
               "doctor_utt": doctor, "prev": prev,
               "gold": g["label"], "pred": pred, "conf": conf, "m0": m0}
        result["records"].append(rec)
        for d in DIMS:
            if out_noc.get(d, {}).get("latency_ms") or out_ctx.get(d, {}).get("latency_ms"):
                pass
        result["latency_ms"].append(out_noc["openness"].get("latency_ms", 0))

    # 逐维统计
    for d in DIMS:
        golds = [r["gold"][d] for r in result["records"]]
        preds = [r["pred"][d] for r in result["records"]]
        confs = [r["conf"][d] for r in result["records"]]
        m0s = [r["m0"][d] for r in result["records"]]

        # 排除 gold 或 pred 为 None 的样本
        valid_idx = [i for i in range(len(golds))
                     if golds[i] is not None and preds[i] is not None]
        gv = [golds[i] for i in valid_idx]
        pv = [preds[i] for i in valid_idx]
        cv = [confs[i] for i in valid_idx]
        mv = [m0s[i] for i in valid_idx]

        agree = sum(1 for a, b in zip(gv, pv) if a == b)
        n = len(gv)

        vals = sorted(set(gv) | set(pv))
        result["per_dim"][d] = {
            "n": n,
            "agreement": agree,
            "agreement_rate": agree / n if n else None,
            "confusion_matrix": confusion(gv, pv, vals),
            "pred_marginal": marginal_dist(pv, vals),
            "m0_marginal": marginal_dist(mv, vals),
            "gold_marginal": marginal_dist(gv, vals),
            "calibration": calibration(gv, pv, cv),
            "degraded": (d == "follow_up"),
        }
        # 边缘分布差 vs M0（判据 2）
        pm = result["per_dim"][d]["pred_marginal"]
        mm = result["per_dim"][d]["m0_marginal"]
        maxdiff = max(abs(pm.get(v, 0) - mm.get(v, 0)) for v in vals)
        result["per_dim"][d]["marginal_max_diff_vs_m0_pp"] = round(maxdiff * 100, 2)

    lat = np.array(result["latency_ms"])
    result["latency"] = {
        "n": int(len(lat)), "mean_ms": round(float(lat.mean()), 1),
        "median_ms": round(float(np.median(lat)), 1),
        "p90_ms": round(float(np.percentile(lat, 90)), 1),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 打印关键数字
    print("=== 四维一致率（pred vs 仲裁后真值） ===")
    for d in DIMS:
        v = result["per_dim"][d]
        flag = "  [降级:follow_up]" if v["degraded"] else ""
        print(f"  {d}: {v['agreement']}/{v['n']} = {v['agreement_rate']:.1%}"
              f"  边缘分布vsM0最大差 {v['marginal_max_diff_vs_m0_pp']}pp{flag}")
    print("=== 延迟 ===")
    print(f"  mean {result['latency']['mean_ms']}ms  median "
          f"{result['latency']['median_ms']}ms  p90 {result['latency']['p90_ms']}ms")


if __name__ == "__main__":
    main()
