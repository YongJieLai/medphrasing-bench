"""M2d 反事实验收主脚本（§7.8 第 3/4 件）：改一处措辞，看患者披露量变不变。

三层跑批（同一对话、同一患者、同一随机种子、只替换 openness 标签）：
  - baseline      原句（openness 原始）
  - open->closed  所有 open 轮 → closed
  - closed->open  所有 closed 轮 → open

主指标：info_units 每轮均值（分 open/semi/closed 层报，验证 open>semi>closed 单调性
        且改写后方向正确）。
次要指标：标签级溢出率（relevance 分布，只做方向检查，不作判据）。
真实对照：holdout 真实 open vs closed 的 iu 均值差 + bootstrap 95% CI + 真实耗尽率。

用法:
  python scripts/m2d_run.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --probs data/simulator_probs_v3_20261127.json \
      --rewrites outputs/20261127_m2d/rewrites.json \
      --out outputs/20261127_m2d --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import ConditionalSimulator
from src.disclosure_ledger import AiuLedger
from src.aiu_selector import select_aius
from src.renderer import render_turn
from src.relevance_scorer import score_all
from scripts.calibrate_simulator import valid_rows

OPENNESS_ORDER = ["closed", "semi", "open"]


def split_holdout(valid: list[dict], truthset: dict, seed: int, frac: float = 0.2):
    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_d[r["dialogue_id"]].append(r)
    for d in by_d:
        by_d[d].sort(key=lambda x: x["turn_idx"])
    dial_ids = sorted(d for d in by_d if d in truthset)
    rng = np.random.default_rng(seed)
    rng.shuffle(dial_ids)
    n_hold = int(len(dial_ids) * frac)
    hold_ids = set(dial_ids[:n_hold])
    return by_d, hold_ids


def replay_iu(dial_ids, by_d, truthset, sim, rng,
              openness_override: dict | None = None) -> list[dict]:
    """标签层回放：按序贯 remaining 推进，返回每轮 iu / relevance。

    openness_override: {(dialogue_id, turn_idx): new_openness}
    只改 openness，leadingness / multi_barreled 保持原始。
    """
    records = []
    for did in dial_ids:
        tsize = truthset[did]["truth_set_size"]
        remaining = tsize
        for i, row in enumerate(by_d[did]):
            openness = row["openness"]
            if openness_override and (did, row["turn_idx"]) in openness_override:
                openness = openness_override[(did, row["turn_idx"])]
            behavior = sim.sample_behavior(
                openness, row["leadingness"], row["multi_barreled"], rng,
                remaining_n=remaining, turn_index=i, dialogue_baseline=tsize)
            iu = int(behavior["info_units"])
            records.append({
                "dialogue_id": did, "turn_idx": row["turn_idx"], "turn": i,
                "openness_orig": row["openness"], "openness_used": openness,
                "iu": iu, "relevance": behavior["relevance"],
            })
            remaining = max(0, remaining - iu)
    return records


def iu_by_layer(records: list[dict]) -> dict:
    """按『原始 openness 层』聚合 iu（原句层 vs 改写层都按原始分组，可比）。"""
    out = {}
    for o in OPENNESS_ORDER:
        arr = [r["iu"] for r in records if r["openness_orig"] == o]
        out[o] = {
            "n": len(arr),
            "mean": float(np.mean(arr)) if arr else float("nan"),
            "std": float(np.std(arr)) if arr else float("nan"),
        }
    overall = [r["iu"] for r in records]
    out["_overall"] = {
        "n": len(overall),
        "mean": float(np.mean(overall)) if overall else float("nan"),
    }
    return out


def spillover_by_layer(records: list[dict]) -> dict:
    """标签级溢出率：relevance 分布（spillover / off-topic 轮占比）。"""
    from collections import Counter
    c = Counter(r["relevance"] for r in records)
    n = len(records)
    return {
        "n": n,
        "spillover_rate": c["spillover"] / n if n else float("nan"),
        "off_topic_rate": c["off-topic"] / n if n else float("nan"),
        "on_topic_rate": c["on-topic"] / n if n else float("nan"),
    }


def real_iu_by_layer(by_d, hold_ids) -> dict:
    """真实 holdout 的 iu 均值（按原始 openness 分组，口径与模拟一致）。"""
    out = {}
    for o in OPENNESS_ORDER:
        arr = [int(r.get("info_units") or 0)
               for d in hold_ids for r in by_d[d] if r["openness"] == o]
        out[o] = {"n": len(arr), "mean": float(np.mean(arr)) if arr else float("nan")}
    return out


def real_exhaustion(by_d, hold_ids, truthset) -> dict:
    """真实耗尽率：remaining 归零后仍有后续轮（与 replay_iu 同口径）。"""
    total = 0
    exhausted = 0
    for d in hold_ids:
        tsize = truthset[d]["truth_set_size"]
        remaining = tsize
        for r in by_d[d]:
            if remaining == 0:
                exhausted += 1
            total += 1
            iu = int(r.get("info_units") or 0)
            remaining = max(0, remaining - iu)
    return {"total_turns": total, "exhausted": exhausted,
            "exhausted_rate": exhausted / total if total else float("nan")}


def bootstrap_diff_open_closed(by_d, hold_ids, rng, B: int = 2000) -> dict:
    """真实 open vs closed 的 iu 均值差的对话级 bootstrap 95% CI。"""
    open_by_d: dict[int, list[int]] = defaultdict(list)
    closed_by_d: dict[int, list[int]] = defaultdict(list)
    for d in hold_ids:
        for r in by_d[d]:
            iu = int(r.get("info_units") or 0)
            if r["openness"] == "open":
                open_by_d[d].append(iu)
            elif r["openness"] == "closed":
                closed_by_d[d].append(iu)
    o_dids = list(open_by_d.keys())
    c_dids = list(closed_by_d.keys())
    point = float(np.mean([iu for d in o_dids for iu in open_by_d[d]])
                  - np.mean([iu for d in c_dids for iu in closed_by_d[d]]))
    diffs = []
    for _ in range(B):
        o = [iu for d in rng.choice(o_dids, size=len(o_dids), replace=True)
             for iu in open_by_d[d]]
        c = [iu for d in rng.choice(c_dids, size=len(c_dids), replace=True)
             for iu in closed_by_d[d]]
        diffs.append(float(np.mean(o)) - float(np.mean(c)))
    return {
        "point": point,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "B": B,
    }


def render_side_by_side(did, rows, truth, sim, rng_seed, rewrites_by_turn,
                        direction="open->closed"):
    """渲染一段对话的『原句版』与『改写版』，用于并排对比。

    direction: 该对话要对比的改写方向（只对 open 轮做 open->closed）。
    两个版本共用同一个行为序列（标签层），只改问句文本与 openness 标签。
    """
    truth_set = truth[did]["truth_set"]

    def _rel_scores(doctor_utt):
        return {idx: s for idx, s in enumerate(score_all(doctor_utt, truth_set))}

    # 原句版渲染（相关性选，与 M2c-b 口径一致）
    render_rng0 = random.Random(0)
    sample_rng0 = np.random.default_rng(rng_seed)
    ledger0 = AiuLedger(truth_set)
    prev = None
    orig_lines = []
    for i, row in enumerate(rows):
        b = sim.sample_behavior(row["openness"], row["leadingness"], row["multi_barreled"],
                                sample_rng0, remaining_n=ledger0.remaining_count(),
                                turn_index=i, dialogue_baseline=len(truth_set))
        inst0 = select_aius(ledger0, b, relevance_scores=_rel_scores(row["doctor_utt"]))
        out0 = render_turn(row["doctor_utt"], prev, inst0, rng=render_rng0)
        orig_lines.append({"doctor": row["doctor_utt"], "patient": out0, "iu": b["info_units"],
                           "openness": row["openness"], "rewritten_turn": False})
        ledger0.record_indices(inst0["indices"])
        prev = out0

    # 改写版渲染：open 轮用改写问句 + openness=closed；其余轮不变
    render_rng1 = random.Random(0)
    sample_rng1 = np.random.default_rng(rng_seed)
    ledger1 = AiuLedger(truth_set)
    prev = None
    new_lines = []
    for i, row in enumerate(rows):
        openness = row["openness"]
        doctor_utt = row["doctor_utt"]
        rewritten_turn = False
        rw = rewrites_by_turn.get(row["turn_idx"])
        if rw is not None:
            openness = "closed"
            doctor_utt = rw
            rewritten_turn = True
        b = sim.sample_behavior(openness, row["leadingness"], row["multi_barreled"],
                                sample_rng1, remaining_n=ledger1.remaining_count(),
                                turn_index=i, dialogue_baseline=len(truth_set))
        inst1 = select_aius(ledger1, b, relevance_scores=_rel_scores(doctor_utt))
        out1 = render_turn(doctor_utt, prev, inst1, rng=render_rng1)
        new_lines.append({"doctor": doctor_utt, "patient": out1, "iu": b["info_units"],
                          "openness": openness, "rewritten_turn": rewritten_turn})
        ledger1.record_indices(inst1["indices"])
        prev = out1

    return {
        "dialogue_id": did,
        "original": orig_lines,
        "rewritten": new_lines,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--rewrites", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}
    sim = ConditionalSimulator.load(Path(args.probs))
    by_d, hold_ids = split_holdout(valid, truthset, args.seed)
    hold_dids = sorted(hold_ids)

    rewrites = json.load(open(args.rewrites, encoding="utf-8"))
    pairs = rewrites["pairs"]

    # 构建 override 映射：
    #   open->closed：覆盖所有 open 轮（改写脚本已全量改写 open 轮，映射一致）
    #   closed->open：覆盖所有 closed 轮（跑批标签全量改，不依赖改写文本；
    #                 改写文本只用于校验与并排对比）
    o2c = {(p["dialogue_id"], p["turn_idx"]): "closed"
           for p in pairs if p["direction"] == "open->closed"}
    c2o = {}
    for d in hold_ids:
        for r in by_d[d]:
            if r["openness"] == "closed":
                c2o[(d, r["turn_idx"])] = "open"

    print(f"holdout 对话 {len(hold_dids)}；open->closed 覆盖 {len(o2c)} 轮，"
          f"closed->open 覆盖 {len(c2o)} 轮")

    # ---------- 三层跑批 ----------
    print("三层跑批（标签层）...")
    layers = {}
    for name, override in [("baseline", None), ("open_to_closed", o2c), ("closed_to_open", c2o)]:
        rng = np.random.default_rng(args.seed)
        recs = replay_iu(hold_dids, by_d, truthset, sim, rng, override)
        layers[name] = {
            "iu_by_layer": iu_by_layer(recs),
            "spillover": spillover_by_layer(recs),
        }

    # ---------- 真实对照 ----------
    print("真实对照 + bootstrap CI ...")
    real_iu = real_iu_by_layer(by_d, hold_ids)
    real_diff = bootstrap_diff_open_closed(by_d, hold_ids, np.random.default_rng(args.seed),
                                           args.bootstrap)
    real_exh = real_exhaustion(by_d, hold_ids, truthset)

    # ---------- 并排对比（一段 demo 对话） ----------
    # 选一段含 open 轮、真值集较大的对话
    rewrites_by_did = defaultdict(dict)
    for p in pairs:
        if p["direction"] == "open->closed":
            rewrites_by_did[p["dialogue_id"]][p["turn_idx"]] = p["rewritten"]
    cands = sorted((d for d in hold_dids if d in rewrites_by_did),
                   key=lambda d: (-len(rewrites_by_did[d]), -truthset[d]["truth_set_size"]))
    demo_did = cands[0]
    demo = render_side_by_side(demo_did, by_d[demo_did], truthset, sim, args.seed,
                               rewrites_by_did[demo_did], direction="open->closed")

    # ---------- 关键数字 ----------
    b = layers["baseline"]["iu_by_layer"]
    o2c_l = layers["open_to_closed"]["iu_by_layer"]
    c2o_l = layers["closed_to_open"]["iu_by_layer"]

    # 模拟降幅：open 层（原句）均值 − open 层（改 closed）均值
    sim_drop = b["open"]["mean"] - o2c_l["open"]["mean"]
    # 模拟升幅：closed 层（改 open）均值 − closed 层（原句）均值
    sim_rise = c2o_l["closed"]["mean"] - b["closed"]["mean"]
    real_gap = real_diff["point"]

    result = {
        "seed": args.seed,
        "n_holdout_dialogues": len(hold_dids),
        "layers": layers,
        "real_iu_by_layer": real_iu,
        "real_diff_open_closed": real_diff,
        "real_exhaustion": real_exh,
        "sim_effect": {
            "open_to_closed_drop": sim_drop,
            "closed_to_open_rise": sim_rise,
            "real_gap": real_gap,
            "drop_over_real": sim_drop / real_gap if real_gap else float("nan"),
            "rise_over_real": sim_rise / real_gap if real_gap else float("nan"),
        },
        "side_by_side_demo": demo,
        "rewrites_check": rewrites.get("check"),
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(outdir / "m2d_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---------- 控制台汇总 ----------
    print("\n" + "=" * 78)
    print("M2d 反事实验收")
    print("=" * 78)
    print(f"真实对照: open {real_iu['open']['mean']:.3f} (n={real_iu['open']['n']})  "
          f"semi {real_iu['semi']['mean']:.3f} (n={real_iu['semi']['n']})  "
          f"closed {real_iu['closed']['mean']:.3f} (n={real_iu['closed']['n']})")
    print(f"真实 open-closed 差 = {real_gap:.3f}  95% CI [{real_diff['ci_low']:.3f}, "
          f"{real_diff['ci_high']:.3f}]")
    print(f"真实耗尽率 = {real_exh['exhausted_rate']:.1%}")
    print()
    print("三层 iu 均值（分原始 openness 层）:")
    print(f"  {'层':<16} {'open':>8} {'semi':>8} {'closed':>8} {'overall':>8}")
    for name, label in [("baseline", "原句对照"), ("open_to_closed", "open->closed"),
                        ("closed_to_open", "closed->open")]:
        d = layers[name]["iu_by_layer"]
        print(f"  {label:<16} {d['open']['mean']:>8.3f} {d['semi']['mean']:>8.3f} "
              f"{d['closed']['mean']:>8.3f} {d['_overall']['mean']:>8.3f}")
    print()
    print(f"模拟降幅(open 层改 closed) = {sim_drop:.3f}   "
          f"模拟升幅(closed 层改 open) = {sim_rise:.3f}")
    print(f"降幅/真实差 = {sim_drop/real_gap:.2%}   "
          f"升幅/真实差 = {sim_rise/real_gap:.2%}")
    print()
    mono_ok = (b["open"]["mean"] > b["semi"]["mean"] > b["closed"]["mean"])
    print(f"单调性(原句 open>semi>closed) = {'✅' if mono_ok else '❌'}")
    dir_ok = (sim_drop > 0) and (sim_rise > 0)
    print(f"方向(改 closed 降、改 open 升) = {'✅' if dir_ok else '❌'}")
    print(f"\n完成 -> {outdir}/m2d_result.json")


if __name__ == "__main__":
    main()
