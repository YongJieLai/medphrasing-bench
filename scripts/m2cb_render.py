"""M2c-b 主脚本：内容相关性选择 + vague 规则修正（§7.7，M2d 的前置）。

覆盖 §7.7 的第 1/2/3/4 件：
  1. 相关性打分（src/relevance_scorer.py，缓存到文件）
  2. 按 relevance 标签选内容（src/aiu_selector.py，on-topic 最高分 / spillover 最低分）
  3. vague 规则修正（src/renderer.py：denied 事实不再含糊）
  4. 重跑报四项：
     a. 集合级 F1（≥ 0.90）
     b. vague 轮 F1（≥ 0.88，修前 0.792 并列）
     c. 构念溢出率：spillover − on-topic 的「说未问事实」轮占比差（≥ 15pp）
        —— 同时跑「顺序选」作阴性对照（D11：顺序选应 < 15pp，否则测量是恒等式）
     d. 真值集耗尽轮占比（修前顺序选 vs 修后相关性选）

用法:
  python scripts/m2cb_render.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --probs data/simulator_probs_v3_20261127.json \
      --out outputs/YYYYMMDD_m2cb --limit 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import ConditionalSimulator
from src.disclosure_ledger import AiuLedger
from src.aiu_selector import select_aius
from src.renderer import render_turn
from src.relevance_scorer import build_cache, load_cache, score_all, is_asked
from scripts.calibrate_simulator import valid_rows
from scripts.m2c_render import align_facts, f1_of, lag1_autocorr

THRESHOLD = 0.2  # 「医生问到了」的门槛，透明上报；见 §7.7 与结果文档


# ---------------------------------------------------------------------------
# 选择层（无 LLM）：跑一次回放，返回每轮的选择结果，用于构念溢出率 / 耗尽率
# ---------------------------------------------------------------------------

def replay_selection(dial_ids, by_d, truthset, sim, rng, mode: str,
                     cache: dict | None) -> list[dict]:
    """mode: 'sequential'（修前）或 'relevance'（修后）。返回每轮选择记录。

    两模式下「说未问事实」都用选中事实的真实相关性分数判定（不因模式不同而
    改变测量口径）：sequential 只是选择器不认相关性，测量仍按同一把尺子。
    """
    records = []
    for did in dial_ids:
        ts = truthset[did]["truth_set"]
        ledger = AiuLedger(ts)
        scores_by_turn = (cache or {}).get(str(did), {}) if cache else {}
        for i, row in enumerate(by_d[did]):
            behavior = sim.sample_behavior(
                row["openness"], row["leadingness"], row["multi_barreled"], rng,
                remaining_n=ledger.remaining_count(), turn_index=i,
                dialogue_baseline=len(ts))
            rem = ledger.remaining_aius()
            n_turns = len(by_d[did])

            if not rem:
                records.append({
                    "dialogue_id": did, "turn": i, "exhausted": True,
                    "is_last": i == n_turns - 1, "behavior": behavior,
                    "selected_scores": [], "said_unasked": False,
                })
                continue

            # 该轮全部事实的真实相关性分数（两种模式共用同一把尺）
            turn_scores = score_all(row["doctor_utt"], ts)
            relev_scores = None
            if mode == "relevance":
                relev_scores = {idx: s for idx, s in enumerate(turn_scores)}
            instruction = select_aius(ledger, behavior, relevance_scores=relev_scores)

            must_idx = instruction["indices"]
            sel_scores = [turn_scores[idx] for idx in must_idx]
            records.append({
                "dialogue_id": did, "turn": i, "exhausted": False,
                "is_last": i == n_turns - 1, "behavior": behavior,
                "selected_scores": sel_scores,
                "said_unasked": any(not is_asked(s, THRESHOLD) for s in sel_scores),
            })
            ledger.record_indices(must_idx)
    return records


def spillover_separation(records: list[dict]) -> dict:
    """构念溢出率：spillover/off-topic 组 vs on-topic 组的「说未问事实」比例。

    排除两类轮：耗尽轮（无事实可披露）与特殊轮（r0/agree/no_content，无事实清单）。
    同时报轮级（任一说未问事实即算）与事实级（按说出的事实数）两个口径。
    """
    def _normal(r):
        return (not r.get("exhausted", False)) and (not r.get("special"))
    on = [r for r in records if r["behavior"]["relevance"] == "on-topic" and _normal(r)]
    sp = [r for r in records if r["behavior"]["relevance"] in ("spillover", "off-topic")
          and _normal(r)]

    on_turn = sum(r["said_unasked"] for r in on) / len(on) if on else float("nan")
    sp_turn = sum(r["said_unasked"] for r in sp) / len(sp) if sp else float("nan")

    def fact_rate(recs):
        n = sum(len(r["selected_scores"]) for r in recs)
        u = sum(sum(1 for s in r["selected_scores"] if not is_asked(s, THRESHOLD))
                for r in recs)
        return u / n if n else float("nan")
    on_fact = fact_rate(on)
    sp_fact = fact_rate(sp)

    return {
        "on_topic_n": len(on), "on_topic_unasked_rate": on_turn,
        "spillover_n": len(sp), "spillover_unasked_rate": sp_turn,
        "diff_pp": (sp_turn - on_turn) * 100,
        "fact_level": {
            "on_topic_unasked_rate": on_fact,
            "spillover_unasked_rate": sp_fact,
            "diff_pp": (sp_fact - on_fact) * 100,
        },
    }


def exhaustion_rate(records: list[dict]) -> dict:
    """真值集耗尽轮占比（remaining=0 的轮次 / 总轮次）。"""
    total = len(records)
    exhausted = [r for r in records if r["exhausted"]]
    non_last = [r for r in exhausted if not r["is_last"]]
    return {
        "total_turns": total,
        "exhausted": len(exhausted),
        "exhausted_rate": len(exhausted) / total if total else float("nan"),
        "exhausted_non_last": len(non_last),
    }


# ---------------------------------------------------------------------------
# LLM 渲染层：相关性选择 + vague 修正后的完整回放
# ---------------------------------------------------------------------------

def render_dialogue_cb(dialogue_id, rows, truth, sim, rng, scores_by_turn) -> list[dict]:
    truth_set = truth[dialogue_id]["truth_set"]
    ledger = AiuLedger(truth_set)
    prev_patient = None
    records = []
    for i, row in enumerate(rows):
        behavior = sim.sample_behavior(
            row["openness"], row["leadingness"], row["multi_barreled"], rng,
            remaining_n=ledger.remaining_count(), turn_index=i,
            dialogue_baseline=len(truth_set))
        turn_scores = scores_by_turn.get(str(i))
        relev_scores = {idx: s for idx, s in enumerate(turn_scores)} if turn_scores else None
        instruction = select_aius(ledger, behavior, relevance_scores=relev_scores)
        rendered = render_turn(row["doctor_utt"], prev_patient, instruction,
                               rng=random.Random(0))
        ledger.record_indices(instruction["indices"])

        if instruction.get("special"):
            align = {"matched": [], "missed": [], "extra": [], "said_omit": []}
            gt = []
            omit = []
        else:
            gt = instruction["must_say"]
            omit = instruction["omit"]
            align = align_facts(gt, omit, rendered)

        sel_scores = [relev_scores[idx] for idx in instruction["indices"]] \
            if relev_scores and instruction["indices"] else []
        records.append({
            "dialogue_id": dialogue_id, "turn": i,
            "doctor_utt": row["doctor_utt"],
            "behavior": behavior,
            "gt_n": len(gt), "omit_n": len(omit),
            "special": instruction.get("special"),
            "selected_scores": sel_scores,
            "said_unasked": any(not is_asked(s, THRESHOLD) for s in sel_scores),
            "rendered": rendered,
            "align": align,
            "pred_n": len(align["matched"]) + len(align["extra"]),
            "label_iu": instruction["info_units"],
        })
        prev_patient = rendered
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}
    sim = ConditionalSimulator.load(Path(args.probs))

    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_d[r["dialogue_id"]].append(r)
    for d in by_d:
        by_d[d].sort(key=lambda x: x["turn_idx"])

    dial_ids = sorted(d for d in by_d if d in truthset)
    if args.limit:
        dial_ids = dial_ids[: args.limit]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "relevance_cache.json"

    print(f"抽样对话 {len(dial_ids)} 个")
    # 第 1 件：相关性打分 + 缓存
    cache = build_cache({d: by_d[d] for d in dial_ids}, truthset, cache_path)
    scores = cache["scores"]

    # ---------- 选择层：构念溢出率 + 耗尽率（修前顺序 vs 修后相关性） ----------
    print("选择层：顺序选(修前) vs 相关性选(修后) ...")
    sel_sep = {}
    for mode in ("sequential", "relevance"):
        rng = np.random.default_rng(args.seed)
        recs = replay_selection(dial_ids, by_d, truthset, sim, rng, mode,
                                cache=scores if mode == "relevance" else None)
        sel_sep[mode] = {
            "spillover_separation": spillover_separation(recs),
            "exhaustion": exhaustion_rate(recs),
        }

    # ---------- LLM 渲染（修后：相关性选 + vague 修正） ----------
    print("LLM 渲染 100 对话（相关性选 + vague 修正）...")
    all_records = []
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for di, did in enumerate(dial_ids):
        recs = render_dialogue_cb(did, by_d[did], truthset, sim, rng,
                                  scores.get(str(did), {}))
        all_records.extend(recs)
        if (di + 1) % 10 == 0:
            print(f"  进度 {di+1}/{len(dial_ids)}  耗时 {time.time()-t0:.0f}s")

    # ---------- 集合级 F1 ----------
    gt_total = sum(r["gt_n"] for r in all_records)
    matched_total = sum(len(r["align"]["matched"]) for r in all_records)
    extra_total = sum(len(r["align"]["extra"]) for r in all_records)
    said_omit_total = sum(len(r["align"]["said_omit"]) for r in all_records)
    missed_total = sum(len(r["align"]["missed"]) for r in all_records)
    denom = gt_total + matched_total + extra_total + said_omit_total
    macro_f1 = (2 * matched_total / denom) if denom > 0 else float("nan")
    per_turn_f1 = [f1_of(r["gt_n"], r["align"]) for r in all_records]
    mean_f1 = float(np.mean(per_turn_f1)) if per_turn_f1 else float("nan")

    n_normal = [r for r in all_records if not r["special"]]
    # 渲染失真率（§7.2.6 记录一：不再叫「溢出率」）
    distort = [r for r in n_normal if len(r["align"]["extra"]) > 0
               or len(r["align"]["said_omit"]) > 0]
    render_distortion_rate = len(distort) / len(n_normal) if n_normal else float("nan")

    # ---------- vague 轮 F1（第 3 件：预期 ≥ 0.88） ----------
    vague_recs = [r for r in n_normal if r["behavior"]["definiteness"] == "vague"]
    vague_f1s = [f1_of(r["gt_n"], r["align"]) for r in vague_recs]
    vague_f1 = float(np.mean(vague_f1s)) if vague_f1s else float("nan")
    vague_distort = [r for r in vague_recs
                     if len(r["align"]["extra"]) > 0 or len(r["align"]["said_omit"]) > 0]
    vague_distort_rate = len(vague_distort) / len(vague_recs) if vague_recs else float("nan")
    exact_recs = [r for r in n_normal if r["behavior"]["definiteness"] == "exact"]
    exact_f1 = float(np.mean([f1_of(r["gt_n"], r["align"]) for r in exact_recs])) \
        if exact_recs else float("nan")

    # ---------- 构念溢出率（渲染后的实际指令级） ----------
    render_sep = spillover_separation([
        {**r, "behavior": r["behavior"]} for r in all_records])

    # ---------- B3 轮间自相关 ----------
    by_dial_records = defaultdict(list)
    for r in all_records:
        by_dial_records[r["dialogue_id"]].append(r)
    label_lag, pred_lag = [], []
    for did, recs in by_dial_records.items():
        ll = lag1_autocorr([r["label_iu"] for r in recs])
        pl = lag1_autocorr([r["pred_n"] for r in recs])
        if not np.isnan(ll):
            label_lag.append(ll)
        if not np.isnan(pl):
            pred_lag.append(pl)
    label_lag_mean = float(np.mean(label_lag)) if label_lag else float("nan")
    pred_lag_mean = float(np.mean(pred_lag)) if pred_lag else float("nan")

    # ---------- 20 条人工可核对样例（问句 / 事实 / 分数） ----------
    samples = []
    seen = set()
    for did in dial_ids:
        ts = truthset[did]["truth_set"]
        for r in by_d[did]:
            q = r["doctor_utt"]
            for idx, a in enumerate(ts):
                key = (q, a["text"])
                if key in seen:
                    continue
                seen.add(key)
                s = score_all(q, [a])[0]
                samples.append({"question": q, "fact": a["text"],
                                "fidelity": a["fidelity"], "score": round(s, 3)})
                if len(samples) >= 60:
                    break
            if len(samples) >= 60:
                break
        if len(samples) >= 60:
            break
    # 混排，保证高/中/低分都有（人工可核对 20 条从这 60 里抽）
    samples_sorted = sorted(samples, key=lambda x: x["score"], reverse=True)
    verify_20 = samples_sorted[:10] + samples_sorted[-10:]

    # ---------- 一段 15 轮完整对话（全程无提前耗尽） ----------
    demo_recs, demo_did = pick_demo(truthset, by_d, sim, rng)

    # ---------- 输出 ----------
    result = {
        "n_dialogues": len(dial_ids),
        "n_turns": len(all_records),
        "threshold": THRESHOLD,
        "selection_layer": sel_sep,
        "recall_consistency": {
            "gt_total": gt_total, "matched_total": matched_total,
            "extra_total": extra_total, "said_omit_total": said_omit_total,
            "missed_total": missed_total,
            "macro_f1": macro_f1, "mean_per_turn_f1": mean_f1,
            "render_distortion_rate": render_distortion_rate,
        },
        "vague": {
            "n_vague": len(vague_recs), "vague_f1": vague_f1,
            "vague_distort_rate": vague_distort_rate,
            "n_exact": len(exact_recs), "exact_f1": exact_f1,
            "before_fix_f1": 0.792,  # M2c 修前（docs/m2c_results.md）
        },
        "construct_spillover_rendered": render_sep,
        "B3_autocorr": {"label_iu_lag1": label_lag_mean,
                        "rendered_iu_lag1": pred_lag_mean,
                        "n_dialogues_with_lag": len(pred_lag)},
        "verify_samples_20": verify_20,
        "demo_dialogue": {"dialogue_id": demo_did, "records": demo_recs},
        "all_records": all_records,
    }
    with open(outdir / "m2cb_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------- 控制台汇总 ----------
    print("\n" + "=" * 78)
    print("M2c-b 四项数字")
    print("=" * 78)
    print(f"(a) 集合级 F1 = {macro_f1:.4f} (每轮均值 {mean_f1:.4f})  阈值 0.90  → "
          f"{'✅' if macro_f1 >= 0.90 else '❌'}")
    print(f"(b) vague 轮 F1 = {vague_f1:.4f} (n={len(vague_recs)})  修前 0.792  → "
          f"{'✅' if vague_f1 >= 0.88 else '❌'}")
    print(f"    exact 轮 F1 = {exact_f1:.4f} (n={len(exact_recs)})")
    sep = sel_sep["relevance"]["spillover_separation"]
    sep_seq = sel_sep["sequential"]["spillover_separation"]
    print(f"(c) 构念溢出率(轮级): 修后 spillover {sep['spillover_unasked_rate']:.1%} "
          f"vs on-topic {sep['on_topic_unasked_rate']:.1%}  差 {sep['diff_pp']:.1f}pp  "
          f"(顺序选对照差 {sep_seq['diff_pp']:.1f}pp)")
    print(f"    事实级: 修后 spillover {sep['fact_level']['spillover_unasked_rate']:.1%} "
          f"vs on-topic {sep['fact_level']['on_topic_unasked_rate']:.1%}  "
          f"差 {sep['fact_level']['diff_pp']:.1f}pp  "
          f"(顺序选对照差 {sep_seq['fact_level']['diff_pp']:.1f}pp)")
    print(f"    判据 轮级差值≥15pp → {'✅' if sep['diff_pp'] >= 15 else '❌'}")
    ex_seq = sel_sep["sequential"]["exhaustion"]
    ex_rel = sel_sep["relevance"]["exhaustion"]
    print(f"(d) 真值集耗尽轮占比: 顺序选 {ex_seq['exhausted_rate']:.1%}  "
          f"vs 相关性选 {ex_rel['exhausted_rate']:.1%}")
    print(f"\nB3: 标签 lag-1 = {label_lag_mean:.4f}  渲染后 lag-1 = {pred_lag_mean:.4f}")
    print(f"渲染失真率 = {render_distortion_rate:.2%}")
    print(f"demo 对话 id={demo_did} 轮数={len(demo_recs)}  全程无提前耗尽"
          f" → {'✅' if all(r['special'] != 'r0' for r in demo_recs) else '❌'}")
    print(f"\n完成 -> {outdir}/m2cb_result.json")


def _scores_for(did, rows, truthset) -> dict[str, list[float]]:
    """为单个对话的每一轮计算 {位置索引(str): [score per fact]}（不依赖预建缓存）。

    键用对话内位置（enumerate 索引），与回放里 turn_index=i 对齐。
    """
    facts = truthset[did]["truth_set"]
    out = {}
    for i, r in enumerate(rows):
        out[str(i)] = score_all(r["doctor_utt"], facts)
    return out


def pick_demo(truthset, by_d, sim, rng, seed=42, target_turns=15):
    """挑一段 ~15 轮、全程无「提前说完了」的对话（真值集足够大）。

    先在选择层（无 LLM）确认候选对话 15 轮内不耗尽，再渲染该对话。
    """
    cands = sorted((d for d in by_d if d in truthset and len(by_d[d]) >= target_turns),
                   key=lambda d: -truthset[d]["truth_set_size"])

    # 选择层筛选：找 15 轮内 remaining 恒 > 0 的对话
    chosen = None
    for did in cands:
        rng0 = np.random.default_rng(seed)
        ts = truthset[did]["truth_set"]
        ledger = AiuLedger(ts)
        ok = True
        rows = by_d[did][:target_turns]
        for i, row in enumerate(rows):
            b = sim.sample_behavior(
                row["openness"], row["leadingness"], row["multi_barreled"], rng0,
                remaining_n=ledger.remaining_count(), turn_index=i,
                dialogue_baseline=len(ts))
            if not ledger.remaining_aius():
                ok = False
                break
            inst = select_aius(ledger, b,
                               relevance_scores={idx: s for idx, s in enumerate(
                                   _scores_for(did, rows, truthset)[str(i)])})
            ledger.record_indices(inst["indices"])
        if ok:
            chosen = did
            break

    if chosen is None:
        chosen = cands[0] if cands else next(iter(by_d))

    demo_rng = np.random.default_rng(seed)
    scores_by_turn = _scores_for(chosen, by_d[chosen][:target_turns], truthset)
    recs = render_dialogue_cb(chosen, by_d[chosen][:target_turns], truthset, sim,
                              demo_rng, scores_by_turn)
    return recs, chosen


if __name__ == "__main__":
    main()
