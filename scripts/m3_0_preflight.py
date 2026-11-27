"""M3-0 预检（§7.12）：判「这套尺子有没有分辨率」。

配对设计：同一病例、同一患者 seed，只改医生问句形式（开放式 vs 是-否式），
测诊断准确率差 Δ = acc_A − acc_B。

主流程：医生问句 → 判别器(qwen-plus)判三维 → 患者模拟器采样披露 → 渲染患者话
       → 12 轮后诊断模型(deepseek-chat)从候选诊断中选 → 与金标准比对。

含：操纵检查(守门)、泄露检查、两个负对照(重复性/消融)、三档判定。

模型同源约束：医生=qwen-flash、诊断=deepseek-chat、判别器=qwen-plus，三者互不同源。

用法：
  python scripts/m3_0_preflight.py --cases outputs/m3_0/cases.jsonl \
      --probs data/simulator_probs_v3_20261127.json \
      --out outputs/m3_0 --max-turns 12 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.conditional_simulator import (ConditionalSimulator, OPENNESS,
                                       LEADINGNESS, MULTI_BARRELED)
from src.disclosure_ledger import AiuLedger
from src.aiu_selector import select_aius
from src.renderer import render_turn
from src.relevance_scorer import score_all
from src.discriminator import Discriminator
from src.llm import chat

load_dotenv()

# ---------------------------------------------------------------------------
# 医生策略 system prompt（只改这一段，其余全部相同）
# ---------------------------------------------------------------------------
DOCTOR_SYSTEM = {
    "open": """You are an emergency-department physician taking a history. The patient's chief complaint: "{chief_complaint}".

You MUST strictly follow these interviewing rules:
- Use ONLY open-ended questions (e.g. "Can you describe the pain?", "What does it feel like?", "Anything else bothering you?").
- NO yes/no questions (e.g. "Do you have a fever?"), NO listing options (e.g. "Is it sharp or dull?").
- Do NOT name any disease or possible diagnosis in your question.
- Ask only ONE question at a time; build on the patient's previous answers, don't repeat what's been asked.

Output ONLY your next question (one English sentence, the question itself).""",

    "closed": """You are an emergency-department physician taking a history. The patient's chief complaint: "{chief_complaint}".

You MUST strictly follow these interviewing rules:
- Use ONLY yes/no questions (e.g. "Do you have a fever?", "Does it hurt when you move?", "Have you had surgery before?").
- NO open-ended questions (e.g. "Describe the pain", "What does it feel like?"), NO listing options.
- Do NOT name any disease or possible diagnosis in your question.
- Ask only ONE question at a time; build on the patient's previous answers, don't repeat what's been asked.

Output ONLY your next question (one English sentence, the question itself).""",
}

# 诊断模型（deepseek-chat）选择 prompt
DIAG_PROMPT = """You are a senior diagnostician. Based on the interview transcript below, choose the MOST LIKELY diagnosis from the candidate list.

Chief complaint: {chief_complaint}

Interview transcript:
{dialogue}

Candidate diagnoses:
{candidates}

Output ONLY the most likely diagnosis name (copy one from the list verbatim, no explanation, no prefix)."""


def clean_question(raw: str) -> str:
    """清理医生输出：去引号、去「医生：」前缀、截断到第一句。"""
    q = raw.strip()
    q = re.sub(r"^['\"“”]+|['\"“”]+$", "", q)
    q = re.sub(r"^(医生|Doctor)[:：]\s*", "", q)
    # 去掉可能的多余解释（取第一个问号或第一个句末）
    m = re.search(r"[^。！？.!]*[？?]", q)
    if m:
        q = m.group(0)
    q = q.strip()
    if not q:
        q = raw.strip()
    return q


def ask_doctor(chief_complaint: str, strategy: str, history: list[tuple[str, str]]) -> str:
    system = DOCTOR_SYSTEM[strategy].format(chief_complaint=chief_complaint)
    messages = [{"role": "system", "content": system}]
    for q, a in history:
        messages.append({"role": "user", "content": f"Doctor: {q}\nPatient: {a}"})
    messages.append({"role": "user", "content": "Output ONLY your next question."})
    raw = chat("doctor", messages, temperature=0.7, max_tokens=128)
    return clean_question(raw)


def diagnose(chief_complaint: str, dialogue: str, candidates: list[str],
             gold: str, client: OpenAI) -> tuple[str, bool, str | None]:
    """诊断模型从候选里选一个。返回 (预测, 是否判对, 原始输出)。"""
    cand_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt = DIAG_PROMPT.format(chief_complaint=chief_complaint,
                                dialogue=dialogue, candidates=cand_str)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 if attempt == 0 else 0.3,
                max_tokens=128)
            raw = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] diagnose: {e}")
            time.sleep(1)
            raw = ""
    pred = clean_diagnosis(raw, candidates)
    correct = (pred is not None and pred == gold)
    return pred, correct, raw


def clean_diagnosis(raw: str, candidates: list[str]) -> str | None:
    """把诊断输出映射到候选之一（精确匹配优先，其次包含匹配）。"""
    r = raw.strip().strip("。. \"'")
    # 精确匹配
    for c in candidates:
        if r == c or r == c.strip():
            return c
    # 包含匹配（输出含候选，或候选含输出）
    for c in candidates:
        if c in r or r in c:
            return c
    # 数字匹配「1. xxx」
    m = re.search(r"^\s*(\d+)[.、]", r)
    if m:
        try:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# 患者模拟器（一次对话）
# ---------------------------------------------------------------------------
def run_dialogue(case: dict, strategy: str, sim: ConditionalSimulator,
                 disc: Discriminator, seed: int, max_turns: int) -> dict:
    """跑一段 A 或 B 组对话。返回记录 + 诊断结果。"""
    facts = case["facts"]
    ledger = AiuLedger(facts)
    rng = np.random.default_rng(seed)
    render_rng = random.Random(seed)

    history: list[tuple[str, str]] = []
    turns = []
    for turn in range(max_turns):
        doctor_q = ask_doctor(case["chief_complaint"], strategy, history)
        # 判别器判三维（患者对「实际说出的问句」反应）
        prev_patient = history[-1][1] if history else None
        disc_out = disc.discriminate(doctor_q, prev=prev_patient)
        openness = disc_out["openness"]["label"]
        leadingness = disc_out["leadingness"]["label"]
        multi = disc_out["multi_barreled"]["label"]
        # 判别器偶发判不出（label None）时回退：开放组默认 open、封闭组默认 closed
        if openness is None:
            openness = "open" if strategy == "open" else "closed"
        if leadingness not in LEADINGNESS:
            leadingness = "neutral"
        if multi not in MULTI_BARRELED:
            multi = "single"

        remaining_n = ledger.remaining_count()
        behavior = sim.sample_behavior(openness, leadingness, multi, rng,
                                       remaining_n=remaining_n, turn_index=turn,
                                       dialogue_baseline=len(facts))
        rel_scores = {idx: s for idx, s in enumerate(score_all(doctor_q, facts))}
        inst = select_aius(ledger, behavior, relevance_scores=rel_scores)
        patient_utt = render_turn(doctor_q, prev_patient, inst, rng=render_rng,
                                  language="en")
        ledger.record_indices(inst["indices"])

        turns.append({
            "turn": turn, "doctor": doctor_q, "patient": patient_utt,
            "openness_disc": openness, "leadingness_disc": leadingness,
            "multi_disc": multi, "iu": behavior["info_units"],
            "special": inst.get("special"),
        })
        history.append((doctor_q, patient_utt))

    # 诊断
    dialogue_txt = "\n".join(f"Doctor: {q}\nPatient: {a}" for q, a in history)
    candidates = [case["gold_diagnosis"]] + list(case["differentials"])
    return {"turns": turns, "dialogue": dialogue_txt, "candidates": candidates,
            "history": history}


def open_vs_closed(turns: list[dict]) -> dict:
    """从判别器标签统计 open/semi/closed 占比。"""
    n = len(turns)
    c = Counter(t["openness_disc"] for t in turns)
    return {
        "n": n,
        "open_rate": (c["open"] + c["semi"]) / n if n else 0.0,
        "closed_rate": c["closed"] / n if n else 0.0,
        "open_only_rate": c["open"] / n if n else 0.0,
        "dist": dict(c),
    }


def leak_check(turns: list[dict], case: dict) -> int:
    """B 组问句泄露检查：问句含金标准/鉴别诊断名（子串匹配）的轮数。"""
    banned = [case["gold_diagnosis"]] + list(case["differentials"])
    hits = 0
    for t in turns:
        for b in banned:
            # 只匹配 >=2 字的中文诊断名（避免英文事实里的短词误报）
            if len(b) >= 2 and b in t["doctor"]:
                hits += 1
                break
    return hits


# ---------------------------------------------------------------------------
# 消融：抹平 info_units 的 openness 依赖
# ---------------------------------------------------------------------------
def ablate_openness(sim: ConditionalSimulator) -> ConditionalSimulator:
    import copy
    ab = copy.deepcopy(sim)
    iu = ab.probs["info_units"]
    meta = ab.probs["_meta"]
    rem_bins = [b["name"] for b in meta["remaining_bins"]]
    turn_bins = [b["name"] for b in meta["turn_bins"]]
    # 每个 (rbin, tbin) 下三个 openness 档平均成同一分布
    rt_marg = defaultdict(lambda: defaultdict(float))
    for (o, rb, tb), dist in iu.items():
        for y, p in dist.items():
            rt_marg[(rb, tb)][y] += p / len(OPENNESS)
    new_iu = {}
    for o in OPENNESS:
        for rb in rem_bins:
            for tb in turn_bins:
                d = rt_marg[(rb, tb)]
                s = sum(d.values())
                new_iu[(o, rb, tb)] = {y: p / s for y, p in d.items()} if s > 0 else d
    ab.probs["info_units"] = new_iu
    ab.probs["_meta"]["ablated_openness"] = True
    return ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="outputs/m3_0/cases.jsonl")
    ap.add_argument("--probs", default="data/simulator_probs_v3_20261127.json")
    ap.add_argument("--out", default="outputs/m3_0")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个病例（pilot）")
    ap.add_argument("--skip-neg", action="store_true", help="跳过负对照（快速主结果）")
    ap.add_argument("--resume", action="store_true", help="从 results.jsonl 断点续跑（跳过已完成病例）")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    if args.limit > 0:
        cases = cases[:args.limit]
    sim = ConditionalSimulator.load(Path(args.probs))
    disc = Discriminator(cache_path=Path(args.out) / "disc_cache.jsonl")
    diag_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                         base_url="https://api.deepseek.com/v1")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # 断点续跑：每跑完一个病例就 append 一行到 results.jsonl，进程被杀后可 --resume 续跑
    results_file = outdir / "results.jsonl"
    results = []
    done_ids = set()
    if args.resume and results_file.exists():
        for line in open(results_file, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done_ids.add(r["case_id"])
            results.append(r)
        print(f"[resume] 已完成 {len(done_ids)} 例，跳过续跑")
    else:
        results_file.write_text("", encoding="utf-8")

    print(f"病例 {len(cases)} 个，max_turns={args.max_turns}，seed={args.seed}")

    # ---------- 主跑批：配对 A/B ----------
    t0 = time.time()
    for i, case in enumerate(cases):
        cid = case["case_id"]
        if cid in done_ids:
            continue
        # 同一 seed 保证患者行为除措辞外一致
        a = run_dialogue(case, "open", sim, disc, args.seed, args.max_turns)
        b = run_dialogue(case, "closed", sim, disc, args.seed, args.max_turns)

        pred_a, corr_a, raw_a = diagnose(case["chief_complaint"], a["dialogue"],
                                         a["candidates"], case["gold_diagnosis"],
                                         diag_client)
        pred_b, corr_b, raw_b = diagnose(case["chief_complaint"], b["dialogue"],
                                         b["candidates"], case["gold_diagnosis"],
                                         diag_client)
        rec = {
            "case_id": cid, "difficulty": case["difficulty"],
            "gold": case["gold_diagnosis"],
            "A": {"correct": corr_a, "pred": pred_a, "openness": open_vs_closed(a["turns"]),
                  "n_turns": len(a["turns"]), "leak_hits": leak_check(a["turns"], case)},
            "B": {"correct": corr_b, "pred": pred_b, "openness": open_vs_closed(b["turns"]),
                  "n_turns": len(b["turns"]), "leak_hits": leak_check(b["turns"], case)},
            "_turns_A": a["turns"], "_turns_B": b["turns"],
            "_raw_diag_A": raw_a, "_raw_diag_B": raw_b,
        }
        results.append(rec)
        # 断点落盘（每病例一行，进程被杀后可 --resume 续跑）
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if len(results) % 5 == 0:
            el = time.time() - t0
            acc_a = sum(r["A"]["correct"] for r in results) / len(results)
            acc_b = sum(r["B"]["correct"] for r in results) / len(results)
            print(f"  进度 {len(results)}/{len(cases)} acc_A={acc_a:.2f} acc_B={acc_b:.2f} "
                  f"Δ={acc_a-acc_b:+.3f} {el:.0f}s")

    # ---------- 汇总 ----------
    acc_a = np.mean([r["A"]["correct"] for r in results])
    acc_b = np.mean([r["B"]["correct"] for r in results])
    delta = acc_a - acc_b

    # McNemar（配对）
    discord_a = sum(1 for r in results if r["A"]["correct"] and not r["B"]["correct"])
    discord_b = sum(1 for r in results if not r["A"]["correct"] and r["B"]["correct"])
    n_disc = discord_a + discord_b
    from math import comb
    mcnemar_p = 0.0
    if n_disc > 0:
        for k in range(n_disc + 1):
            prob = comb(n_disc, k) * (0.5 ** n_disc)
            if abs(k - n_disc / 2) >= abs(discord_a - n_disc / 2):
                mcnemar_p += prob

    # bootstrap CI（病例级重采样）
    rng = np.random.default_rng(args.seed)
    B = 2000
    n = len(results)
    acc_arr = np.array([r["A"]["correct"] for r in results], dtype=float)
    acc_brr = np.array([r["B"]["correct"] for r in results], dtype=float)
    diffs = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(float(acc_arr[idx].mean() - acc_brr[idx].mean()))
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    # 分层
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)
    layered = {}
    for d in ["easy", "medium", "hard"]:
        sub = by_diff.get(d, [])
        if sub:
            layered[d] = {
                "n": len(sub),
                "acc_A": float(np.mean([r["A"]["correct"] for r in sub])),
                "acc_B": float(np.mean([r["B"]["correct"] for r in sub])),
            }

    # 操纵检查汇总
    manip = {
        "A_open_rate": float(np.mean([r["A"]["openness"]["open_rate"] for r in results])),
        "B_closed_rate": float(np.mean([r["B"]["openness"]["closed_rate"] for r in results])),
        "open_gap_pp": float(np.mean([r["A"]["openness"]["open_only_rate"]
                                      - r["B"]["openness"]["open_only_rate"]
                                      for r in results]) * 100),
    }
    # 泄露汇总（B 组为主）
    leak_total = sum(r["B"]["leak_hits"] for r in results)
    leak_turns = sum(r["B"]["n_turns"] for r in results)

    main_result = {
        "seed": args.seed, "n_cases": n, "max_turns": args.max_turns,
        "acc_A": float(acc_a), "acc_B": float(acc_b), "delta": float(delta),
        "delta_ci_95": [ci_lo, ci_hi],
        "mcnemar": {"discord_A_only": discord_a, "discord_B_only": discord_b,
                    "p": mcnemar_p},
        "layered": layered,
        "manipulation": manip,
        "leak": {"B_hits": leak_total, "B_turns": leak_turns,
                 "B_rate": leak_total / leak_turns if leak_turns else 0.0},
    }

    # ---------- 负对照 1：重复性（A 组换 seed 重跑） ----------
    repeat = None
    if not args.skip_neg:
        print("负对照 1：重复性（A 组换 seed）...")
        results_r = []
        for case in cases:
            a2 = run_dialogue(case, "open", sim, disc, args.seed + 999, args.max_turns)
            pred, corr, _ = diagnose(case["chief_complaint"], a2["dialogue"],
                                     a2["candidates"], case["gold_diagnosis"], diag_client)
            results_r.append({"correct": corr})
        acc_a2 = np.mean([r["correct"] for r in results_r])
        # Δ_AA：A(seed) vs A(seed+999)，配对
        repeat = {"acc_A_seed2": float(acc_a2), "delta_AA": float(acc_a - acc_a2),
                  "n": len(results_r)}

    # ---------- 负对照 2：消融（抹平 openness 依赖） ----------
    ablation = None
    if not args.skip_neg:
        print("负对照 2：消融（抹平 openness 依赖）...")
        sim_abl = ablate_openness(sim)
        disc_abl = Discriminator(cache_path=Path(args.out) / "disc_cache_abl.jsonl")
        res_abl = []
        for case in cases:
            a3 = run_dialogue(case, "open", sim_abl, disc_abl, args.seed, args.max_turns)
            b3 = run_dialogue(case, "closed", sim_abl, disc_abl, args.seed, args.max_turns)
            _, ca, _ = diagnose(case["chief_complaint"], a3["dialogue"], a3["candidates"],
                                case["gold_diagnosis"], diag_client)
            _, cb, _ = diagnose(case["chief_complaint"], b3["dialogue"], b3["candidates"],
                                case["gold_diagnosis"], diag_client)
            res_abl.append({"A": ca, "B": cb})
        acc_a3 = np.mean([r["A"] for r in res_abl])
        acc_b3 = np.mean([r["B"] for r in res_abl])
        ablation = {"acc_A": float(acc_a3), "acc_B": float(acc_b3),
                    "delta_ablated": float(acc_a3 - acc_b3), "n": len(res_abl)}

    # ---------- 三档判定 ----------
    def verdict(ci_lo, ci_hi, delta, acc_a, acc_b):
        if ci_hi <= 0.03 and ci_lo >= -0.03:
            return "无分辨率（CI ⊂ [−3,+3]pp）", "停 M3 正题，回查模拟器对 openness 敏感度与病例难度"
        if ci_lo > 0 and delta > 0.05:
            return "有分辨率（下界>0 且点估计>5pp）", "放行 M3 正题"
        return "功效不足（CI 跨 0 或过宽）", "报所需样本量，扩样本或改配对"

    v, followup = verdict(ci_lo, ci_hi, delta, acc_a, acc_b)
    # 天花板/地板警报
    alarm = None
    if acc_a > 0.9 or acc_b > 0.9:
        alarm = f"天花板警报：A={acc_a:.2f}/B={acc_b:.2f} 有一组>90%，Δ不可解释，须先调病例难度"
    elif acc_a < 0.3 and acc_b < 0.3:
        alarm = f"地板警报：两组均<30%（A={acc_a:.2f}/B={acc_b:.2f}），Δ不可解释"

    main_result["verdict"] = {"conclusion": v, "followup": followup, "alarm": alarm}
    main_result["negative_controls"] = {"repeat": repeat, "ablation": ablation}

    # 落盘（剥离大的 turns 原始记录，另存）
    turns_file = outdir / "turns.jsonl"
    with open(turns_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"case_id": r["case_id"], "difficulty": r["difficulty"],
                                "gold": r["gold"],
                                "turns_A": r["_turns_A"], "turns_B": r["_turns_B"],
                                "pred_A": r["A"]["pred"], "pred_B": r["B"]["pred"],
                                "correct_A": r["A"]["correct"], "correct_B": r["B"]["correct"]},
                               ensure_ascii=False) + "\n")
    for r in results:
        r.pop("_turns_A", None); r.pop("_turns_B", None)
        r.pop("_raw_diag_A", None); r.pop("_raw_diag_B", None)
    json.dump({"cases": results, "summary": main_result},
              open(outdir / "m3_0_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---------- 控制台汇总 ----------
    print("\n" + "=" * 78)
    print("M3-0 预检结果")
    print("=" * 78)
    print(f"病例 {n}，每组 {args.max_turns} 轮")
    print(f"acc_A(开放) = {acc_a:.3f}   acc_B(封闭) = {acc_b:.3f}")
    print(f"Δ = acc_A − acc_B = {delta:+.3f}   95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"McNemar: A对B错 {discord_a}  A错B对 {discord_b}  p={mcnemar_p:.4g}")
    layered_str = " ".join(f"{k}:A{v['acc_A']:.2f}/B{v['acc_B']:.2f}" for k, v in layered.items())
    print(f"分层: {layered_str}")
    print(f"操纵检查: A开放率 {manip['A_open_rate']:.1%}  B封闭率 {manip['B_closed_rate']:.1%}  "
          f"open差 {manip['open_gap_pp']:.1f}pp")
    print(f"泄露: B组命中 {leak_total}/{leak_turns} = {leak_total/leak_turns:.1%}")
    if repeat:
        print(f"负对照1 重复性: acc_A(seed2)={repeat['acc_A_seed2']:.3f}  "
              f"Δ_AA={repeat['delta_AA']:+.3f}")
    if ablation:
        print(f"负对照2 消融: acc_A={ablation['acc_A']:.3f} acc_B={ablation['acc_B']:.3f}  "
              f"Δ_ablated={ablation['delta_ablated']:+.3f}")
    print(f"三档判定: {v}")
    print(f"后续: {followup}")
    if alarm:
        print(f"⚠️ {alarm}")
    print(f"\n完成 -> {outdir}/m3_0_result.json")


if __name__ == "__main__":
    main()
