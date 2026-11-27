"""M2e-0 量化真实患者的是非题应答分布（§7.9，只统计，不做实现）。

覆盖 §7.9 三件事：
  1. 圈定「是非问句」：openness=closed 且 yes/no 型（LLM 判 is_yesno，排除 wh- 真问句与 A-or-B 选择题）
  2. 对每轮患者回答分五类 a/b/c/d/e + 对话级 bootstrap 95% CI + turn_bin 分层
  3. 三态交叉：真值集槽状态（有肯定/有否定/无该槽）× 五类，重点两个条件概率

用法:
  python scripts/m2e0_quantify.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --out outputs/YYYYMMDD_m2e0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat
from scripts.calibrate_simulator import valid_rows

# ---- 圈定 + 五类（一次调用） ----
CLASSIFY_PROMPT = """You are analyzing a doctor-patient dialogue for a research study.

Doctor's question:
"{doctor}"

Patient's answer:
"{patient}"

Task 1 — Is the doctor's question a YES/NO question?
A yes/no question is one the patient can naturally answer with "yes", "no", or express not knowing.
- YES: auxiliary/be-verb questions ("Do you smoke?"), "Any ...?" questions ("Any fever?"), tag questions ("You're 41, correct?").
- NO: wh-questions asking for a specific fact ("How old are you?", "When did it start?", "Which hand?"), and A-or-B choice questions ("Are you right handed or left handed?").

Task 2 — Classify the patient's answer into ONE of these five classes:
- "a": affirms — says yes, or directly states the fact asked about.
- "b": denies — says no, or denies the fact asked about (e.g. "No.", "Neither of them.").
- "c": unknown — expresses not knowing / not sure / not remembering / not noticing (e.g. "I don't know.", "I'm not sure.", "I don't remember.", "Not recently." for a "have you checked X" question).
- "d": off-answer — does not answer the question, instead says something else (e.g. asked about fever, answers about back pain).
- "e": unclassifiable — empty, filler, or pure acknowledgement ("Okay.", "Mm-hmm.", "Right.").

Output STRICT JSON only:
{{"is_yesno": 1, "answer_class": "a"}}"""

# ---- 三态判断（一次调用） ----
SLOT_PROMPT = """You are analyzing a doctor-patient dialogue for a research study.

Doctor's question:
"{doctor}"

The patient's KNOWN facts (things the patient has mentioned earlier). fidelity: definite=stated as true, fuzzy=stated vaguely, denied=the patient EXPLICITLY DENIED it (e.g. "no fever", "no smoking", "no allergies").

Here are the facts most related to the question (ranked by lexical relevance):
{fact_list}

The doctor is asking about a specific slot (e.g. fever, smoking, pain, allergy, medication).
If the question mentions several slots (joined by "or"/"and"), judge the FIRST/most central slot.

Determine the slot's status among the patient's known facts:
- "affirmed": the facts contain a POSITIVE statement about this slot (fidelity definite or fuzzy), e.g. patient said they HAVE a fever / they smoke.
- "denied": the facts contain a NEGATIVE statement about this slot (fidelity denied), e.g. patient said "no fever" / "no smoking".
- "absent": the facts contain NOTHING about this slot — neither positive nor negative.

CRITICAL: a fact like "no fever" (fidelity denied) means the slot is DENIED, NOT absent. Only when no fact mentions the slot at all should you output "absent".

Output STRICT JSON only:
{{"slot_status": "affirmed"}}"""

FID_LABEL = {"definite": "definite", "fuzzy": "fuzzy", "denied": "denied"}


def split_holdout(valid, truthset, seed, frac=0.2):
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


def parse_json(out: str) -> dict:
    out = (out or "").strip()
    # 去掉可能的 markdown 代码块
    out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip())
    try:
        return json.loads(out)
    except Exception:
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def classify_one(doctor, patient) -> dict:
    prompt = CLASSIFY_PROMPT.format(doctor=doctor, patient=patient)
    try:
        out = chat("annotator", [{"role": "user", "content": prompt}], temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return {"is_yesno": -1, "answer_class": "e", "error": str(e)}
    d = parse_json(out)
    yn = d.get("is_yesno", -1)
    ac = str(d.get("answer_class", "e")).strip().lower()
    if ac not in ("a", "b", "c", "d", "e"):
        ac = "e"
    return {"is_yesno": int(yn) if yn in (0, 1) else -1, "answer_class": ac}


def slot_one(doctor, truth_set) -> dict:
    # 相关性筛 top-8 事实，让 LLM 聚焦；同时把相关性分数一并给出
    from src.relevance_scorer import score_all
    sc = score_all(doctor, truth_set)
    order = sorted(range(len(sc)), key=lambda i: -sc[i])[:8]
    lines = []
    for j, i in enumerate(order, 1):
        a = truth_set[i]
        fid = FID_LABEL.get(a.get("fidelity", "definite"), "definite")
        lines.append(f"{j}. [{fid}] {a['text']} (relevance {sc[i]:.2f})")
    fact_list = "\n".join(lines) if lines else "（无已知事实）"
    prompt = SLOT_PROMPT.format(doctor=doctor, fact_list=fact_list)
    try:
        out = chat("annotator", [{"role": "user", "content": prompt}], temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return {"slot_status": "error", "error": str(e)}
    d = parse_json(out)
    ss = str(d.get("slot_status", "")).strip().lower()
    if ss not in ("affirmed", "denied", "absent"):
        ss = "error"
    return {"slot_status": ss}


def bootstrap_proportions(records: list[dict], rng, B: int = 2000, classes=("a", "b", "c", "d", "e")):
    """对话级 bootstrap：重采样对话，重算各类比例，取 2.5/97.5 百分位。"""
    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_d[r["dialogue_id"]].append(r)
    dids = list(by_d.keys())
    n = len(records)
    out = {}
    for cls in classes:
        boot = []
        for _ in range(B):
            samp = [r for d in rng.choice(dids, size=len(dids), replace=True)
                    for r in by_d[d]]
            cnt = sum(1 for r in samp if r["answer_class"] == cls)
            boot.append(cnt / len(samp) if samp else 0.0)
        point = sum(1 for r in records if r["answer_class"] == cls) / n if n else float("nan")
        out[cls] = {"point": point,
                    "ci_low": float(np.percentile(boot, 2.5)),
                    "ci_high": float(np.percentile(boot, 97.5))}
    return out


def turn_bin_of(turn: int) -> str:
    if turn == 0:
        return "t0"
    if turn == 1:
        return "t1"
    if turn == 2:
        return "t2"
    return "t3p"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}
    by_d, hold_ids = split_holdout(valid, truthset, args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "m2e0_quantify.json"

    # 断点续跑
    if out_path.exists():
        done = json.load(open(out_path, encoding="utf-8"))
        recs = done["records"]
        done_keys = {(r["dialogue_id"], r["turn_idx"]) for r in recs}
    else:
        recs = []
        done_keys = set()

    # 候选：holdout 所有 closed 问句
    closed_rows = [(d, r) for d in sorted(hold_ids) for r in by_d[d]
                   if r["openness"] == "closed"]
    if args.limit:
        closed_rows = closed_rows[: args.limit]

    print(f"holdout 对话 {len(hold_ids)}；closed 问句候选 {len(closed_rows)} 条")

    t0 = time.time()
    for i, (d, r) in enumerate(closed_rows):
        key = (d, r["turn_idx"])
        if key in done_keys:
            continue
        cls = classify_one(r["doctor_utt"], r.get("patient_utt") or "")
        rec = {
            "dialogue_id": d, "turn_idx": r["turn_idx"],
            "doctor_utt": r["doctor_utt"], "patient_utt": r.get("patient_utt", ""),
            "is_yesno": cls["is_yesno"], "answer_class": cls["answer_class"],
        }
        if cls.get("error"):
            rec["error"] = cls["error"]
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(closed_rows)}  耗时 {time.time()-t0:.0f}s")

    # ---- 三态判断（只对 is_yesno==1 的） ----
    yesno_recs = [r for r in recs if r["is_yesno"] == 1]
    slot_done = {"_done": False}
    for r in yesno_recs:
        if "slot_status" in r:
            continue
        ts = truthset[r["dialogue_id"]]["truth_set"]
        ss = slot_one(r["doctor_utt"], ts)
        r["slot_status"] = ss["slot_status"]
        if ss.get("error"):
            r["slot_error"] = ss["error"]

    # ---- 统计 ----
    n_closed = len(recs)
    n_yesno = len(yesno_recs)
    n_wh = sum(1 for r in recs if r["is_yesno"] == 0)
    n_err = sum(1 for r in recs if r["is_yesno"] == -1)

    # 五类比例（只统计 yes/no 圈定内的）
    yesno_valid = [r for r in yesno_recs if r["answer_class"] in ("a", "b", "c", "d", "e")]
    prop = bootstrap_proportions(yesno_valid, np.random.default_rng(args.seed),
                                 args.bootstrap)

    # turn_bin 分层
    by_turn = {}
    for tb in ("t0", "t1", "t2", "t3p"):
        sub = [r for r in yesno_valid if turn_bin_of(r["turn_idx"]) == tb]
        if len(sub) >= 50:
            by_turn[tb] = {"n": len(sub),
                           "prop": bootstrap_proportions(
                               sub, np.random.default_rng(args.seed), args.bootstrap)}
        else:
            by_turn[tb] = {"n": len(sub), "note": "n<50，判据精度低于抽样噪声，不参与判定"}

    # 3×5 交叉表
    cross = defaultdict(lambda: defaultdict(int))
    for r in yesno_valid:
        ss = r.get("slot_status", "error")
        cross[ss][r["answer_class"]] += 1

    # 两个条件概率
    def cond_prob(slot, ans_cls):
        denom = sum(cross[slot].values())
        if denom == 0:
            return float("nan"), 0
        return cross[slot][ans_cls] / denom, denom

    p_b_given_denied, n_denied = cond_prob("denied", "b")
    p_c_given_absent, n_absent = cond_prob("absent", "c")

    # 30 条人工样例（每类 ≥5）
    samples = []
    for cls in ("a", "b", "c", "d", "e"):
        pool = [r for r in yesno_valid if r["answer_class"] == cls]
        rng_s = np.random.default_rng(args.seed + 7)
        pick = [pool[i] for i in rng_s.choice(len(pool), size=min(6, len(pool)), replace=False)]
        samples.extend(pick)

    result = {
        "seed": args.seed,
        "n_holdout_dialogues": len(hold_ids),
        "n_closed_candidates": n_closed,
        "n_yesno": n_yesno,
        "n_non_yesno": n_wh,
        "n_parse_error": n_err,
        "proportions": prop,
        "by_turn_bin": by_turn,
        "cross_table": {k: dict(v) for k, v in cross.items()},
        "conditional_probs": {
            "p_b_given_slot_denied": {"value": p_b_given_denied, "n": n_denied},
            "p_c_given_slot_absent": {"value": p_c_given_absent, "n": n_absent},
        },
        "samples_30": samples,
        "records": recs,
    }
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 控制台汇总 ----
    print("\n" + "=" * 78)
    print("M2e-0 是非题应答分布")
    print("=" * 78)
    print(f"closed 候选 {n_closed}；圈定 yes/no {n_yesno}；排除 {n_wh}；解析失败 {n_err}")
    print("\n五类比例（yes/no 圈定内，含 95% CI）:")
    for cls, name in [("a", "肯定"), ("b", "否定"), ("c", "明确不知道"),
                      ("d", "答了别的"), ("e", "无法判")]:
        p = prop[cls]
        print(f"  {cls} {name:<6}: {p['point']:.1%}  CI [{p['ci_low']:.1%}, {p['ci_high']:.1%}]")
    print(f"\n三态交叉表 (slot × answer):")
    print(f"  {'slot':<10} {'a':>6} {'b':>6} {'c':>6} {'d':>6} {'e':>6} {'sum':>6}")
    for ss in ("affirmed", "denied", "absent", "error"):
        row = cross[ss]
        print(f"  {ss:<10} {row['a']:>6} {row['b']:>6} {row['c']:>6} "
              f"{row['d']:>6} {row['e']:>6} {sum(row.values()):>6}")
    print(f"\n关键条件概率:")
    print(f"  P(答否定 | 清单有否定槽) = {p_b_given_denied:.1%} (n={n_denied})")
    print(f"  P(答不知道 | 清单无该槽) = {p_c_given_absent:.1%} (n={n_absent})")
    c_point = prop["c"]["point"]
    verdict = ("≥10% → 立项 M2e-1" if c_point >= 0.10
               else ("5-10% → 停，回主控" if c_point >= 0.05
                     else "<5% → 不立项，轻量方案"))
    print(f"\nc 类比例 {c_point:.1%} → 判定: {verdict}")
    print(f"\n完成 -> {out_path}")


if __name__ == "__main__":
    main()
