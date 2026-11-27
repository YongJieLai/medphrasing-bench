"""§7.11 M0 三维审计：随机抽 200 条 → deepseek 照抄手册复标 → 四个数 + 污染率三档判定。

本单只做审计，不改标注、不重建表。审计与修复分开（避免「修完再测」自我实现）。

用法：
  python3 scripts/m0_audit.py sample                 # 随机抽 200 条
  python3 scripts/m0_audit.py relabel                # deepseek 复标三维
  python3 scripts/m0_audit.py report                 # 四个数 + 判定
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from scripts.calibrate_simulator import (LEADINGNESS, MULTI_BARRELED,
                                         OPENNESS)

load_dotenv()

SEED = 20261127
N_SAMPLE = 200
OUTDIR = Path("outputs/m0_audit")
SAMPLE_PATH = OUTDIR / "sample.jsonl"
SAMPLE_META = OUTDIR / "sample_meta.json"
RELABEL_PATH = OUTDIR / "relabel.jsonl"
RESULT_PATH = OUTDIR / "result.json"

DIMS = ["openness", "leadingness", "multi_barreled"]

# ---------------------------------------------------------------------------
# 手册原文：照抄 annotate_mtsdialog.py 的医生侧三维定义（含例子），不加解释。
# 本单测的是「M0 有没有按自己的手册执行」，不是「谁的定义更好」。
# ---------------------------------------------------------------------------
RUBRIC = """
- openness: classify by how much the question CONSTRAINS the answer.
  * "closed": answerable with yes/no or a choice. Starts with an auxiliary verb (Do/Does/Did/Is/Are/Have/Has/Can/Could/Would/Should) OR "Any..." OR offers "A or B?". Examples: "Do you have a fever?", "Any chest pain or shortness of breath?", "Did it start suddenly or gradually?"
  * "semi": a wh-question expecting ONE concrete fact (a number, name, time, or list). Examples: "What medications are you taking?", "How many packs a day?", "When did it start?", "How old are you?"
  * "open": invites a free narrative/description with no single expected answer. Examples: "Tell me about your pain.", "Describe what happened.", "What brings you in today?", "How are you feeling?"
- leadingness: "neutral" (no answer implied, e.g. "What does the pain feel like?") | "mild" (suggests a possible answer but the patient can freely deny, e.g. "Is the pain sharp?") | "strong" (pushes the patient to confirm, e.g. tag questions "...correct?" / "...right?" / "...isn't it?" or "Your pain is sharp, yes?").
- multi_barreled: "single" (asks for ONE piece of info) | "multi" (asks for >=2 distinct pieces in one utterance, e.g. "Do you have fever, cough, or chest pain?").
"""

RELABEL_PROMPT = """You are labeling a DOCTOR'S QUESTION from a doctor-patient dialogue. Below is the EXACT annotation rubric (copy-pasted verbatim from the original annotation manual). Follow the rubric literally; do NOT add your own interpretation.

DOCTOR'S UTTERANCE:
"{doctor}"

RUBRIC (verbatim):
{rubric}

Label the doctor's utterance on the three dimensions above. Output STRICT JSON only (no extra text):
{{"openness":"..","leadingness":"..","multi_barreled":".."}}"""

VALID = {
    "openness": set(OPENNESS),
    "leadingness": set(LEADINGNESS),
    "multi_barreled": set(MULTI_BARRELED),
}


def parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not set(DIMS).issubset(d.keys()):
        return None
    for k in DIMS:
        v = str(d.get(k, "")).strip()
        if v not in VALID[k]:
            return None
        d[k] = v
    return d


def load_pool() -> list[dict]:
    rows = [json.loads(l) for l in open("data/annot_all_flash.jsonl", encoding="utf-8")]
    pool = [r for r in rows
            if not r.get("label_error")
            and r.get("is_question") == 1
            and r.get("openness") in OPENNESS
            and r.get("leadingness") in LEADINGNESS
            and r.get("multi_barreled") in MULTI_BARRELED]
    return pool


def used_keys() -> set:
    ev = [json.loads(l) for l in open("outputs/m3_0a/eval_set.jsonl", encoding="utf-8")]
    return {(r["dialogue_id"], r["turn_idx"]) for r in ev}


def cmd_sample() -> None:
    pool = load_pool()
    used = used_keys()
    pool_unused = [r for r in pool if (r["dialogue_id"], r["turn_idx"]) not in used]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pool_unused))[:N_SAMPLE]
    sample = [pool_unused[i] for i in idx]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    def marginal(rows):
        out = {}
        for d in DIMS:
            out[d] = dict(Counter(r[d] for r in rows))
        return out

    meta = {
        "seed": SEED,
        "population_n": len(pool),
        "pool_after_exclude_eval_n": len(pool_unused),
        "excluded_eval_n": len(used),
        "sample_n": len(sample),
        "population_marginal": marginal(pool),
        "sample_marginal": marginal(sample),
        "representativeness_note": (
            "全库 multi 率 %.1f%% vs 样本 multi 率见 sample_marginal；"
            "随机 200 条的多重抽样波动用卡方/二项近似评估见 report。"
            % (marginal(pool)["multi_barreled"]["multi"] / len(pool) * 100)),
    }

    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps({
                "dialogue_id": r["dialogue_id"],
                "turn_idx": r["turn_idx"],
                "doctor_utt": r["doctor_utt"],
                "m0": {d: r[d] for d in DIMS},
            }, ensure_ascii=False) + "\n")
    json.dump(meta, open(SAMPLE_META, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"抽样完成：全库 {len(pool)} → 排除评测集 {len(used)} → 池 {len(pool_unused)}"
          f" → 抽 {len(sample)} 条，seed={SEED}")
    print("样本 M0 三维分布：")
    for d in DIMS:
        print(f"  {d}: {marginal(sample)[d]}")
    print("全库 M0 三维分布：")
    for d in DIMS:
        print(f"  {d}: {marginal(pool)[d]}")


def cmd_relabel() -> None:
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1")
    sample = [json.loads(l) for l in open(SAMPLE_PATH, encoding="utf-8")]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = 0
    with open(RELABEL_PATH, "w", encoding="utf-8") as f:
        for i, r in enumerate(sample):
            prompt = RELABEL_PROMPT.format(doctor=r["doctor_utt"], rubric=RUBRIC)
            lab = None
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0 if attempt == 0 else 0.3,
                        max_tokens=256)
                    lab = parse_json(resp.choices[0].message.content or "")
                    if lab:
                        break
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] {e}")
                time.sleep(1)
            row = {"dialogue_id": r["dialogue_id"], "turn_idx": r["turn_idx"],
                   "doctor_utt": r["doctor_utt"], "label": lab}
            if lab:
                n_ok += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  进度 {i+1}/{len(sample)} ok={n_ok} {el:.0f}s "
                      f"({el/(i+1)*1000:.0f}ms/条)")
    print(f"复标完成: {n_ok}/{len(sample)} 耗时 {time.time()-t0:.0f}s -> {RELABEL_PATH}")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """二项比例的 Clopper-Pearson 精确 CI。k=成功数, n=总数。"""
    from scipy.stats import beta as _beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else _beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def bootstrap_ci_proportion(k: int, n: int, rng, B: int = 5000) -> tuple[float, float]:
    """对话级 bootstrap 二项比例 CI（每对话单位重采样，这里简化为对 200 条重采样）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    draws = rng.binomial(n, k / n, B) / n
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def cmd_report() -> None:
    sample = [json.loads(l) for l in open(SAMPLE_PATH, encoding="utf-8")]
    relabel = [json.loads(l) for l in open(RELABEL_PATH, encoding="utf-8")]
    rel_by_key = {(r["dialogue_id"], r["turn_idx"]): r["label"] for r in relabel}

    per_dim = {}
    for d in DIMS:
        agree = 0
        tot = 0
        full_conf = Counter()   # (m0, ds) 全量，含一致
        for r in sample:
            m0 = r["m0"][d]
            ds = (rel_by_key.get((r["dialogue_id"], r["turn_idx"])) or {}).get(d)
            if m0 is None or ds is None:
                continue
            tot += 1
            agree += (m0 == ds)
            full_conf[(m0, ds)] += 1
        lo, hi = clopper_pearson(agree, tot)
        # 完整混淆矩阵：{m0: {ds: count}}
        matrix = {}
        for (m0, ds), n in full_conf.items():
            matrix.setdefault(m0, {})[ds] = n
        # 分歧（非对角线）
        disagreement = {f"{m0}->{ds}": n for (m0, ds), n in full_conf.items() if m0 != ds}
        per_dim[d] = {
            "n": tot,
            "agree": agree,
            "agreement_rate": agree / tot if tot else None,
            "ci_95": [lo, hi],
            "confusion": disagreement,
            "confusion_matrix": matrix,
        }

    # --- 污染率（核心数）：M0=single 里 deepseek 判 multi 的比例 ---
    single_total = 0
    single_is_multi = 0
    for r in sample:
        m0 = r["m0"]["multi_barreled"]
        ds = (rel_by_key.get((r["dialogue_id"], r["turn_idx"])) or {}).get("multi_barreled")
        if m0 == "single" and ds is not None:
            single_total += 1
            if ds == "multi":
                single_is_multi += 1
    rng = np.random.default_rng(SEED)
    c_lo, c_hi = clopper_pearson(single_is_multi, single_total)
    b_lo, b_hi = bootstrap_ci_proportion(single_is_multi, single_total, rng)
    contamination = {
        "definition": "M0=single 样本中 deepseek 复标为 multi 的比例",
        "m0_single_n": single_total,
        "relabeled_multi_n": single_is_multi,
        "rate": single_is_multi / single_total if single_total else None,
        "ci_95_clopper_pearson": [c_lo, c_hi],
        "ci_95_bootstrap": [b_lo, b_hi],
    }

    # --- 单向性检验：single->multi vs multi->single 是否对称 ---
    s2m = 0
    m2s = 0
    for r in sample:
        m0 = r["m0"]["multi_barreled"]
        ds = (rel_by_key.get((r["dialogue_id"], r["turn_idx"])) or {}).get("multi_barreled")
        if m0 == "single" and ds == "multi":
            s2m += 1
        elif m0 == "multi" and ds == "single":
            m2s += 1
    # McNemar 精确检验（二项符号检验，双侧）
    n_disc = s2m + m2s
    from math import comb
    p_two_sided = 0.0
    if n_disc > 0:
        for k in range(0, n_disc + 1):
            prob = comb(n_disc, k) * (0.5 ** n_disc)
            if abs(k - n_disc / 2) >= abs(s2m - n_disc / 2):
                p_two_sided += prob
    directionality = {
        "single_to_multi": s2m,
        "multi_to_single": m2s,
        "mcnemar_p_two_sided": float(p_two_sided),
        "verdict": ("单向（系统偏差）" if p_two_sided < 0.05
                    else "对称（噪声）" if n_disc > 0 else "无分歧"),
    }

    # --- 三档判定（先写定再看结果） ---
    rate = contamination["rate"]
    if rate is None:
        verdict = "无法判定"
        followup = "复标缺失，需重跑 relabel"
    elif rate < 0.05:
        verdict = "标注噪声（<5%），不影响结论"
        followup = "R13 降为「已量化，可忽略」；M1/M2b 不动；M3 照常"
    elif rate <= 0.15:
        verdict = "效应被稀释（5%–15%），方向不变"
        followup = "M1 四个效应量重算并同报旧值；M2b 表加敏感性列；M3 可用但须注明"
    else:
        verdict = "分组基础失效（>15%）"
        followup = "停 M3；M0 三维重标 → M1/M2b 全部重估"

    result = {
        "seed": SEED,
        "sample_n": len(sample),
        "per_dim": per_dim,
        "contamination_rate": contamination,
        "directionality": directionality,
        "verdict": {"rate": rate, "conclusion": verdict, "followup": followup},
    }
    json.dump(result, open(RESULT_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("=== 三维一致率 + CI ===")
    for d in DIMS:
        v = per_dim[d]
        print(f"  {d}: {v['agreement_rate']:.1%}  CI [{v['ci_95'][0]:.1%}, {v['ci_95'][1]:.1%}]  (n={v['n']})")
    print("=== 混淆矩阵（m0->deepseek 分歧） ===")
    for d in DIMS:
        print(f"  {d}: {per_dim[d]['confusion']}")
    print("=== 污染率（核心） ===")
    print(f"  M0=single {single_total} 条，deepseek 判 multi {single_is_multi} 条")
    print(f"  污染率 {rate:.1%}  CP-CI [{c_lo:.1%}, {c_hi:.1%}]  bootstrap-CI [{b_lo:.1%}, {b_hi:.1%}]")
    print("=== 单向性 ===")
    print(f"  single->multi {s2m}   multi->single {m2s}   McNemar p={p_two_sided:.4f}  -> {directionality['verdict']}")
    print("=== 三档判定 ===")
    print(f"  {verdict}  ->  {followup}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "relabel", "report"])
    args = ap.parse_args()
    {"sample": cmd_sample, "relabel": cmd_relabel, "report": cmd_report}[args.stage]()


if __name__ == "__main__":
    main()
