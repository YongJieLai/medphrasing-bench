"""M3-0b 归因（§7.13）：分清「假病人钝」(H_b) 还是「考题简单」(H_a)。

两个检查，只做归因，不改任何已有数据、不重建病例池：

  检查 1（零成本，只读）——从 results.jsonl 回抽患者【实际发言】的 AIU 数，
      报四组 A-12 / B-12 / A-6 / B-6 的五个数，判 H_b / H_a / 混合。
      回抽器用纯词法（非 LLM），与医生(qwen-flash)/诊断(deepseek-chat)/
      判别(qwen-plus) 均不同源。

  检查 2（约 1/20 成本）——只重跑诊断，不重跑问诊：同一段 12 轮对话，
      把「三选一候选」改成「开放式自由作答」，deepseek-chat 判诊断，
      与 gold 做语义等价判定（qwen 系列，与诊断不同源）。

用法：
  python scripts/m3_0b_attribution.py --check 1   # 只跑检查 1（零成本）
  python scripts/m3_0b_attribution.py --check 2   # 只跑检查 2
  python scripts/m3_0b_attribution.py --check all # 两个都跑
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src.llm import chat  # 判等用 qwen-plus（reviewer），与诊断 deepseek 不同源
from src.relevance_scorer import STOPWORDS, SYNONYM_MAP  # 复用词法归一

load_dotenv()

# 数字归一（英文病例 facts 用阿拉伯数字、患者口语用英文单词）
_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}


def _normalize(text: str) -> list[str]:
    """小写、去标点、去停用词、同义词+数字归一，返回内容词序列。"""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = []
    for t in text.split():
        if t in STOPWORDS:
            continue
        t = SYNONYM_MAP.get(t, t)
        t = _NUM.get(t, t)
        if t:
            toks.append(t)
    return toks


# 覆盖阈值：事实内容词被患者发言覆盖到多少才算「说到了」。
# 多内容词事实 >= 0.5；单内容词事实必须 100%（否则一个词命中就误判）。
COVERAGE_THRESHOLD = 0.5


def fact_covered(patient_utt: str, fact_text: str) -> bool:
    """判断一句患者发言是否「说到了」某条事实（词法覆盖）。"""
    ft = _normalize(fact_text)
    if not ft:
        return False
    ut = _normalize(patient_utt)
    if not ut:
        return False
    uset = set(ut)
    hits = sum(1 for t in ft if t in uset)
    cov = hits / len(ft)
    if len(ft) == 1:
        return cov >= 1.0
    return cov >= COVERAGE_THRESHOLD


def reextract_aius(patient_utt: str, facts: list[dict]) -> int:
    """数出患者发言里包含多少条真值集事实（回抽 AIU 数）。"""
    return sum(1 for f in facts if fact_covered(patient_utt, f["text"]))


def load_cases(cases_path: Path) -> dict[str, list[dict]]:
    return {json.loads(l)["case_id"]: json.loads(l)["facts"]
            for l in open(cases_path, encoding="utf-8")}


def load_results(results_path: Path) -> list[dict]:
    return [json.loads(l) for l in open(results_path, encoding="utf-8")]


def reextract_group(recs: list[dict], cases: dict[str, list[dict]], arm: str):
    """对一组（A 或 B）回抽，返回 [{case_id, turns:[每轮AIU], total, zero_rate}]。"""
    out = []
    for r in recs:
        facts = cases[r["case_id"]]
        turns = r[f"_turns_{arm}"]
        per_turn = [reextract_aius(t["patient"], facts) for t in turns]
        total = sum(per_turn)
        zero = sum(1 for x in per_turn if x == 0)
        out.append({
            "case_id": r["case_id"],
            "per_turn": per_turn,
            "total": total,
            "zero_rate": zero / len(per_turn) if per_turn else 0.0,
        })
    return out


def bootstrap_paired_diff(a_totals, b_totals, seed=42, B=2000):
    """A−B 配对差（按病例重采样）的点估计 + bootstrap 95% CI。"""
    n = len(a_totals)
    a = np.array(a_totals, dtype=float)
    b = np.array(b_totals, dtype=float)
    d = a - b
    est = float(d.mean())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(float(d[idx].mean()))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return est, lo, hi


def check1(cases_path, res12_path, res6_path, outdir: Path):
    cases = load_cases(cases_path)
    res12 = load_results(res12_path)
    res6 = load_results(res6_path)

    print("=" * 70)
    print("检查 1：披露量回抽（只读，零成本）")
    print("=" * 70)

    groups = {
        "A-12": reextract_group(res12, cases, "A"),
        "B-12": reextract_group(res12, cases, "B"),
        "A-6": reextract_group(res6, cases, "A"),
        "B-6": reextract_group(res6, cases, "B"),
    }

    report = {"groups": {}, "verdict": None, "samples_for_audit": []}

    # 指标 1：每段对话总条数（均值 + 中位数 + IQR）
    # 指标 4：零披露轮次占比
    for name, g in groups.items():
        totals = [x["total"] for x in g]
        zr = [x["zero_rate"] for x in g]
        arr = np.array(totals)
        report["groups"][name] = {
            "n": len(totals),
            "total_mean": float(arr.mean()),
            "total_median": float(np.median(arr)),
            "total_iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "zero_rate_mean": float(np.mean(zr)),
        }
        print(f"\n[{name}] n={len(totals)}")
        print(f"  总条数: 均值 {arr.mean():.2f}  中位数 {np.median(arr):.2f}  "
              f"IQR [{np.percentile(arr,25):.0f}, {np.percentile(arr,75):.0f}]")
        print(f"  零披露轮次占比: {np.mean(zr):.3f}")

    # 指标 2：A−B 差（配对，按病例重采样 bootstrap）
    print("\n--- 指标 2：A−B 披露量差（条/对话，配对 bootstrap）---")
    for tag, (aname, bname) in {"12轮": ("A-12", "B-12"), "6轮": ("A-6", "B-6")}.items():
        a = [x["total"] for x in groups[aname]]
        b = [x["total"] for x in groups[bname]]
        est, lo, hi = bootstrap_paired_diff(a, b)
        report["groups"][aname]["diff_vs_B"] = {"est": est, "ci": [lo, hi]}
        print(f"  {tag}: A−B = {est:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

    # 指标 3：每轮均值（分轮次曲线）
    print("\n--- 指标 3：每轮均值（分轮次）---")
    curves = {}
    for name, g in groups.items():
        max_t = max(len(x["per_turn"]) for x in g)
        curve = []
        for t in range(max_t):
            vals = [x["per_turn"][t] for x in g if t < len(x["per_turn"])]
            curve.append(float(np.mean(vals)) if vals else None)
        curves[name] = curve
    # 打印前 6 轮（6轮组只有6轮）对比
    for t in range(6):
        row = f"  turn{t}: "
        for name in ["A-12", "B-12", "A-6", "B-6"]:
            v = curves[name][t] if t < len(curves[name]) else None
            row += f"{name}={v:.2f}  " if v is not None else f"{name}=--  "
        print(row)
    report["curves"] = curves

    # 指标 5：抽 20 段供人工/LLM 比对回抽一致率
    print("\n--- 指标 5：抽 20 段供比对（回抽一致率）---")
    rng = np.random.default_rng(20261127)
    all_turns = []
    for arm in ["A", "B"]:
        for r in res12:
            facts = cases[r["case_id"]]
            for t in r[f"_turns_{arm}"]:
                all_turns.append((r["case_id"], arm, t["turn"],
                                  t["patient"], facts))
    picks = rng.choice(len(all_turns), size=20, replace=False)
    samples = [all_turns[i] for i in picks]
    for i, (cid, arm, tn, utt, facts) in enumerate(samples):
        re = reextract_aius(utt, facts)
        report["samples_for_audit"].append({
            "case_id": cid, "arm": arm, "turn": tn,
            "patient": utt, "reextracted_aiu": re,
        })
        print(f"  [{i}] {cid} {arm} turn{tn} 回抽AIU={re}: {utt[:100]}")
    # 落盘抽检样本（供主控转交用户）
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "check1_audit_samples.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8")

    # 判据（用 12 轮主结果，6 轮为敏感性分析）
    a12 = [x["total"] for x in groups["A-12"]]
    b12 = [x["total"] for x in groups["B-12"]]
    est12, lo12, hi12 = bootstrap_paired_diff(a12, b12)
    print("\n--- 判据（12 轮主结果）---")
    print(f"  A−B = {est12:+.2f} 条/对话（CI [{lo12:+.2f}, {hi12:+.2f}]）")
    # 诊断差（12 轮 acc_A − acc_B = +3.8pp < 5pp，用于 H_a 分支判定）
    if est12 < 1.0:
        verdict = "H_b 成立：假病人对措辞几乎无反应（A−B < 1.0 条）"
        follow = "换数据集/换因变量都无用，先修模拟器的措辞响应（回到 M2d 的 46–51% 幅度）"
    elif est12 >= 2.0:
        verdict = "H_a 成立：信息确实少了但诊断不受影响（A−B ≥ 2.0 且诊断差 < 5pp）"
        follow = "走检查 2 验证，再考虑换难题"
    else:
        verdict = "混合：1.0–2.0 之间，两个都要处理"
        follow = "检查 2 照跑，结论写「两者叠加」"
    report["verdict"] = {"A_minus_B": est12, "ci": [lo12, hi12],
                         "conclusion": verdict, "followup": follow}
    print(f"  -> {verdict}")
    print(f"  -> {follow}")

    (outdir / "check1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n检查 1 完成 -> {outdir}/check1_report.json")
    return report


# ---------------------------------------------------------------------------
# 检查 2：开放式诊断（只重跑诊断，不重跑问诊）
# ---------------------------------------------------------------------------
OPEN_DIAG_PROMPT = """You are a senior diagnostician. Based on the interview transcript below, state the single most likely diagnosis.

Chief complaint: {chief_complaint}

Interview transcript:
{dialogue}

State ONLY the diagnosis name (a short medical term or phrase; no explanation, no prefix, no differential list, no hedging like "most likely")."""

# 判等（qwen-plus，与诊断 deepseek-chat 不同源）
JUDGE_PROMPT = """Are the following two diagnosis phrases the SAME diagnosis? Treat abbreviations and synonyms as equivalent (e.g. "acute appendicitis" = "appendicitis", "MI" = "myocardial infarction", "UTI" = "urinary tract infection").

Diagnosis A: {gold}
Diagnosis B: {pred}

Answer ONLY "yes" or "no"."""


def clean_open_diag(raw: str) -> str:
    """清理开放式诊断输出：去引号、去常见前缀、截断到第一个换行/句号。"""
    r = (raw or "").strip()
    r = re.sub(r"^['\"“”]+|['\"“”]+$", "", r)
    # 去常见前缀
    for pref in ["the most likely diagnosis is", "the diagnosis is",
                 "most likely diagnosis:", "diagnosis:", "the patient has",
                 "likely", "answer:"]:
        r = re.sub(rf"^{re.escape(pref)}\s*", "", r, flags=re.IGNORECASE)
    # 截断到第一个换行或句号
    r = re.split(r"[\n\.;]", r)[0].strip()
    return r


def judge_equal(gold: str, pred: str) -> bool:
    """语义等价判定：规则归一（保守）+ LLM 判等（qwen-plus）。"""
    g = re.sub(r"[^a-z0-9\s]", " ", gold.lower())
    p = re.sub(r"[^a-z0-9\s]", " ", pred.lower())
    g = re.sub(r"\s+", " ", g).strip()
    p = re.sub(r"\s+", " ", p).strip()
    if not p:
        return False
    if g == p:
        return True
    # 包含关系（保守：只接受短语级精确包含，避免误判）
    if g and (g in p or p in g):
        return True
    # LLM 判等兜底
    try:
        ans = chat("reviewer", [{"role": "user", "content":
                                 JUDGE_PROMPT.format(gold=gold, pred=pred)}],
                   temperature=0.0, max_tokens=8).strip().lower()
        return ans.startswith("yes")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] judge_equal: {e}")
        return False


def check2(cases_path, res12_path, outdir: Path):
    cases = {json.loads(l)["case_id"]: json.loads(l)
             for l in open(cases_path, encoding="utf-8")}
    recs = load_results(res12_path)

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1")

    outdir.mkdir(parents=True, exist_ok=True)
    results_file = outdir / "check2_results.jsonl"
    done_ids = set()
    results = []
    if results_file.exists():
        for line in open(results_file, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done_ids.add(r["case_id"])
            results.append(r)
        print(f"[resume] 检查2 已完成 {len(done_ids)} 例，跳过")

    print("=" * 70)
    print("检查 2：开放式诊断（去候选列表，只重跑诊断）")
    print("=" * 70)

    t0 = time.time()
    for i, r in enumerate(recs):
        cid = r["case_id"]
        if cid in done_ids:
            continue
        case = cases[cid]
        gold = case["gold_diagnosis"]
        cc = case["chief_complaint"]

        row = {"case_id": cid, "difficulty": r["difficulty"], "gold": gold}
        for arm in ["A", "B"]:
            turns = r[f"_turns_{arm}"]
            dialogue = "\n".join(f"Doctor: {t['doctor']}\nPatient: {t['patient']}"
                                 for t in turns)
            prompt = OPEN_DIAG_PROMPT.format(chief_complaint=cc, dialogue=dialogue)
            raw = ""
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=64)
                    raw = (resp.choices[0].message.content or "").strip()
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] check2 diag {cid}/{arm}: {e}")
                    time.sleep(1)
            pred = clean_open_diag(raw)
            row[f"pred_{arm}"] = pred
            row[f"raw_{arm}"] = raw
            row[f"correct_{arm}"] = judge_equal(gold, pred)

        results.append(row)
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if len(results) % 10 == 0:
            acc_a = np.mean([x["correct_A"] for x in results])
            acc_b = np.mean([x["correct_B"] for x in results])
            print(f"  进度 {len(results)}/{len(recs)} acc_A={acc_a:.2f} "
                  f"acc_B={acc_b:.2f} Δ={acc_a-acc_b:+.3f} {time.time()-t0:.0f}s")

    # 汇总
    acc_a = np.mean([x["correct_A"] for x in results])
    acc_b = np.mean([x["correct_B"] for x in results])
    delta = acc_a - acc_b
    disc_a = sum(1 for x in results if x["correct_A"] and not x["correct_B"])
    disc_b = sum(1 for x in results if not x["correct_A"] and x["correct_B"])
    n_disc = disc_a + disc_b
    from math import comb
    mcnemar_p = 0.0
    if n_disc > 0:
        for k in range(n_disc + 1):
            prob = comb(n_disc, k) * (0.5 ** n_disc)
            if abs(k - n_disc / 2) >= abs(disc_a - n_disc / 2):
                mcnemar_p += prob

    n = len(results)
    rng = np.random.default_rng(42)
    a = np.array([x["correct_A"] for x in results], dtype=float)
    b = np.array([x["correct_B"] for x in results], dtype=float)
    diffs = []
    for _ in range(2000):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(float(a[idx].mean() - b[idx].mean()))
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    # 分层
    by = defaultdict(list)
    for x in results:
        by[x["difficulty"]].append(x)
    layered = {d: {"n": len(by.get(d, [])),
                   "acc_A": float(np.mean([x["correct_A"] for x in by.get(d, [])]))
                   if by.get(d) else None,
                   "acc_B": float(np.mean([x["correct_B"] for x in by.get(d, [])]))
                   if by.get(d) else None}
               for d in ["easy", "medium", "hard"]}

    # 判据
    if delta > 0.05 and ci_lo > 0:
        verdict = "H_a 成立：饱和是候选列表造成的 → 换数据集/去候选即可，M3 可开工"
    elif -0.03 <= ci_lo and ci_hi <= 0.03:
        verdict = "不是候选列表的问题 → 与检查 1 合并解读，多半指向 H_b"
    else:
        verdict = "功效不足：按 discordant 率报所需样本量，不判定"

    summary = {
        "n": n, "acc_A": float(acc_a), "acc_B": float(acc_b),
        "delta": float(delta), "delta_ci_95": [ci_lo, ci_hi],
        "mcnemar": {"discord_A_only": disc_a, "discord_B_only": disc_b,
                    "p": mcnemar_p},
        "layered": layered, "verdict": verdict,
    }
    print("\n--- 检查 2 结果（开放式诊断）---")
    print(f"  acc_A={acc_a:.3f}  acc_B={acc_b:.3f}  Δ={delta:+.3f}")
    print(f"  95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]  McNemar {disc_a}:{disc_b} p={mcnemar_p:.4f}")
    print(f"  分层: " + " ".join(f"{k}:A{layered[k]['acc_A']:.2f}/B{layered[k]['acc_B']:.2f}"
                                for k in layered if layered[k]["acc_A"] is not None))
    print(f"  绝对准确率 acc_A={acc_a:.3f}（若<0.4 则 12 轮 73% 不可引用）")
    print(f"  判据 -> {verdict}")

    (outdir / "check2_report.json").write_text(
        json.dumps({"summary": summary, "cases": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    # 抽 20 条诊断判等样本供人工核对
    picks = [results[i] for i in rng.choice(n, size=min(20, n), replace=False)]
    (outdir / "check2_judge_samples.jsonl").write_text(
        "\n".join(json.dumps({
            "case_id": p["case_id"], "gold": p["gold"],
            "pred_A": p["pred_A"], "correct_A": p["correct_A"],
            "pred_B": p["pred_B"], "correct_B": p["correct_B"],
        }, ensure_ascii=False) for p in picks) + "\n", encoding="utf-8")
    print(f"\n检查 2 完成 -> {outdir}/check2_report.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="outputs/m3_0/cases.jsonl")
    ap.add_argument("--res12", default="outputs/m3_0/results.jsonl")
    ap.add_argument("--res6", default="outputs/m3_0_t6/results.jsonl")
    ap.add_argument("--out", default="outputs/m3_0b")
    ap.add_argument("--check", default="1", choices=["1", "2", "all"])
    args = ap.parse_args()

    outdir = Path(args.out)
    if args.check in ("1", "all"):
        check1(Path(args.cases), Path(args.res12), Path(args.res6), outdir)
    if args.check in ("2", "all"):
        check2(Path(args.cases), Path(args.res12), outdir)
