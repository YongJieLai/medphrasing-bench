"""M2b-0 前置粒度核查：确认真值集「中位 4.0」是对话短造成的，还是抽取器漏拆。

四步（章程 §7.2 M2b-0）：
1. 报「AIU 数 / 问答对数」分布（均值、中位），期望 0.8–1.5
2. 分层抽 20 段（size 1–3 / 4–6 / 7+），用 qwen-plus(reviewer) 逐条判断「应拆未拆」
3. 判定：仅 size 4–6 层漏拆 > 20% 才需改抽取 prompt 重跑
4. 顺带产出：|T_d| × 对话轮次 相关系数，|T_d|≥12 的对话数及平均轮次

用法:
  python scripts/m2b0_granularity.py data/aiu_truthset_20261126.json \
      data/annot_all_flash.jsonl --out outputs/20261126_m2b0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.llm import chat

REPO = Path(__file__).resolve().parent.parent

REVIEW_SYSTEM = """You are auditing an AIU (atomic information unit) extractor for UNDER-SPLITTING: one AIU that bundles MULTIPLE substantive, independently-actionable clinical facts.

UNDER-SPLIT (flag true) ONLY when the AIU bundles >=2 SUBSTANTIVE facts that each require separate clinical judgment, e.g.:
- "spider bite on shoulder, tender, red, growing" -> location / tenderness / erythema / progression
- "no blood in stool, no alcohol, no vomiting blood" -> 3 separate negations
- "nausea and vomiting and diarrhea" -> 3 symptoms
- "I took OTC antacids, got relief, and did not see a doctor" -> 3 separate facts

NOT under-split (flag false):
- A symptom with its natural modifier: "fever for 3 days", "dark-colored blood in stool", "married for 10 years", "79-year-old female".
- A single broad negation: "no neurological symptoms", "no blood disorders" (ONE negation = ONE fact).
- Demographics in one phrase: "26 years old with a son born 2008".

Be CONSERVATIVE: only flag CLEAR bundling of substantive independent facts. When in doubt, flag false.

PATIENT UTTERANCES (context):
\"\"\"
{patient}
\"\"\"

AIU LIST:
{aius}

Output STRICT JSON only:
{{"judgments":[{{"text":"...","under_split":true,"reason":"..."}},...],"n_under_split":<int>}}"""


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truthset")
    ap.add_argument("annot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_sample", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "run.log", "w", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")

    # ---------- 读数据 ----------
    truth = json.load(open(args.truthset, encoding="utf-8"))
    annot_rows = [json.loads(l) for l in open(args.annot, encoding="utf-8")]

    # 每对话的患者回答原文（用于 review 上下文）
    patient_utts = {}
    for r in annot_rows:
        if r.get("is_question") == 1 and not r.get("label_error"):
            patient_utts.setdefault(r["dialogue_id"], []).append(r["patient_utt"])

    log(f"真值集对话数: {len(truth)}")
    log(f"标注行数: {len(annot_rows)}")

    # ---------- 第 1 步：AIU 数 / 问答对数 ----------
    rows = []
    for t in truth:
        did = t["dialogue_id"]
        n_pairs = t["n_pairs"]
        raw = t["raw_aiu_count"]
        size = t["truth_set_size"]
        rows.append({
            "dialogue_id": did,
            "n_pairs": n_pairs,
            "raw_aiu": raw,
            "truth_size": size,
            "aiu_per_pair": raw / n_pairs if n_pairs > 0 else np.nan,
        })
    df = pd.DataFrame(rows)

    total_pairs = int(df["n_pairs"].sum())
    total_raw = int(df["raw_aiu"].sum())
    log(f"\n=== 第 1 步：AIU 数 / 问答对数 ===")
    log(f"总问答对数 n_pairs 合计: {total_pairs}")
    log(f"总 AIU(去重前) 合计: {total_raw}")
    aiu_per_pair = df["aiu_per_pair"].dropna()
    log(f"AIU/问答对数  均值={aiu_per_pair.mean():.3f}  中位={aiu_per_pair.median():.3f}  "
        f"范围=[{aiu_per_pair.min():.3f}, {aiu_per_pair.max():.3f}]")
    q = aiu_per_pair.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    log(f"分位数: P10={q[0.1]:.3f} P25={q[0.25]:.3f} P50={q[0.5]:.3f} "
        f"P75={q[0.75]:.3f} P90={q[0.9]:.3f}")
    in_band = ((aiu_per_pair >= 0.8) & (aiu_per_pair <= 1.5)).mean()
    log(f"落在 0.8–1.5 区间占比: {in_band:.1%}")

    # ---------- 第 4 步：|T_d| × 轮次 相关系数 ----------
    log(f"\n=== 第 4 步：|T_d| × 对话轮次 相关系数 ===")
    x, y = df["n_pairs"].values, df["truth_size"].values
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    log(f"Pearson r = {pearson:.3f}")
    log(f"Spearman ρ = {spearman:.3f}")
    big = df[df["truth_size"] >= 12]
    log(f"|T_d| >= 12 的对话数: {len(big)}")
    if len(big):
        log(f"  其平均轮次(问答对数): {big['n_pairs'].mean():.2f}  "
            f"(全库平均轮次 {df['n_pairs'].mean():.2f})")
    # 联合分布表（轮次档 × 真值集档）
    df["turn_bin"] = pd.cut(df["n_pairs"], bins=[0, 3, 6, 10, 100],
                            labels=["1-3轮", "4-6轮", "7-10轮", "11+轮"])
    df["size_bin"] = pd.cut(df["truth_size"], bins=[-1, 3, 6, 12, 200],
                            labels=["1-3", "4-6", "7-11", "12+"])
    ctab = pd.crosstab(df["turn_bin"], df["size_bin"])
    log(f"\n联合分布 (轮次档 × 真值集档):\n{ctab.to_string()}")

    # ---------- 第 2 步：分层抽样 20 段 ----------
    log(f"\n=== 第 2 步：分层抽样 {args.n_sample} 段 ===")
    rng = np.random.default_rng(args.seed)
    d1 = df[(df["truth_size"] >= 1) & (df["truth_size"] <= 3)]
    d2 = df[(df["truth_size"] >= 4) & (df["truth_size"] <= 6)]
    d3 = df[df["truth_size"] >= 7]
    # 判定层 4-6 多抽一点
    n2 = max(8, args.n_sample // 3)
    n1 = (args.n_sample - n2) // 2
    n3 = args.n_sample - n1 - n2
    log(f"各层样本量: 1-3层={n1}  4-6层={n2}  7+层={n3}")
    log(f"各层总量: 1-3层={len(d1)}  4-6层={len(d2)}  7+层={len(d3)}")

    def sample_layer(d, n):
        if len(d) == 0 or n <= 0:
            return []
        idx = rng.choice(d.index, size=min(n, len(d)), replace=False)
        return idx.tolist()

    sampled_idx = (sample_layer(d1, n1) + sample_layer(d2, n2) + sample_layer(d3, n3))
    sampled = df.loc[sampled_idx].reset_index(drop=True)
    log(f"实际抽样: {len(sampled)} 段")

    # ---------- 第 2 步续：reviewer 漏拆检查 ----------
    log(f"\n=== reviewer(qwen-plus) 漏拆检查 ===")
    truth_by_id = {t["dialogue_id"]: t for t in truth}
    results = []
    t0 = time.time()
    for i, s in sampled.iterrows():
        did = int(s["dialogue_id"])
        t = truth_by_id[did]
        aius = t["truth_set"]
        ptext = "\n".join(patient_utts.get(did, []))[:4000]
        aiu_payload = [{"text": a["text"], "fidelity": a["fidelity"]} for a in aius]
        prompt = REVIEW_SYSTEM.format(patient=ptext or "(none)",
                                      aius=json.dumps(aiu_payload, ensure_ascii=False))
        judged = None
        for attempt in range(3):
            try:
                raw = chat("reviewer", [{"role": "user", "content": prompt}],
                           temperature=0.0, max_tokens=1024)
                judged = parse_json(raw)
                if judged and isinstance(judged.get("judgments"), list):
                    break
            except Exception as e:
                log(f"  [warn] did={did} attempt{attempt}: {e}")
            time.sleep(1)
        n_under = judged.get("n_under_split") if judged else None
        judgments = judged.get("judgments", []) if judged else []
        results.append({
            "dialogue_id": did,
            "truth_size": t["truth_set_size"],
            "n_pairs": t["n_pairs"],
            "layer": ("1-3" if t["truth_set_size"] <= 3 else
                      "4-6" if t["truth_set_size"] <= 6 else "7+"),
            "n_aius": len(aius),
            "n_under_split": n_under,
            "judgments": judgments,
            "patient_utts": patient_utts.get(did, []),
        })
        log(f"  [{i+1}/{len(sampled)}] did={did} size={t['truth_set_size']} "
            f"aius={len(aius)} under_split={n_under}")
    log(f"review 完成，耗时 {time.time()-t0:.0f}s")

    # ---------- 第 3 步：判定 ----------
    log(f"\n=== 第 3 步：判定 ===")
    res_df = pd.DataFrame(results)
    for layer in ["1-3", "4-6", "7+"]:
        sub = res_df[res_df["layer"] == layer]
        if len(sub) == 0:
            continue
        total_aius = int(sub["n_aius"].sum())
        total_under = int(sub["n_under_split"].fillna(0).sum())
        rate = total_under / total_aius if total_aius else 0.0
        log(f"  {layer} 层: {len(sub)} 段 / {total_aius} 条 AIU / "
            f"漏拆 {total_under} 条 / 漏拆率 {rate:.1%}")
    # 4-6 层判定
    sub46 = res_df[res_df["layer"] == "4-6"]
    if len(sub46):
        total_aius = int(sub46["n_aius"].sum())
        total_under = int(sub46["n_under_split"].fillna(0).sum())
        rate46 = total_under / total_aius if total_aius else 0.0
        verdict = "不通过（需改抽取 prompt 重跑）" if rate46 > 0.20 else "通过"
        log(f"  → size 4-6 层漏拆率 {rate46:.1%}  {'>' if rate46>0.2 else '<='} 20%  → {verdict}")

    # ---------- 落盘 ----------
    json.dump(results, open(outdir / "sample_20.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    config = {
        "seed": args.seed,
        "n_sample": args.n_sample,
        "layers": {"1-3": n1, "4-6": n2, "7+": n3},
        "review_model": "qwen-plus (reviewer)",
        "aiu_per_pair_mean": float(aiu_per_pair.mean()),
        "aiu_per_pair_median": float(aiu_per_pair.median()),
        "pearson": pearson,
        "spearman": spearman,
        "n_big_td_ge12": int(len(big)),
    }
    json.dump(config, open(outdir / "config.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log(f"\n已写出: {outdir}/sample_20.json, config.json, run.log")
    logf.close()


if __name__ == "__main__":
    main()
