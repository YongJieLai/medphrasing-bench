"""M2a 核对：AIU 抽取的 recall/precision + 保真度一致性（强模型交叉替代人工）。

用 reviewer(qwen-plus) 作为参考抽取，评估 extractor(qwen-flash) 的抽取质量：
- 对每条患者回答，flash 抽 A 组 AIU、plus 抽 B 组 AIU。
- LLM 合并去重 A∪B，反推交集 |A∩B| = |A| + |B| - |A∪B|。
- precision = |A∩B| / |A|（flash 抽出的有多少是 plus 也认同的）
- recall    = |A∩B| / |B|（plus 抽出的有多少是 flash 也抽到的）
- 保真度一致性：回答层面保真度聚合 vs definiteness 字段。

用法:
  python scripts/check_aiu_agreement.py data/annot_all_flash.jsonl --limit 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat
from scripts.extract_aiu import AIU_SYSTEM, parse_json

MERGE_SYSTEM = """You merge two lists of atomic information units (AIUs) and deduplicate across BOTH lists.

Merge AIUs that state the SAME fact. Output the merged DISTINCT list as a JSON array of strings (just the fact text, no fidelity).

LIST A:
{a}

LIST B:
{b}

Output STRICT JSON only:
["fact 1", "fact 2", ...]"""


def extract(role: str, doctor: str, patient: str) -> list[dict]:
    prompt = AIU_SYSTEM.format(doctor=doctor, patient=patient)
    for _ in range(2):
        try:
            raw = chat(role, [{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=512)
            d = parse_json(raw)
            if isinstance(d, dict) and isinstance(d.get("aius"), list):
                return [a for a in d["aius"]
                        if isinstance(a, dict) and a.get("text")
                        and a.get("fidelity") in ("definite", "fuzzy", "denied")]
        except Exception as e:
            print(f"    [warn] {e}")
        time.sleep(1)
    return []


def merge(a_texts: list[str], b_texts: list[str]) -> int:
    """返回 |A∪B|（合并去重后数量）。"""
    if not a_texts and not b_texts:
        return 0
    prompt = MERGE_SYSTEM.format(
        a=json.dumps(a_texts, ensure_ascii=False),
        b=json.dumps(b_texts, ensure_ascii=False))
    for _ in range(2):
        try:
            raw = chat("reviewer", [{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=1024)
            d = parse_json(raw)
            if isinstance(d, list):
                return len([s for s in d if isinstance(s, str) and s.strip()])
        except Exception as e:
            print(f"    [warn] {e}")
        time.sleep(1)
    return len(set(a_texts) | set(b_texts))  # 失败退化


def fidelity_to_definiteness(aius: list[dict]) -> str:
    """回答层面：AIU 保真度聚合 -> 推断 definiteness。"""
    if not aius:
        return "agree"
    fids = [a["fidelity"] for a in aius]
    if "fuzzy" in fids:
        return "vague"
    return "exact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = [r for r in rows if not r.get("label_error") and r.get("is_question") == 1]

    # 分层抽样：按 definiteness 分层，保证 agree/vague 稀有类也抽到
    import random
    rng = random.Random(args.seed)
    by_def = {}
    for r in valid:
        by_def.setdefault(r["definiteness"], []).append(r)
    sample = []
    for defv, group in by_def.items():
        k = max(1, round(args.limit * len(group) / len(valid)))
        sample.extend(rng.sample(group, min(k, len(group))))
    rng.shuffle(sample)
    sample = sample[: args.limit]
    print(f"分层抽样 {len(sample)} 条（definiteness 分布: {Counter(r['definiteness'] for r in sample)}）")

    recs = []
    precisions, recalls = [], []
    fidelity_agree = 0
    t0 = time.time()
    for i, r in enumerate(sample):
        a = extract("extractor", r["doctor_utt"], r["patient_utt"])  # flash
        b = extract("reviewer", r["doctor_utt"], r["patient_utt"])   # plus
        a_texts = [x["text"] for x in a]
        b_texts = [x["text"] for x in b]
        union = merge(a_texts, b_texts)
        na, nb = len(a_texts), len(b_texts)
        inter = na + nb - union
        prec = inter / na if na else 1.0
        rec = inter / nb if nb else 1.0
        precisions.append(prec)
        recalls.append(rec)

        # 保真度一致性
        pred = fidelity_to_definiteness(a)
        if pred == r["definiteness"]:
            fidelity_agree += 1

        recs.append({
            "dialogue_id": r["dialogue_id"], "turn_idx": r["turn_idx"],
            "definiteness": r["definiteness"],
            "n_flash": na, "n_plus": nb, "n_union": union,
            "precision": prec, "recall": rec,
        })
        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{len(sample)}  {time.time()-t0:.0f}s")

    # 汇总
    import statistics
    print("\n" + "=" * 60)
    print(f"n={len(recs)} 条")
    print(f"precision: mean={statistics.mean(precisions):.3f}  median={statistics.median(precisions):.3f}")
    print(f"recall   : mean={statistics.mean(recalls):.3f}  median={statistics.median(recalls):.3f}")
    print(f"保真度 vs definiteness 一致率: {fidelity_agree/len(recs)*100:.1f}%")

    # 按 definiteness 分组看 recall
    print("\n按 definiteness 分组的 recall/precision:")
    by_def2 = {}
    for rec in recs:
        by_def2.setdefault(rec["definiteness"], []).append(rec)
    for defv, grp in sorted(by_def2.items()):
        pr = statistics.mean(g["precision"] for g in grp)
        rc = statistics.mean(g["recall"] for g in grp)
        print(f"  {defv:8s}: n={len(grp):3d}  precision={pr:.3f}  recall={rc:.3f}")


if __name__ == "__main__":
    main()
