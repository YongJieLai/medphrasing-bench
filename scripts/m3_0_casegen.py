"""M3-0 预检：合成病例池 v1 生成（§7.12 第 1 件，M3 正题复用）。

每个病例：主诉(中文) + 10-15 条事实(英文 AIU，含 fidelity + category 四类槽)
         + 金标准诊断(中文) + 鉴别诊断 2-3 个 + 难度分层。

满足 R12：事实覆盖 symptom/history/med/social 四类槽，使医生可问的常见维度
都落在真值集内（避免「无槽脑补」）。金标准单一可判；诊断名去重 ≥ 80%。

用法：
  python scripts/m3_0_casegen.py --out outputs/m3_0/cases.jsonl --per-tier 35
  python scripts/m3_0_casegen.py --out outputs/m3_0/cases.jsonl --per-tier 5 --limit-pilot
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat

TIERS = ["easy", "medium", "hard"]
TIER_LABEL = {"easy": "easy", "medium": "medium", "hard": "hard"}
CATEGORIES = {"symptom", "history", "med", "social"}
FIDELITIES = {"definite", "fuzzy", "denied"}

PROMPT = """You are a medical education expert generating synthetic clinical cases to benchmark how a doctor's QUESTION PHRASING (open-ended vs yes/no) affects diagnostic accuracy. Generate {n} cases of difficulty "{tier}" ({tier_label}).

Each case MUST contain these fields:
1. chief_complaint: the patient's opening complaint (English, one sentence).
2. facts: 10-15 atomic facts the patient "knows" (English), each with:
   - text: fact description (English, e.g. "pain in right lower quadrant for 2 days", "no fever")
   - fidelity: "definite" (affirmed) / "fuzzy" (hedged) / "denied" (explicit negation)
   - category: "symptom" / "history" / "med" / "social"
   Hard requirements:
   - all 4 categories (symptom/history/med/social) must appear at least once;
   - at least 3 denied facts (key negative findings that rule out differentials);
   - facts must not contradict each other; each fact is an independently verifiable atom.
3. gold_diagnosis: the single, unambiguous gold-standard diagnosis (English, e.g. "acute appendicitis"; do NOT write "rule out ..." or "unspecified").
4. differentials: 2-3 differential diagnoses (English).
5. difficulty: "{tier}"

Difficulty ({tier}={tier_label}):
- easy: differentials differ clearly from the gold on key symptoms; routine history-taking distinguishes them.
- medium: partial overlap between differentials and gold; targeted questioning is needed.
- hard: high clinical overlap; only subtle differences (a key negative finding / trigger / associated symptom) separate them.

Medical requirements:
- Only common diseases; the gold diagnosis must be inferable from facts (facts contain enough info to support it and exclude differentials).
- facts MUST be consistent with gold_diagnosis: do NOT include any fact that would invalidate the diagnosis (e.g. "acute appendicitis" with "appendectomy performed at age 12").
- Diagnoses should be diverse across systems (GI / respiratory / cardiovascular / urinary / neuro / endocrine etc.).
- Do NOT reuse any diagnosis in this excluded list: {exclude}

Output STRICT JSON array of case objects only, no explanation, no markdown."""


def parse_json_array(raw: str) -> list | None:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else None
    except json.JSONDecodeError:
        return None


def validate_case(c: dict) -> str | None:
    """返回 None 表示合法，否则返回错误信息。"""
    if not isinstance(c, dict):
        return "非对象"
    cc = c.get("chief_complaint")
    if not cc or not isinstance(cc, str) or not cc.strip():
        return "缺主诉"
    facts = c.get("facts")
    if not isinstance(facts, list) or not (10 <= len(facts) <= 15):
        return f"事实数 {len(facts) if isinstance(facts, list) else 'N/A'} 不在 10-15"
    cats = set()
    denied = 0
    for f in facts:
        if not isinstance(f, dict):
            return "事实元素非对象"
        if not f.get("text") or not isinstance(f["text"], str):
            return "事实缺 text"
        if f.get("fidelity") not in FIDELITIES:
            return f"非法 fidelity {f.get('fidelity')}"
        if f.get("category") not in CATEGORIES:
            return f"非法 category {f.get('category')}"
        cats.add(f["category"])
        if f["fidelity"] == "denied":
            denied += 1
    if len(cats) < 4:
        return f"四类槽未全覆盖，缺 {CATEGORIES - cats}"
    if denied < 3:
        return f"denied 事实 {denied} < 3"
    gd = c.get("gold_diagnosis")
    if not gd or not isinstance(gd, str) or not gd.strip():
        return "缺金标准诊断"
    gd_l = gd.lower()
    if any(w in gd_l for w in ("rule out", "r/o", "unspecified", "待查", "待诊")):
        return f"金标准不可判 {gd}"
    diffs = c.get("differentials")
    if not isinstance(diffs, list) or not (2 <= len(diffs) <= 3):
        return "鉴别诊断数不在 2-3"
    if not all(isinstance(d, str) and d.strip() for d in diffs):
        return "鉴别诊断非法"
    if c.get("difficulty") not in TIERS:
        return f"非法 difficulty {c.get('difficulty')}"
    return None


def gen_tier(tier: str, n: int, batch: int = 8) -> list[dict]:
    """分批生成（每批 batch 个），循环直到凑够 n 个。避免一次生成过多被 token 截断。"""
    valid_all: list[dict] = []
    seen_diag: set[str] = set()
    max_batches = n + 15  # 防无限循环（LLM 持续失败时兜底）
    batches = 0
    while len(valid_all) < n and batches < max_batches:
        batches += 1
        remaining = n - len(valid_all)
        want = min(batch, remaining)
        exclude = ", ".join(sorted(seen_diag)) if seen_diag else "(none yet)"
        prompt = PROMPT.format(n=want, tier=tier, tier_label=TIER_LABEL[tier],
                               exclude=exclude)
        got = 0
        for attempt in range(3):
            try:
                raw = chat("reviewer", [{"role": "user", "content": prompt}],
                           temperature=0.7, max_tokens=8192)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {tier} 调用失败: {e}", flush=True)
                time.sleep(2)
                continue
            arr = parse_json_array(raw)
            if not arr:
                print(f"  [warn] {tier} 解析失败 (attempt {attempt+1})，尾 80 字: {raw[-80:]!r}",
                      flush=True)
                continue
            for c in arr:
                err = validate_case(c)
                if err is not None:
                    print(f"  [drop] {tier} 病例不合法: {err}", flush=True)
                    continue
                d = c["gold_diagnosis"]
                if d in seen_diag:
                    print(f"  [skip] {tier} 诊断重复: {d}", flush=True)
                    continue
                seen_diag.add(d)
                valid_all.append(c)
                got += 1
            if got > 0:
                break
            print(f"  [retry] {tier} 本批解析 0 个", flush=True)
        print(f"  {tier}: 已累计 {len(valid_all)}/{n}", flush=True)
    return valid_all[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/m3_0/cases.jsonl")
    ap.add_argument("--per-tier", type=int, default=35)
    ap.add_argument("--pilot", action="store_true", help="仅每档 5 个，快速验证")
    ap.add_argument("--roles", choices=["reviewer", "builder"], default="reviewer")
    args = ap.parse_args()

    n = 5 if args.pilot else args.per_tier
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_cases = []
    t0 = time.time()
    for tier in TIERS:
        print(f"生成 {tier} ({TIER_LABEL[tier]}) ...")
        cases = gen_tier(tier, n)
        for c in cases:
            c["case_id"] = f"case_{len(all_cases) + 1:03d}"
            all_cases.append(c)
        print(f"  {tier}: {len(cases)} 个，累计 {len(all_cases)}，用时 {time.time()-t0:.0f}s")

    # 写入
    with open(out, "w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    diag = [c["gold_diagnosis"] for c in all_cases]
    dup = len(diag) - len(set(diag))
    uniq_rate = len(set(diag)) / len(diag) if diag else 0.0
    tier_cnt = Counter(c["difficulty"] for c in all_cases)

    print("\n=== 病例池汇总 ===")
    print(f"总数 {len(all_cases)}")
    print(f"难度分层: {dict(tier_cnt)}")
    print(f"诊断去重: 唯一 {len(set(diag))}/{len(diag)} = {uniq_rate:.1%}（重复 {dup}）")
    print(f"去重判据 ≥80% → {'✅' if uniq_rate >= 0.8 else '❌'}")
    print(f"每层 ≥25 → {'✅' if all(tier_cnt[t] >= 25 for t in TIERS) else '❌'}")

    meta = {
        "n_cases": len(all_cases),
        "tier_counts": dict(tier_cnt),
        "diagnosis_unique_rate": uniq_rate,
        "diagnosis_dup": dup,
        "roles": args.roles,
        "per_tier_requested": n,
        "pilot": args.pilot,
    }
    json.dump(meta, open(out.with_suffix(".meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n完成 -> {out}")


if __name__ == "__main__":
    main()
