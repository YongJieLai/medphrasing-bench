"""M2a 核心：从患者回答抽取 AIU 文本（非计数）+ 跨轮去重 + 保真度标注。

背景（章程 §2.1）：现有标注只有 info_units 计数，全库无一条 AIU 文本。
模拟器必须知道「这个患者知道哪几件事」才能决定说哪几件 -> 重建真值集 T_d。

流程：
1. 对每个对话，逐条患者回答抽取 AIU（含保真度 definite/fuzzy/denied）。
2. 合并该对话全部 AIU，语义去重 -> 真值集 T_d（每个患者「知道哪几件事」）。
3. 输出 JSON，供 M2b 参数表 / M2c 生成 / M2d 回放使用。

注意（禁止项）：
- 只做抽取，不生成任何「患者回答」。
- 不改 rubric 口径。
- 不因「抽不出 3 个以上 AIU」而放宽 AIU 定义。

用法:
  python scripts/extract_aiu.py data/annot_all_flash.jsonl --limit 20 --out data/aiu_pilot.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat

REPO = Path(__file__).resolve().parent.parent

AIU_SYSTEM = """You extract atomic information units (AIUs) from a patient's utterance in a medical dialogue. Use the doctor's question ONLY to resolve references (pronouns / ellipsis / "the ones").

An AIU is a MINIMAL, independently-verifiable clinical fact. Rules:
- Split compound statements into separate AIUs: "fever for 3 days, worse at night" -> "has fever", "duration 3 days", "worse at night".
- Each AIU is ONE fact, phrased briefly in first person.
- Label fidelity for each AIU:
  * "definite": stated affirmatively and concretely
  * "fuzzy": hedged / uncertain ("maybe", "kind of", "not sure", "a little", "I think")
  * "denied": the patient explicitly negates the doctor's asked fact. If the doctor asks "any fever?" and the patient says "no", extract {{"text": "no fever", "fidelity": "denied"}}.
- IMPORTANT: a "no/only" answer still contains information. Extract both the negated fact (denied) AND any affirmed alternative (definite). Example: doctor "are you taking OTC medicines?" + patient "No, only the prescribed ones" -> [{{"text": "not taking OTC medicines", "fidelity": "denied"}}, {{"text": "taking prescribed medicines", "fidelity": "definite"}}].
- Only if the utterance is a pure acknowledgment / greeting with zero clinical content, return an empty list.

DOCTOR'S QUESTION:
\"\"\"
{doctor}
\"\"\"

PATIENT'S UTTERANCE:
\"\"\"
{patient}
\"\"\"

Output STRICT JSON only:
{{"aius": [{{"text": "...", "fidelity": "definite"}}, ...]}}"""

DEDUP_SYSTEM = """You deduplicate a list of atomic information units (AIUs) collected across a patient-doctor dialogue.

Merge AIUs that state the SAME fact (paraphrases, or a more specific restatement of an earlier fact). Keep the most specific / informative version, and the most confident fidelity (definite > fuzzy > denied). If two AIUs are related but distinct facts, keep both. A "denied" fact and its affirmed counterpart are DIFFERENT (keep both).

Output the deduplicated list as a JSON array of objects, in first person.

AIU LIST:
{aiu_list}

Output STRICT JSON only:
[{{"text": "fact 1", "fidelity": "definite"}}, ...]"""


def parse_json(raw: str) -> dict | list | None:
    m = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def extract_aius(doctor_utt: str, patient_utt: str, retries: int = 2) -> list[dict]:
    prompt = AIU_SYSTEM.format(doctor=doctor_utt, patient=patient_utt)
    for attempt in range(retries + 1):
        try:
            raw = chat("extractor", [{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=512)
            d = parse_json(raw)
            if isinstance(d, dict) and isinstance(d.get("aius"), list):
                aius = [a for a in d["aius"]
                        if isinstance(a, dict) and a.get("text")
                        and a.get("fidelity") in ("definite", "fuzzy", "denied")]
                return aius
        except Exception as e:
            print(f"    [warn] {e}")
        time.sleep(1)
    return []


def dedup(aius: list[dict], retries: int = 2) -> list[dict]:
    if not aius:
        return []
    payload = [{"text": a["text"], "fidelity": a["fidelity"]} for a in aius]
    prompt = DEDUP_SYSTEM.format(aiu_list=json.dumps(payload, ensure_ascii=False))
    for attempt in range(retries + 1):
        try:
            raw = chat("extractor", [{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=1024)
            d = parse_json(raw)
            if isinstance(d, list):
                out = [a for a in d
                       if isinstance(a, dict) and a.get("text")
                       and a.get("fidelity", "definite") in ("definite", "fuzzy", "denied")]
                if out:
                    return out
        except Exception as e:
            print(f"    [warn] {e}")
        time.sleep(1)
    return aius  # 去重失败则退化为原列表


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个对话（pilot）")
    ap.add_argument("--shard", type=int, default=None, help="分片编号（0-based，配合 --nshards）")
    ap.add_argument("--nshards", type=int, default=1, help="分片总数")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = [r for r in rows if not r.get("label_error") and r.get("is_question") == 1]

    # 按对话分组，保留 (turn_idx, patient_utt)
    by_dialogue: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_dialogue[r["dialogue_id"]].append(r)

    dial_ids = sorted(by_dialogue.keys())
    if args.limit:
        dial_ids = dial_ids[: args.limit]

    if args.shard is not None and args.nshards > 1:
        chunk = (len(dial_ids) + args.nshards - 1) // args.nshards
        start = args.shard * chunk
        end = min(start + chunk, len(dial_ids))
        dial_ids = dial_ids[start:end]
        print(f"分片 {args.shard}/{args.nshards}: 对话 [{start}:{end}) 共 {len(dial_ids)} 个")

    out_records = []
    t0 = time.time()
    for i, did in enumerate(dial_ids):
        pairs = sorted(by_dialogue[did], key=lambda x: x["turn_idx"])
        all_aius: list[dict] = []
        for p in pairs:
            aius = extract_aius(p["doctor_utt"], p["patient_utt"])
            p["aius_extracted"] = aius
            all_aius.extend(aius)

        # 跨轮去重（保真度保留，合并时取最确定）
        deduped = dedup(all_aius)

        out_records.append({
            "dialogue_id": did,
            "n_pairs": len(pairs),
            "truth_set": deduped,          # 真值集 T_d
            "truth_set_size": len(deduped),
            "raw_aiu_count": len(all_aius),  # 去重前
        })
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  进度 {i+1}/{len(dial_ids)}  耗时 {el:.0f}s")

    out_path = Path(args.out)
    if args.shard is not None and args.nshards > 1:
        out_path = out_path.with_name(f"{out_path.stem}-shard{args.shard:02d}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    # 汇总
    sizes = [r["truth_set_size"] for r in out_records]
    import statistics
    print(f"\n完成 {len(out_records)} 对话 -> {out_path}")
    print(f"真值集大小: 中位数={statistics.median(sizes):.1f}  "
          f"均值={statistics.mean(sizes):.2f}  范围=[{min(sizes)}, {max(sizes)}]")
    big = [r for r in out_records if r["truth_set_size"] > 20]
    print(f">20 的对话: {len(big)} 个 {[r['dialogue_id'] for r in big]}")


if __name__ == "__main__":
    main()
