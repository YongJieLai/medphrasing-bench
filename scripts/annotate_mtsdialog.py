"""对 MTS-Dialog 的「医生提问 → 患者回答」对做四维×四属性标注。

标注单元: (doctor utterance, 下一条 patient utterance)
医生侧 4 维: is_question / openness / leadingness / multi_barreled / follow_up
患者侧 4 属性: definiteness / completeness / relevance / info_units

用法:
  # 主标注（qwen-flash），抽 200 条
  python scripts/annotate_mtsdialog.py --role annotator --limit 200 --seed 42 --out data/annot_200_flash.jsonl
  # 核对（qwen-plus），同一批
  python scripts/annotate_mtsdialog.py --role reviewer --limit 200 --seed 42 --out data/annot_200_plus.jsonl
  # 全量标注
  python scripts/annotate_mtsdialog.py --role annotator --out data/annot_all_flash.jsonl
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

REPO = Path(__file__).resolve().parent.parent
PARSED = REPO / "data" / "mtsdialog_parsed.json"

Q_STARTS = (
    "what ", "when ", "where ", "why ", "how ", "which ", "who ",
    "do ", "does ", "did ", "is ", "are ", "have ", "has ", "can ",
    "could ", "would ", "any ", "tell me", "describe ", "so ", "and ", "or ",
)

PROMPT = """You are annotating a doctor-patient dialogue for a research study on how a doctor's question PHRASING affects how much a patient discloses.

PREVIOUS PATIENT UTTERANCE (context only; use it to judge follow_up):
"{prev}"

DOCTOR'S UTTERANCE:
"{doctor}"

PATIENT'S NEXT UTTERANCE:
"{patient}"

Annotate the fields below. Output STRICT JSON only (no extra text).

DOCTOR-SIDE (about the doctor's utterance):
- is_question: 1 if the doctor is asking for information; 0 if it is a statement / advice / acknowledgement / greeting.
- openness: classify by how much the question CONSTRAINS the answer.
  * "closed": answerable with yes/no or a choice. Starts with an auxiliary verb (Do/Does/Did/Is/Are/Have/Has/Can/Could/Would/Should) OR "Any..." OR offers "A or B?". Examples: "Do you have a fever?", "Any chest pain or shortness of breath?", "Did it start suddenly or gradually?"
  * "semi": a wh-question expecting ONE concrete fact (a number, name, time, or list). Examples: "What medications are you taking?", "How many packs a day?", "When did it start?", "How old are you?"
  * "open": invites a free narrative/description with no single expected answer. Examples: "Tell me about your pain.", "Describe what happened.", "What brings you in today?", "How are you feeling?"
  If is_question=0, use "n/a".
- leadingness: "neutral" (no answer implied, e.g. "What does the pain feel like?") | "mild" (suggests a possible answer but the patient can freely deny, e.g. "Is the pain sharp?") | "strong" (pushes the patient to confirm, e.g. tag questions "...correct?" / "...right?" / "...isn't it?" or "Your pain is sharp, yes?"). If is_question=0, use "n/a".
- multi_barreled: "single" (asks for ONE piece of info) | "multi" (asks for >=2 distinct pieces in one utterance, e.g. "Do you have fever, cough, or chest pain?"). If is_question=0, use "n/a".
- follow_up: how this question relates to the PREVIOUS PATIENT UTTERANCE (shown above) — "surface" (repeats/rephrases a previous question) | "deepen" (drills into what the patient just said to make it more specific, e.g. patient said "headache", doctor asks "what kind of headache?") | "new" (new topic). If is_question=0 or the previous patient utterance is empty, use "n/a".

PATIENT-SIDE (about the patient's utterance):
- definiteness: "exact" (concrete & verifiable: a number, name, time, or clear description, e.g. "it started 3 days ago", "the pain is sharp") | "vague" (hedged/uncertain without concrete info, e.g. "kind of", "maybe", "not sure", "a little") | "agree" (ONLY confirms the doctor's suggested answer with yes/right, adding nothing new).
- completeness: 1 if the patient answered ALL sub-questions the doctor asked in that utterance; 0 if any sub-question was left unanswered.
- relevance: "on-topic" (answers only what was asked) | "spillover" (answers what was asked AND voluntarily adds related but unasked info) | "off-topic" (drifts away from the question).
- info_units: the count of DISTINCT atomic pieces of information in the patient's utterance (an integer).

Output STRICT JSON:
{{"is_question":..,"openness":"..","leadingness":"..","multi_barreled":"..","follow_up":"..","definiteness":"..","completeness":..,"relevance":"..","info_units":..}}"""


def is_question_rule(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    tl = t.lower()
    return any(tl.startswith(q) for q in Q_STARTS)


def extract_pairs(dialogues: list) -> list[dict]:
    pairs = []
    for di, turns in enumerate(dialogues):
        prev_patient = None
        for i, (sp, text) in enumerate(turns):
            if sp == "doctor" and i + 1 < len(turns) and turns[i + 1][0] == "patient":
                pairs.append({
                    "dialogue_id": di,
                    "turn_idx": i,
                    "prev_patient_utt": prev_patient,  # 医生这句之前的患者发言（用于 follow_up）
                    "doctor_utt": text,
                    "patient_utt": turns[i + 1][1],
                })
            if sp == "patient":
                prev_patient = text
    return pairs


def parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        # 校验必需字段
        required = {"is_question", "openness", "leadingness", "multi_barreled",
                    "follow_up", "definiteness", "completeness", "relevance", "info_units"}
        if required.issubset(d.keys()):
            return d
    except json.JSONDecodeError:
        pass
    return None


def load_done(out_path: Path) -> set[tuple]:
    """读取已标注输出，返回已完成的 (dialogue_id, turn_idx) 集合，用于断点续跑。"""
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done.add((d["dialogue_id"], d["turn_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def annotate(pair: dict, role: str, retries: int = 2) -> dict:
    prompt = PROMPT.format(
        prev=pair.get("prev_patient_utt") or "",
        doctor=pair["doctor_utt"],
        patient=pair["patient_utt"],
    )
    for attempt in range(retries + 1):
        temp = 0.0 if attempt == 0 else 0.3
        try:
            raw = chat(role, [{"role": "user", "content": prompt}],
                       temperature=temp, max_tokens=512)
            d = parse_json(raw)
            if d:
                return d
        except Exception as e:
            print(f"  [warn] {e}")
        time.sleep(1)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["annotator", "reviewer"])
    ap.add_argument("--limit", type=int, default=None, help="只标前 N 条（配合 --seed 抽样）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=None, help="分片编号（0-based，配合 --nshards）")
    ap.add_argument("--nshards", type=int, default=1, help="分片总数")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dialogues = json.load(open(PARSED, encoding="utf-8"))
    pairs = extract_pairs(dialogues)
    q_pairs = [p for p in pairs if is_question_rule(p["doctor_utt"])]
    print(f"提取 {len(pairs)} 个 (doctor, patient) 对，规则筛出提问 {len(q_pairs)} 条")

    if args.limit:
        import random
        rng = random.Random(args.seed)
        q_pairs = rng.sample(q_pairs, min(args.limit, len(q_pairs)))
        print(f"抽样 {len(q_pairs)} 条 (seed={args.seed})")

    if args.shard is not None and args.nshards > 1:
        chunk = (len(q_pairs) + args.nshards - 1) // args.nshards
        start = args.shard * chunk
        end = min(start + chunk, len(q_pairs))
        q_pairs = q_pairs[start:end]
        print(f"分片 {args.shard}/{args.nshards}: [{start}:{end}) 共 {len(q_pairs)} 条")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：跳过已标注的 pair，用追加模式续写
    done = load_done(out)
    todo = [p for p in q_pairs if (p["dialogue_id"], p["turn_idx"]) not in done]
    if done:
        print(f"检测到已有标注 {len(done)} 条，跳过，剩余 {len(todo)} 条待标注")
    else:
        print(f"全新标注，共 {len(todo)} 条")

    n_ok = 0
    n_fail = 0
    t0 = time.time()
    mode = "a" if done else "w"
    with open(out, mode, encoding="utf-8") as f:
        for i, p in enumerate(todo):
            lab = annotate(p, args.role)
            # 补充规则字段（词数，作披露量交叉验证）
            p["word_count"] = len(p["patient_utt"].split())
            p["doctor_word_count"] = len(p["doctor_utt"].split())
            if lab:
                row = {**p, **lab}
                n_ok += 1
            else:
                row = {**p, "label_error": 1}
                n_fail += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  进度 {i+1}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                      f"{el:.0f}s  ({el/(i+1)*1000:.0f}ms/条)")

    print(f"\n完成: {n_ok} ok, {n_fail} fail -> {out}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
