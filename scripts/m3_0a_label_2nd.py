"""M3-0a 第二标注源：deepseek-chat 独立标注评测集四维。

用途：与 M0 标注（qwen-flash）对比，找分歧项 → 执行侧仲裁生成真值。
四维定义与 M0 完全一致（只保留医生侧），差异纯粹来自模型，非定义。

产出 outputs/m3_0a/label_2nd.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 医生侧四维定义，与 annotate_mtsdialog.py 的 PROMPT 保持一致（定义一致是可比的前提）
PROMPT = """You are labeling a DOCTOR'S QUESTION from a doctor-patient dialogue on four dimensions, for a research study on how question PHRASING affects patient disclosure.

PREVIOUS PATIENT UTTERANCE (context only; used to judge follow_up; empty if none):
"{prev}"

DOCTOR'S UTTERANCE:
"{doctor}"

Label the doctor's utterance on the four dimensions below. Output STRICT JSON only (no extra text).

- openness: classify by how much the question CONSTRAINS the answer.
  * "closed": answerable with yes/no or a choice. Starts with an auxiliary verb (Do/Does/Did/Is/Are/Have/Has/Can/Could/Would/Should) OR "Any..." OR offers "A or B?".
  * "semi": a wh-question expecting ONE concrete fact (a number, name, time, or list).
  * "open": invites a free narrative/description with no single expected answer.
- leadingness: "neutral" (no answer implied) | "mild" (suggests a possible answer but patient can freely deny) | "strong" (pushes the patient to confirm, e.g. tag questions "...correct?" / "...right?" / "...isn't it?").
- multi_barreled: "single" (asks for ONE piece of info) | "multi" (asks for >=2 distinct pieces in one utterance).
- follow_up: how this question relates to the PREVIOUS PATIENT UTTERANCE — "surface" (repeats/rephrases a previous question) | "deepen" (drills into what the patient just said to make it more specific) | "new" (new topic). If the previous patient utterance is empty, use "n/a".

Output STRICT JSON:
{{"openness":"..","leadingness":"..","multi_barreled":"..","follow_up":".."}}"""

VALID = {
    "openness": {"open", "semi", "closed"},
    "leadingness": {"neutral", "mild", "strong"},
    "multi_barreled": {"single", "multi"},
    "follow_up": {"surface", "deepen", "new", "n/a"},
}


def parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        required = {"openness", "leadingness", "multi_barreled", "follow_up"}
        if not required.issubset(d.keys()):
            return None
        # 非法取值校验，防模型输出越界标签
        for k in required:
            v = str(d[k]).strip()
            if v not in VALID[k]:
                return None
            d[k] = v
        return d
    except json.JSONDecodeError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set", help="outputs/m3_0a/eval_set.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1")
    recs = [json.loads(l) for l in open(args.eval_set, encoding="utf-8")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_ok = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(recs):
            prompt = PROMPT.format(prev=r.get("prev_patient_utt") or "",
                                   doctor=r["doctor_utt"])
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
                except Exception as e:
                    print(f"  [warn] {e}")
                time.sleep(1)
            row = {"dialogue_id": r["dialogue_id"], "turn_idx": r["turn_idx"],
                   "doctor_utt": r["doctor_utt"], "label": lab}
            if lab:
                n_ok += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  进度 {i+1}/{len(recs)} ok={n_ok} {el:.0f}s "
                      f"({el/(i+1)*1000:.0f}ms/条)")

    print(f"完成: {n_ok}/{len(recs)} 耗时 {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
