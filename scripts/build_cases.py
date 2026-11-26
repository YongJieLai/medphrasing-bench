"""从 DDxPlus 构建 50 例病例库（英文）。

流程：
1. 读 release_evidences.json（223 证据定义）+ validate.csv
2. 随机抽 N 例（固定 seed）
3. 每例用 LLM（qwen-flash）：
   a. 把证据编码 → 自然语言 AIU 列表 + 一句主诉
   b. 为选中的 6 个信息点生成 open/closed 措辞 + targets
4. 输出 JSON（data/ddxplus_50.json），格式与 src.cases.Case 对齐

用法：
  python scripts/build_cases.py --n 50 --seed 42 --out data/ddxplus_50.json
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.llm import chat

DDX = Path("/root/.codebuddy/artifact/ddxplus")
REPO = Path(__file__).resolve().parent.parent

N_INFO_POINTS = 6


def load_evidences() -> dict:
    return json.load(open(DDX / "release_evidences.json", encoding="utf-8"))


def parse_code(code: str) -> tuple[str, str | None]:
    if "_@_" in code:
        name, val = code.split("_@_")
        return name, val
    return code, None


def code_description(code: str, ev: dict) -> str:
    """把单个证据编码转成给 LLM 看的描述行。"""
    name, val = parse_code(code)
    e = ev[name]
    q = e["question_en"]
    if val is None:
        return f"[binary] {code}: \"{q}\""
    # 有值
    if e.get("value_meaning") and val in e["value_meaning"]:
        meaning = e["value_meaning"][val].get("en", val)
    else:
        meaning = str(val)
    return f"[value: {meaning}] {code}: \"{q}\""


def evidence_to_aiu(evidences: list[str], initial: str, ev: dict) -> tuple[list[str], str]:
    """LLM 转换：证据编码 → AIU 列表 + 主诉。"""
    lines = [f"{i+1}. {code_description(c, ev)}" for i, c in enumerate(evidences)]
    # initial evidence 描述
    initial_desc = code_description(initial, ev)
    prompt = f"""You are converting a structured patient record into natural-language atomic information units (AIUs).

Below is a list of the patient's reported findings (symptoms/antecedents). Convert EACH finding into ONE concise AIU — a single verifiable fact, in English, present tense, patient perspective.

Rules:
- [binary] finding = the patient HAS it. E.g. "Do you have a fever?" -> "Has fever".
- [value: X] finding = include that specific value. E.g. "How intense is the pain?" value "5" -> "Pain intensity 5"; "Characterize your pain:" value "burning" -> "Pain is burning".
- If a value means absence/negative (e.g. "N", "nowhere", "NA", "no"), phrase it as the negative finding (e.g. "No recent travel abroad").
- Keep each AIU short and faithful; do NOT add information not given.

Findings:
{chr(10).join(lines)}

Also write ONE natural sentence as the patient's opening chief complaint, based on this initial finding:
{initial_desc}

Output strict JSON: {{"aius": ["...", "..."], "chief_complaint": "..."}}
AIU count must equal the number of findings. No extra text."""
    for attempt in range(3):
        temp = 0.0 if attempt == 0 else 0.3
        raw = chat("builder", [{"role": "user", "content": prompt}], temperature=temp, max_tokens=2048)
        aius, chief = _parse_aiu(raw, len(evidences))
        if aius:
            return aius, chief
    return [], ""


def _parse_aiu(raw: str, n: int) -> tuple[list[str], str]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            aius = d.get("aius", [])
            cc = d.get("chief_complaint", "")
            if len(aius) == n and cc:
                return aius, cc
        except json.JSONDecodeError:
            pass
    return [], ""


def generate_info_points(aius: list[str], ev_codes: list[str], ev: dict) -> dict:
    """为病例生成 N_INFO_POINTS 个信息点（open/closed 措辞 + targets）。"""
    # 选信息点：优先非 antecedent 的 symptom
    symptom_idx = [i for i, c in enumerate(ev_codes) if not ev[parse_code(c)[0]]["is_antecedent"]]
    ante_idx = [i for i, c in enumerate(ev_codes) if ev[parse_code(c)[0]]["is_antecedent"]]
    chosen = (symptom_idx + ante_idx)[:N_INFO_POINTS]

    items = []
    for i in chosen:
        code = ev_codes[i]
        aiu = aius[i]
        q = ev[parse_code(code)[0]]["question_en"]
        items.append(f"{len(items)+1}. AIU: \"{aiu}\"  |  original clinical question: \"{q}\"")

    prompt = f"""For each finding below, write TWO questions a doctor could ask to obtain that finding:
- "closed": a precise question answerable with "yes/no" or a single short fact.
- "open": an open-ended question that invites the patient to describe or elaborate (e.g. "Could you tell me more about...", "Can you describe...").

Both questions target the SAME finding (same information intent). Give each a short label.

Findings:
{chr(10).join(items)}

Output strict JSON: {{"points": [{{"label": "...", "open": "...", "closed": "...", "targets": ["<the AIU exactly as given>"]}}]}}
One entry per finding. No extra text."""
    raw = chat("builder", [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        info_points = {}
        for p in d.get("points", []):
            label = p.get("label", f"p{len(info_points)+1}")
            info_points[label] = {
                "open": p.get("open", ""),
                "closed": p.get("closed", ""),
                "targets": p.get("targets", []),
            }
        return info_points
    except json.JSONDecodeError:
        return {}


def build(n: int, seed: int, out: str):
    ev = load_evidences()
    df = pd.read_csv(DDX / "validate.csv")
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    cases = []
    failed = 0
    for idx, row in df.iterrows():
        evidences = ast.literal_eval(row["EVIDENCES"])
        initial = row["INITIAL_EVIDENCE"]
        diagnosis = row["PATHOLOGY"]
        age = row["AGE"]
        sex = row["SEX"]

        aius, chief = evidence_to_aiu(evidences, initial, ev)
        if not aius:
            failed += 1
            print(f"[{idx}] AIU 转换失败，跳过")
            continue
        info_points = generate_info_points(aius, evidences, ev)
        if len(info_points) < 3:
            failed += 1
            print(f"[{idx}] 信息点生成失败，跳过")
            continue

        cases.append({
            "id": f"ddx-{idx:03d}",
            "age": int(age),
            "sex": sex,
            "chief_complaint": chief,
            "diagnosis": diagnosis,
            "aius": aius,
            "info_points": info_points,
        })
        print(f"[{idx}] {diagnosis}: {len(aius)} AIU, {len(info_points)} 信息点")

    out_path = REPO / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"\n构建完成：{len(cases)} 例（失败 {failed}），已写入 {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/ddxplus_50.json")
    args = ap.parse_args()
    build(args.n, args.seed, args.out)
