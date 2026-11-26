"""信息提取与溢出指标（英文版）。

用 extractor（LLM）从患者回答中抽取"被披露的 AIU"，再与 ground-truth AIU 匹配，
从而区分：意图命中（回答被问到的） vs 溢出（顺带披露的其它 AIU）。
"""
from __future__ import annotations

import json
import re

from .llm import chat
from .cases import Case

_EXTRACT_SYSTEM = """You are an information extractor. Given a patient's answer and a list of "atomic information units" (AIUs), determine which AIUs are disclosed in the answer — semantic mention counts, paraphrasing is allowed.
Output ONLY one JSON object of the form: {"disclosed": ["AIU text 1", "AIU text 2"]}
"disclosed" may only contain AIUs that (a) appear verbatim in the given list and (b) are actually disclosed. Do not add anything outside the list, and do not explain."""


def extract_disclosed(answer: str, case: Case) -> set[str]:
    """从回答中提取被披露的 AIU 原文集合。"""
    if not answer.strip():
        return set()
    aiu_list = "\n".join(f"- {a}" for a in case.aius)
    user = f"【Patient answer】\n{answer}\n\n【AIU list】\n{aiu_list}"
    raw = chat("extractor", [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": user},
    ], temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return set()
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return set()
    disclosed = data.get("disclosed", [])
    valid = set(case.aius)
    return {d for d in disclosed if d in valid}


def measure(answer: str, case: Case, targets: list[str]) -> dict:
    """测量一轮回答的披露情况。"""
    disclosed = extract_disclosed(answer, case)
    tset = set(targets)
    target_hits = disclosed & tset
    spillover = disclosed - tset
    return {
        "disclosed": sorted(disclosed),
        "target_hits": sorted(target_hits),
        "spillover": sorted(spillover),
        "spillover_count": len(spillover),
        "target_hit_count": len(target_hits),
    }
