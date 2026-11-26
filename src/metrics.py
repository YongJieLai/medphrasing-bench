"""信息提取与溢出指标。

用 extractor（LLM）从患者回答中抽取"被披露的 AIU"，再与 ground-truth AIU 匹配，
从而区分：意图命中（回答被问到的） vs 溢出（顺带披露的其它 AIU）。
"""
from __future__ import annotations

import json
import re

from .llm import chat
from .cases import Case

_EXTRACT_SYSTEM = """你是信息提取器。给你一段患者的回答，和一份"原子信息单元(AIU)"列表。
请判断列表中哪些 AIU 在这段回答里被披露了——只要语义被提及即算披露，允许同义改写。
只输出一个 JSON 对象，格式：{"disclosed": ["AIU原文1", "AIU原文2"]}
disclosed 里只能包含列表中原样存在、且确实被披露的条目，不要新增列表外的内容，不要解释。"""


def extract_disclosed(answer: str, case: Case) -> set[str]:
    """从回答中提取被披露的 AIU 原文集合。"""
    if not answer.strip():
        return set()
    aiu_list = "\n".join(f"- {a}" for a in case.aius)
    user = f"【患者回答】\n{answer}\n\n【AIU 列表】\n{aiu_list}"
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
    """测量一轮回答的披露情况。

    返回：
      disclosed      : 本轮披露的全部 AIU
      target_hits    : 意图内（被问到的）AIU
      spillover      : 溢出（非意图、顺带披露的）AIU
      spillover_count: 溢出数量
    """
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
