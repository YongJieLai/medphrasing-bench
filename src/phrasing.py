"""措辞库：同一信息点 → 开放式 / 封闭式 两种问法。

实验的核心自变量。两种问法的"信息寻求意图"一致，仅开放度不同。
"""
from __future__ import annotations

from .cases import Case


def phrasing_variants(case: Case) -> list[dict]:
    """展开一个病例的所有 (信息点, 措辞) 组合。"""
    out = []
    for label, ph in case.info_points.items():
        out.append({"label": label, "style": "open", "question": ph["open"]})
        out.append({"label": label, "style": "closed", "question": ph["closed"]})
    return out
