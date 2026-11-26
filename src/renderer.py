"""M2c 渲染层：把「行为标签 + 事实清单」转译成一句中文患者口语。

职责边界（§7.6 第 1 件）：只负责【怎么说】，不负责【说什么】。
【说什么】由 aiu_selector.select_aius 完成，本模块只把给定的事实清单说成人话。

硬约束（违反即判渲染不忠实，见回抽一致性判据）：
- 事实清单必须全部说出，不得多写一个、不得少写一个；
- 不得在 prompt 里写任何「医生问得开放就多说」之类的措辞→披露量推理（D1/F2）。
"""
from __future__ import annotations

import random

from .llm import chat

# r0 档固定应答（§7.6 第 4 件）：剩余 0 只许说「没有了」，零披露、不调 LLM
R0_REPLIES = ("没有了", "就这些了", "暂时想不起来了", "别的没有了")

# 附和固定应答（§2.2 契约）：点头称是，不披露新事实
AGREE_REPLIES = ("嗯，对", "对，是这样", "嗯，是的", "对，没错")

# 无新事实的空回应（非 agree、非 r0，但标签 info_units=0）
NO_CONTENT_REPLIES = ("嗯，就这些情况", "哦，也没什么要补充的", "这个我也不太清楚")

DEFINITENESS_HINT = {
    "exact": "语气明确、肯定，直接陈述。",
    "vague": "语气含糊、不确定，可用「可能」「大概」「有点」「说不好」这类词，不要说得斩钉截铁。",
}

RELEVANCE_HINT = {
    "on-topic": "直接回答医生刚才问的问题。",
    "spillover": "在回答医生问题之外，顺带主动提到上面的事实（即使医生没有直接问），像补充说明。",
    "off-topic": "说这些话时，与医生问题的关联较弱，像有点答非所问。",
}

RENDER_SYSTEM = """你是模拟医患问诊对话中的患者。请按下面的要求，说一句自然、简短的中文患者口语（第一人称）。

【医生刚才问】
\"\"\"
{doctor}
\"\"\"

【你本轮必须说的事实】下面每一条都必须如实说出，不得遗漏、不得添加、不得改变事实含义。事实清单是英文描述，请用中文自然表达：
{facts}

【说话方式】
- {definiteness_hint}
- {relevance_hint}
- {completeness_hint}

只输出患者的一句话（中文，第一人称，自然简短）。务必只说出【你本轮必须说的事实】里列出的内容，不要补充清单外的任何事实（包括「其他也没有」「别的都正常」这类额外收尾）。不要输出任何解释、标签、引号或格式。"""


def _facts_block(facts: list[dict]) -> str:
    if not facts:
        return "（本轮无新增事实）"
    lines = []
    for i, a in enumerate(facts, 1):
        fidelity = a.get("fidelity", "definite")
        prefix = {
            "definite": "",
            "fuzzy": "（语气含糊地说）",
            "denied": "（否定地说）",
        }.get(fidelity, "")
        lines.append(f"{i}. {a['text']} {prefix}".rstrip())
    return "\n".join(lines)


def render_turn(doctor_utt: str, prev_patient: str | None, instruction: dict,
                rng: random.Random | None = None) -> str:
    """把一条渲染指令转译成一句患者口语。

    instruction 来自 aiu_selector.select_aius，含 special/facts/must_say/omit/
    definiteness/completeness/relevance。
    """
    special = instruction.get("special")

    if special == "r0":
        return (rng or random).choice(R0_REPLIES)
    if special == "agree":
        return (rng or random).choice(AGREE_REPLIES)
    if special == "no_content":
        return (rng or random).choice(NO_CONTENT_REPLIES)

    must_say = instruction.get("must_say", [])
    omit = instruction.get("omit", [])
    completeness = instruction.get("completeness", 1)
    relevance = instruction.get("relevance", "on-topic")
    definiteness = instruction.get("definiteness", "exact")

    if completeness == 1:
        completeness_hint = "把上面所有事实都说完整，不要漏。"
    else:
        omit_txt = "；".join(a["text"] for a in omit) if omit else "（略去末尾内容）"
        completeness_hint = (
            f"故意漏掉下面这条内容、不要提它（模拟「没答全」）：{omit_txt}。"
            "其余事实照常说出。"
        )

    prompt = RENDER_SYSTEM.format(
        doctor=doctor_utt,
        facts=_facts_block(must_say),
        definiteness_hint=DEFINITENESS_HINT[definiteness],
        relevance_hint=RELEVANCE_HINT.get(relevance, RELEVANCE_HINT["on-topic"]),
        completeness_hint=completeness_hint,
    )

    # 上一轮患者回复作为衔接上下文（自然度辅助，不影响事实清单）
    if prev_patient:
        prompt += f"\n\n【你上一轮说过】{prev_patient}\n注意：不要重复已经说过的事实，本轮只说上面列出的新事实。"

    raw = chat("patient", [{"role": "user", "content": prompt}],
               temperature=0.3, max_tokens=256)
    return raw.strip()
