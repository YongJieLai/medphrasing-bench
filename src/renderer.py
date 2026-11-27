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
R0_REPLIES_EN = ("Nothing else.", "That's all.", "I can't think of anything else.", "That's everything.")

# 附和固定应答（§2.2 契约）：点头称是，不披露新事实
AGREE_REPLIES = ("嗯，对", "对，是这样", "嗯，是的", "对，没错")
AGREE_REPLIES_EN = ("Yes, that's right.", "Right, correct.", "Yes.", "Correct.")

# 无新事实的空回应（非 agree、非 r0，但标签 info_units=0）
NO_CONTENT_REPLIES = ("嗯，就这些情况", "哦，也没什么要补充的", "这个我也不太清楚")
NO_CONTENT_REPLIES_EN = ("Nothing more to add.", "I don't have anything else to mention.", "I'm not sure about that.")

DEFINITENESS_HINT = {
    "exact": "语气明确、肯定，直接陈述。",
    "vague": "语气含糊、不确定，可用「可能」「大概」「有点」「说不好」这类词。注意：含糊只用于肯定/可能的事实；否定的事实（如「没有过敏史」）必须说成明确的「没有」，不得含糊成「大概没有」。",
}

DEFINITENESS_HINT_EN = {
    "exact": "State it clearly and definitely.",
    "vague": "Hedge with uncertainty (\"maybe\", \"kind of\", \"I'm not sure\"). IMPORTANT: hedging only applies to affirmed/possible facts; negated facts (e.g. \"no allergies\") must be stated as a definite \"no\", never \"maybe no\".",
}

RELEVANCE_HINT = {
    "on-topic": "直接回答医生刚才问的问题。",
    "spillover": "在回答医生问题之外，顺带主动提到上面的事实（即使医生没有直接问），像补充说明。",
    "off-topic": "说这些话时，与医生问题的关联较弱，像有点答非所问。",
}

RELEVANCE_HINT_EN = {
    "on-topic": "Directly answer what the doctor just asked.",
    "spillover": "Beyond answering the question, voluntarily add the facts above (even if not directly asked), like a side note.",
    "off-topic": "Say these with weak connection to the doctor's question, slightly off-topic.",
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

【硬规则——否定事实】
事实清单里凡标注「否定地说 / 明确否定」的事实（如「no fever」「no hypertension」），
是患者【明确否认】的事实，必须说成斩钉截铁的「没有 / 不是」，例如「没有发烧」「没有高血压」。
即使「说话方式」要求语气含糊，也【绝不】把否定事实含糊成「大概没有 / 说不好有没有 / 可能没有」。
含糊语气只用于肯定或可能的事实，不用于否定事实。

只输出患者的一句话（中文，第一人称，自然简短）。务必只说出【你本轮必须说的事实】里列出的内容，不要补充清单外的任何事实（包括「其他也没有」「别的都正常」这类额外收尾）。不要输出任何解释、标签、引号或格式。"""

RENDER_SYSTEM_EN = """You are simulating the patient in a doctor-patient consultation. Say ONE natural, short English sentence in first person, following the requirements below.

【What the doctor just asked】
\"\"\"
{doctor}
\"\"\"

【Facts you MUST state this turn】State EVERY one of the facts below truthfully — do not omit, add, or change their meaning. The facts are listed in English; express them naturally:
{facts}

【How to say it】
- {definiteness_hint}
- {relevance_hint}
- {completeness_hint}

【HARD RULE — negated facts】
Facts marked "denied" (e.g. "no fever", "no hypertension") are things the patient EXPLICITLY denies. State them as a firm "no / not", e.g. "I don't have a fever", "no high blood pressure". Even if the "how to say it" asks for hedging, NEVER turn a negated fact into "maybe no / not sure". Hedging only applies to affirmed or possible facts, never to negated ones.

Output ONLY the patient's single sentence (English, first person, natural and short). State ONLY the facts listed in 【Facts you MUST state this turn】; do not add any fact outside the list (including "nothing else" / "everything else is normal" style closers). Do not output any explanation, label, quotes, or formatting."""


def _facts_block(facts: list[dict], definiteness: str = "exact",
                 language: str = "zh") -> str:
    """把事实清单渲染成 prompt 里的编号列表，每条附「怎么说」的语气前缀。

    M2c-b（§7.7 第 3 件）修 vague：`definiteness=vague` 只作用于
    `fidelity ∈ {definite, fuzzy}` 的事实；`fidelity=denied` 的事实始终明确否定，
    不得被含糊成「大概没有」（那是语义错误）。
    """
    en = language == "en"
    if not facts:
        return "(no new facts this turn)" if en else "（本轮无新增事实）"
    vague = definiteness == "vague"
    lines = []
    for i, a in enumerate(facts, 1):
        fidelity = a.get("fidelity", "definite")
        if fidelity == "denied":
            prefix = ("(state as a firm NO: must say a definite \"no/not\", "
                      "NEVER \"maybe/probably/not sure\")" if en
                      else "（明确否定地说：必须说成明确的「没有/不是」，严禁「可能/大概/说不好」）")
        elif vague:
            prefix = "(say this in a hedged, uncertain tone)" if en else "（语气含糊地说）"
        else:
            prefix = ("(say this in a hedged, uncertain tone)" if fidelity == "fuzzy" and en
                      else "（语气含糊地说）" if fidelity == "fuzzy" else "")
        lines.append(f"{i}. {a['text']} {prefix}".rstrip())
    return "\n".join(lines)


def render_turn(doctor_utt: str, prev_patient: str | None, instruction: dict,
                rng: random.Random | None = None, language: str = "zh") -> str:
    """把一条渲染指令转译成一句患者口语。

    instruction 来自 aiu_selector.select_aius，含 special/facts/must_say/omit/
    definiteness/completeness/relevance。
    language: "zh"（默认，M2c/M2d 口径）或 "en"（M3-0 预检，与英文问诊一致）。
    """
    en = language == "en"
    special = instruction.get("special")

    if special == "r0":
        pool = R0_REPLIES_EN if en else R0_REPLIES
        return (rng or random).choice(pool)
    if special == "agree":
        pool = AGREE_REPLIES_EN if en else AGREE_REPLIES
        return (rng or random).choice(pool)
    if special == "no_content":
        pool = NO_CONTENT_REPLIES_EN if en else NO_CONTENT_REPLIES
        return (rng or random).choice(pool)

    must_say = instruction.get("must_say", [])
    omit = instruction.get("omit", [])
    completeness = instruction.get("completeness", 1)
    relevance = instruction.get("relevance", "on-topic")
    definiteness = instruction.get("definiteness", "exact")

    if completeness == 1:
        completeness_hint = ("State all the facts above completely, don't omit any."
                             if en else "把上面所有事实都说完整，不要漏。")
    else:
        omit_txt = ("; ".join(a["text"] for a in omit) if omit
                    else ("(omit the trailing content)" if en else "（略去末尾内容）"))
        completeness_hint = (
            f"Deliberately OMIT the following content and do NOT mention it "
            f"(simulating an incomplete answer): {omit_txt}. State the rest as usual."
            if en else
            f"故意漏掉下面这条内容、不要提它（模拟「没答全」）：{omit_txt}。"
            "其余事实照常说出。"
        )

    system = RENDER_SYSTEM_EN if en else RENDER_SYSTEM
    definiteness_hint_map = DEFINITENESS_HINT_EN if en else DEFINITENESS_HINT
    relevance_hint_map = RELEVANCE_HINT_EN if en else RELEVANCE_HINT
    prompt = system.format(
        doctor=doctor_utt,
        facts=_facts_block(must_say, definiteness=definiteness, language=language),
        definiteness_hint=definiteness_hint_map[definiteness],
        relevance_hint=relevance_hint_map.get(relevance, relevance_hint_map["on-topic"]),
        completeness_hint=completeness_hint,
    )

    # 上一轮患者回复作为衔接上下文（自然度辅助，不影响事实清单）
    if prev_patient:
        if en:
            prompt += (f"\n\n【What you said last turn】{prev_patient}\n"
                       "Note: do not repeat facts already stated; this turn only state the new facts above.")
        else:
            prompt += f"\n\n【你上一轮说过】{prev_patient}\n注意：不要重复已经说过的事实，本轮只说上面列出的新事实。"

    raw = chat("patient", [{"role": "user", "content": prompt}],
               temperature=0.3, max_tokens=256)
    return raw.strip()
