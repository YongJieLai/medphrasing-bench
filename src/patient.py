"""患者模拟器（英文版，M2 改造后）。

D1 合规：患者的披露行为（多说 / 附和 / 漏答）【必须】来自条件概率表采样，
不是 system prompt 里的角色规则。system prompt 只保留中立约束：
诚实、只说自己已知的事实、不报诊断名。

回答分两步：
  1. 行为采样：给定医生问句的措辞标注（openness / leadingness / multi_barreled），
     从 ConditionalSimulator 采样出本轮披露行为规格（relevance / definiteness / completeness）。
  2. 渲染：LLM 把采样的行为规格翻译成自然语言回答。

M2 只交付「行为采样 + 渲染接口」；端到端（问句文本→自动标措辞→采样→渲染）
在 M3 自由问诊 harness 里完成（需要措辞标注器 + DDxPlus 真值病例）。
"""
from __future__ import annotations

from .llm import chat
from .cases import Case
from .conditional_simulator import ConditionalSimulator

# system prompt：仅中立约束，不再写任何「措辞→行为」规则（D1）
_PATIENT_SYSTEM = """You are a patient at a doctor's visit. Below is everything you know about your own condition.

【Chief complaint】
{chief_complaint}

【Your complete medical information】（these are ALL the facts you know）
{aiu_list}

Ground rules:
1. Be honest: only state facts that appear in "Your complete medical information". Never invent or exaggerate.
2. Do NOT state the medical diagnosis name — ordinary patients usually do not know their exact diagnosis.
3. Answer in first person, natural conversational English, like chatting."""

# 采样行为规格 -> 本轮渲染指令（机制来自采样，指令只是渲染执行）
_PROFILE_INSTRUCTION = {
    ("spillover",): "besides directly answering, also volunteer ONE related fact you know that the doctor did NOT ask about.",
    ("on-topic",): "answer only what was asked, without volunteering extra facts.",
    ("off-topic",): "answer briefly; you may drift slightly from the exact question.",
    ("exact",): "be specific and concrete (give the exact fact you know).",
    ("vague",): "sound a bit uncertain (use hedges like \"kind of\" / \"not sure\").",
    ("agree",): "just confirm what the doctor suggested (e.g. yes/right), adding no new information.",
}


def _render_instruction(behavior: dict) -> str:
    parts = []
    for key in ("relevance", "definiteness"):
        parts.append(_PROFILE_INSTRUCTION[(behavior[key],)])
    if behavior.get("completeness") == 0:
        parts.append("if the question has several sub-parts, answer only the main one and leave one sub-part out.")
    return " ".join(parts)


class Patient:
    def __init__(self, case: Case, simulator: ConditionalSimulator | None = None):
        self.case = case
        self.simulator = simulator
        aiu_list = "\n".join(f"- {a}" for a in case.aius)
        self.system = _PATIENT_SYSTEM.format(
            chief_complaint=case.chief_complaint,
            aiu_list=aiu_list,
        )

    def answer(self, question: str, phrasing: dict | None = None,
               history: list[dict] | None = None, temperature: float = 0.2,
               seed: int | None = None) -> str:
        """对医生的一轮提问作答。

        phrasing: 该问句的措辞标注 {"openness","leadingness","multi_barreled"}。
        若提供 phrasing 且已装载 simulator，则先从条件概率表采样行为规格，
        再让 LLM 按采样规格渲染（机制来自采样，非 prompt 规则，D1）。
        若不提供，则退化为中立回答（仅 M2 前自检用）。
        """
        import numpy as np
        rng = np.random.default_rng(seed)

        profile_line = ""
        if phrasing is not None and self.simulator is not None:
            behavior = self.simulator.sample_behavior(
                phrasing.get("openness", "closed"),
                phrasing.get("leadingness", "neutral"),
                phrasing.get("multi_barreled", "single"),
                rng,
            )
            profile_line = ("\nThis turn's answer profile (sampled): "
                            + _render_instruction(behavior))

        messages = [{"role": "system", "content": self.system + profile_line}]
        for h in (history or []):
            messages.append(h)
        messages.append({"role": "user", "content": question})
        return chat("patient", messages, temperature=temperature)
