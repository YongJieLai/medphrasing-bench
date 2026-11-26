"""医生 agent（英文版，阶段 B 用）。

两种策略通过 system prompt 注入，仅改变"提问措辞风格"。
"""
from __future__ import annotations

from .llm import chat
from .cases import Case

_OPEN_DOCTOR = """You are an internist conducting a text-based medical consultation. You must gather information through questions and finally give a diagnosis.

【Questioning rule】Each turn, ask only ONE open-ended question — use "Can you describe...", "Tell me more...", "Anything else...", "How does it feel..." to invite the patient to elaborate. Do NOT ask yes/no closed questions.
【Diagnosis rule】When you have enough information, start your reply with "Diagnosis:" and give your final diagnosis, then stop asking."""

_CLOSED_DOCTOR = """You are an internist conducting a text-based medical consultation. You must gather information through questions and finally give a diagnosis.

【Questioning rule】Each turn, ask only ONE closed question — answerable with yes/no or a single number/short phrase. Do NOT ask open-ended questions that invite elaboration.
【Diagnosis rule】When you have enough information, start your reply with "Diagnosis:" and give your final diagnosis, then stop asking."""


class Doctor:
    def __init__(self, style: str):
        assert style in ("open", "closed")
        self.style = style
        self.system = _OPEN_DOCTOR if style == "open" else _CLOSED_DOCTOR

    def first_question(self, case: Case) -> str:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"Patient's chief complaint: {case.chief_complaint}\nStart the consultation with your first question."},
        ]
        return chat("doctor", messages, temperature=0.0)

    def next_turn(self, case: Case, history: list[dict]) -> str:
        messages = [{"role": "system", "content": self.system}]
        messages.append({"role": "user", "content": f"Patient's chief complaint: {case.chief_complaint}"})
        messages.extend(history)
        return chat("doctor", messages, temperature=0.0)
