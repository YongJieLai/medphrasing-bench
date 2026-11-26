"""患者模拟器（英文版）。

核心设定：患者掌握完整 ground-truth（AIU 全集），回答医生提问时——
  开放式问题 → 自然展开、主动补充相关但未被直接问到的信息（"溢出"）；
  封闭式问题 → 简短作答，只答所问。

关键实验纪律：患者 system prompt 在所有条件下【完全一致】，
变化的只有医生问题的措辞。因此任何"溢出量"差异都可归因于措辞本身。
"""
from __future__ import annotations

from .llm import chat
from .cases import Case

_PATIENT_SYSTEM = """You are a patient at a doctor's visit. Below is everything you know about your own condition.

【Chief complaint】
{chief_complaint}

【Your complete medical information】（these are ALL the facts you know）
{aiu_list}

You must role-play a real patient answering the doctor's questions. Follow these rules strictly:
1. Be honest: only state facts that appear in "Your complete medical information" above. Never invent or exaggerate.
2. Match your answer to how the question is asked (MOST IMPORTANT):
   - If the doctor asks an OPEN-ENDED question (e.g. "Can you describe...", "Tell me more...", "Anything else...", "How does it feel..."), answer like a real patient and elaborate naturally — besides directly answering, also volunteer related information about your condition that the doctor did NOT specifically ask about.
   - If the doctor asks a CLOSED question (answerable with yes/no or a single number/short phrase), answer briefly, only what was asked, without elaborating.
3. Do NOT state the medical diagnosis name — ordinary patients usually do not know their exact diagnosis.
4. Answer in first person, natural conversational English, like chatting."""


class Patient:
    def __init__(self, case: Case):
        self.case = case
        aiu_list = "\n".join(f"- {a}" for a in case.aius)
        self.system = _PATIENT_SYSTEM.format(
            chief_complaint=case.chief_complaint,
            aiu_list=aiu_list,
        )

    def answer(self, question: str, history: list[dict] | None = None,
               temperature: float = 0.2) -> str:
        """对医生的一轮提问作答。"""
        messages = [{"role": "system", "content": self.system}]
        for h in (history or []):
            messages.append(h)
        messages.append({"role": "user", "content": question})
        return chat("patient", messages, temperature=temperature)
