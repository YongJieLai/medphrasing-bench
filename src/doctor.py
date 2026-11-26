"""医生 agent（阶段 B：自由问诊端到端实验用）。

两种策略通过 system prompt 注入，仅改变"提问措辞风格"：
  open   = 始终使用开放式提问
  closed = 始终使用封闭式提问
"""
from __future__ import annotations

from .llm import chat
from .cases import Case

_OPEN_DOCTOR = """你是一名内科医生，正在通过文字问诊一位患者。你必须通过提问收集诊断所需信息，最后给出诊断。

【提问纪律】每轮只问一个【开放式】问题——用"详细说说""还有什么""感觉怎么样""描述一下"这类方式引导患者展开叙述。不要用"是/否"就能回答的封闭式问题。
【诊断纪律】当你认为信息足够时，在回复开头以"诊断："给出你的最终诊断并停止提问。"""

_CLOSED_DOCTOR = """你是一名内科医生，正在通过文字问诊一位患者。你必须通过提问收集诊断所需信息，最后给出诊断。

【提问纪律】每轮只问一个【封闭式】问题——能用"是/否"或单个数字、短词回答的精确问题。不要用开放式问题引导患者展开叙述。
【诊断纪律】当你认为信息足够时，在回复开头以"诊断："给出你的最终诊断并停止提问。"""


class Doctor:
    def __init__(self, style: str):
        assert style in ("open", "closed")
        self.style = style
        self.system = _OPEN_DOCTOR if style == "open" else _CLOSED_DOCTOR

    def first_question(self, case: Case) -> str:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"患者主诉：{case.chief_complaint}\n请开始问诊，先问第一个问题。"},
        ]
        return chat("doctor", messages, temperature=0.0)

    def next_turn(self, case: Case, history: list[dict]) -> str:
        """根据完整对话历史，生成下一句（继续提问或以"诊断："开头给出诊断）。"""
        messages = [{"role": "system", "content": self.system}]
        messages.append({"role": "user", "content": f"患者主诉：{case.chief_complaint}"})
        messages.extend(history)
        return chat("doctor", messages, temperature=0.0)
