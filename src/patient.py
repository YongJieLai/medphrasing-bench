"""患者模拟器。

核心设定：患者掌握完整 ground-truth（AIU 全集），回答医生提问时——
  开放式问题 → 自然展开、主动补充相关但未被直接问到的信息（"溢出"）；
  封闭式问题 → 简短作答，只答所问。

关键实验纪律：患者 system prompt 在所有条件下【完全一致】，
变化的只有医生问题的措辞。因此任何"溢出量"差异都可归因于措辞本身。
"""
from __future__ import annotations

from .llm import chat
from .cases import Case

_PATIENT_SYSTEM = """你是正在就诊的患者。下面是你掌握的、关于自己病情的全部真实信息。

【主诉】
{chief_complaint}

【你的完整病情信息】（按条列出，这就是你全部已知的事实）
{aiu_list}

你要扮演一位真实患者来回答医生的提问，严格遵守：
1. 诚实：只陈述上面"完整病情信息"里真实存在的内容，绝不编造、不夸大。
2. 跟随提问方式作答（最重要）：
   - 当医生用【开放式】问题询问（例如"详细说说""还有别的吗""感觉怎么样""描述一下"这类邀请你展开的问题），你要像真实患者一样自然地展开叙述——除了直接回答，还要主动说出你想到的、相关的、医生没直接问到的病情信息。
   - 当医生用【封闭式】问题询问（能用"是/否"或单个数字、短词回答的问题），你就简短作答，只回答被问到的内容，不额外展开。
3. 不要说出疾病的专业诊断名称——普通患者通常不知道自己的确诊名称。
4. 用第一人称、口语化的中文自然回答，像聊天一样。"""


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
        """对医生的一轮提问作答。history 为之前的对话（可选）。"""
        messages = [{"role": "system", "content": self.system}]
        for h in (history or []):
            messages.append(h)
        messages.append({"role": "user", "content": question})
        return chat("patient", messages, temperature=temperature)
