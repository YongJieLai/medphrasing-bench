"""M2c 内容选择器：把「行为标签」翻译成「本轮具体要说哪几件事」。

职责边界（对应 §7.6 第 1 件「渲染层只负责怎么说，不负责说什么」）：
本模块负责【说什么】——从真值集 T_d \\ D_d 中确定性选出本轮该说出口的 AIU 文本。
渲染层（renderer.py）只负责【怎么说】——把这里给出的 AIU 清单转译成一句人话。

M2c-b（§7.7 第 2 件）起，选法支持「按医生问题相关性选」：
  - 传入 relevance_scores（{真值集索引: 0-1 分数}）时：
      on-topic        取相关性【最高】的 min(iu, k) 个（回答医生问的）
      spillover/off-topic 取相关性【最低】的 min(iu, k) 个（说出医生没问的）
  - 不传 relevance_scores 时，回退为「顺序选」（旧行为，作修前对照）。
k = info_units（不改标签层给出的数量）。
"""
from __future__ import annotations

from .disclosure_ledger import AiuLedger

# r0 档特判（§7.6 第 4 件）：剩余量为 0 时，只出「没有了」，不得出新事实。
# 理由：标签层 r0 档存在定性错误（剩余 0 却说 2.56 个事实，§7.2.5 记录二），
# 渲染层是最后一道闸。
R0_REPLIES = ("没有了", "就这些了", "暂时想不起来了", "别的没有了")


def select_aius(ledger: AiuLedger, behavior: dict,
                relevance_scores: dict[int, float] | None = None) -> dict:
    """根据行为标签选出本轮要说的事实，返回渲染指令。

    behavior: conditional_simulator.sample_behavior 的返回，
              含 relevance / definiteness / completeness / info_units。
    relevance_scores: {真值集索引: 0-1 相关性分数}（M2c-b §7.7）。缺省时顺序选。

    返回字段：
      special: None | "r0" | "agree"   —— 特殊回合（无事实清单）
      facts:   完整计划事实（completeness=1 时全部说出）
      must_say: 实际必须说的（completeness=0 时 = facts 去掉 omit）
      omit:    completeness=0 时故意漏掉的事实
      indices: 计划披露的真值集索引（调用方据此入账）
      definiteness / completeness / relevance: 透传标签，供渲染层语气/方式控制
    """
    remaining = ledger.remaining_aius()  # [(idx, aiu), ...] 按真值集顺序
    definiteness = behavior["definiteness"]
    completeness = behavior["completeness"]
    relevance = behavior["relevance"]
    info_units = int(behavior.get("info_units") or 0)

    # --- r0 档特判：剩余 0 就只许说「没有了」 ---
    if len(remaining) == 0:
        return {
            "special": "r0",
            "facts": [], "must_say": [], "omit": [], "indices": [],
            "definiteness": definiteness, "completeness": completeness,
            "relevance": relevance, "info_units": 0,
        }

    # --- 附和：点头称是，不主动披露新事实（§2.2 契约） ---
    if definiteness == "agree":
        return {
            "special": "agree",
            "facts": [], "must_say": [], "omit": [], "indices": [],
            "definiteness": definiteness, "completeness": completeness,
            "relevance": relevance, "info_units": 0,
        }

    # --- 正常披露：选内容 ---
    n = min(info_units, len(remaining))
    if n == 0:
        # 非 agree 且 remaining>0 但 info_units=0（真实约占 1.4%）：无新事实的回应
        return {
            "special": "no_content",
            "facts": [], "must_say": [], "omit": [], "indices": [],
            "definiteness": definiteness, "completeness": completeness,
            "relevance": relevance, "info_units": 0,
        }

    if relevance_scores is not None:
        # M2c-b：按相关性选。ascending = 最不相关在前（spillover/off-topic 取这些）
        scored = sorted(remaining, key=lambda p: relevance_scores.get(p[0], 0.0))
        if relevance == "on-topic":
            picked = list(reversed(scored))[:n]   # 最高分在前
        else:
            picked = scored[:n]                   # spillover/off-topic 取最低分
    else:
        picked = remaining[:n]                    # 顺序选（修前对照）

    facts = [aiu for _, aiu in picked]
    indices = [idx for idx, _ in picked]

    if completeness == 0 and n >= 1:
        # 漏答：故意漏掉最后一条（§7.6 第 1 件「completeness=0 就故意漏掉部分」）
        must_say = facts[:-1]
        omit = facts[-1:]
        # 入账只按「真正说出口的」must_say 对应的索引
        indices = indices[:-1]
    else:
        must_say = facts
        omit = []

    return {
        "special": None,
        "facts": facts,
        "must_say": must_say,
        "omit": omit,
        "indices": indices,
        "definiteness": definiteness,
        "completeness": completeness,
        "relevance": relevance,
        "info_units": len(must_say),
    }
