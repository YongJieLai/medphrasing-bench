"""病例数据结构与样例病例。

一个病例 = 主诉 + 原子信息单元（AIU）列表 + 金标准诊断。
AIU（Atomic Information Unit）是"可独立验证的最小事实"，是测量"溢出"的计量单位。

每个 info_point（信息点）标注：
- open   : 开放式问法（引导患者展开）
- closed : 封闭式问法（可被"是/否"或最小信息回答）
- targets: 该问题"意图获取"的 AIU 原文列表（用于区分"被问到"与"溢出"）
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Case:
    id: str
    chief_complaint: str            # 主诉（最初公开信息）
    diagnosis: str                  # 金标准诊断（仅评测用，不给 agent）
    aius: list[str]                 # 全部 AIU（患者 ground-truth 全集）
    info_points: dict[str, dict] = field(default_factory=dict)


# ---- 样例病例（占位，后续替换为真实/更多病例） ----

CASES: list[Case] = [
    Case(
        id="cap-01",
        chief_complaint="我这两天咳嗽、发烧，感觉浑身没力气。",
        diagnosis="社区获得性肺炎（Community-Acquired Pneumonia）",
        aius=[
            "发热，最高体温 39.2°C",
            "咳嗽，咳黄绿色脓痰",
            "右侧胸痛，深呼吸或咳嗽时加重",
            "病程 3 天",
            "无吸烟史",
            "无慢性肺病或心脏病史",
            "无药物过敏",
            "曾与感冒的家人接触",
            "乏力、食欲下降",
            "无咯血",
        ],
        info_points={
            "发热": {
                "open": "您能详细说说发烧的情况吗？",
                "closed": "您发烧到多少度？",
                "targets": ["发热，最高体温 39.2°C"],
            },
            "咳嗽咳痰": {
                "open": "咳嗽的情况是怎么样的？",
                "closed": "您咳嗽有痰吗？",
                "targets": ["咳嗽，咳黄绿色脓痰"],
            },
            "伴随症状": {
                "open": "除了咳嗽发烧，还有别的什么不舒服吗？",
                "closed": "您有没有胸痛？",
                "targets": ["右侧胸痛，深呼吸或咳嗽时加重"],
            },
            "病程": {
                "open": "这些症状是从什么时候开始的？",
                "closed": "您病了几天了？",
                "targets": ["病程 3 天"],
            },
            "既往史": {
                "open": "您以前得过什么病，或者平时身体怎么样？",
                "closed": "您有没有慢性肺病或心脏病？",
                "targets": ["无吸烟史", "无慢性肺病或心脏病史"],
            },
            "接触史": {
                "open": "最近生活上有没有什么特殊情况？",
                "closed": "您最近接触过感冒的人吗？",
                "targets": ["曾与感冒的家人接触"],
            },
        },
    ),
    Case(
        id="migraine-01",
        chief_complaint="我最近总是头疼，一阵一阵的。",
        diagnosis="偏头痛（Migraine）",
        aius=[
            "头痛为搏动性、单侧",
            "发作时伴恶心，偶有呕吐",
            "怕光、怕吵",
            "每次发作持续 4-6 小时",
            "近一个月发作约 5 次",
            "劳累、睡眠不足后易诱发",
            "发作前有时眼前出现闪光/暗点",
            "无发热",
            "无头部外伤史",
            "母亲有类似头痛史",
        ],
        info_points={
            "头痛性质": {
                "open": "您能描述一下头疼是什么样的吗？",
                "closed": "头疼是跳着疼吗？",
                "targets": ["头痛为搏动性、单侧"],
            },
            "伴随症状": {
                "open": "头疼发作的时候，还有别的不舒服吗？",
                "closed": "您头疼的时候恶心吗？",
                "targets": ["发作时伴恶心，偶有呕吐"],
            },
            "诱因": {
                "open": "您有没有注意到，什么情况下容易头疼？",
                "closed": "您头疼和睡眠不足有关吗？",
                "targets": ["劳累、睡眠不足后易诱发"],
            },
            "发作频率": {
                "open": "头疼多久发作一次？",
                "closed": "这个月头疼发作了几次？",
                "targets": ["近一个月发作约 5 次"],
            },
            "先兆": {
                "open": "头疼之前，您身体有没有什么预兆？",
                "closed": "头疼前眼前会不会有闪光？",
                "targets": ["发作前有时眼前出现闪光/暗点"],
            },
            "家族史": {
                "open": "家里其他人有没有类似的情况？",
                "closed": "您母亲有没有头疼的毛病？",
                "targets": ["母亲有类似头痛史"],
            },
        },
    ),
]
