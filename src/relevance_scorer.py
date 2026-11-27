"""M2c-b 相关性打分：对 (医生问句, 事实 AIU) 打 0-1 相关性分数。

用途（§7.7 第 1/2 件）：内容选择器按该分数选内容——
  on-topic       取相关性【最高】的 min(iu, k) 个事实（回答医生问的）
  spillover/off-topic 取相关性【最低】的 min(iu, k) 个事实（说出医生没问的）

派单要求「不必追求精度，只要单调」：相关事实的分数必须稳定高于不相关事实。
本实现用纯词法（词重叠 + bigram 短语 + 少量医学同义词归一），确定性、零 API
依赖、可复现（D14）。分数落到 JSON 缓存，20 条样例供人工核对。

评分定义（可解释）：
  score = 事实内容词被问句覆盖的比例（unigram），bigram 短语命中作加分项。
  即「这个事实里有几成内容词，医生刚才真的问到了」。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# 停用词：问句与事实两侧同时剔除。注意「no / not / never」也剔除——对相关性而言，
# 「no fever」与「any fever」的关键内容是 fever，否定不改变相关性。
# 同时剔除问句里的搭话词（great/also/as/identify/records 等），避免它们把
# 「identifies as white」之类的事实错误拉高。
STOPWORDS = frozenset("""
    a an the and or of in on at to for with by from your my our their his her
    have has had do does did is are was were be been being
    you i we they he she it me us them this that these those
    what how when where why which who whom whose
    any some ever since recently today now currently just still
    please tell describe explain also really very much many more
    notice noticed concerns issues problem problems anything something
    no not never none without yes yeah yep nope
    as identify identifies identified identifying yourself myself ourselves himself herself
    great good okay ok well sure right let know think record records
    about because if then than too only other others another each every all most
""".split())

# 医学同义词归一（保守：只合并确定同义的，避免引入噪声）。
# 用于把问句与事实里写法不同但同义的词对齐。
SYNONYM_MAP = {
    "medication": "medication", "medicine": "medication", "medications": "medication",
    "medicines": "medication", "meds": "medication", "med": "medication",
    "drug": "medication", "drugs": "medication", "pill": "medication", "pills": "medication",
    "hypertension": "hypertension", "hypertensive": "hypertension",
    "xray": "xray", "x-ray": "xray", "x ray": "xray", "rays": "xray",
    "pain": "pain", "pains": "pain", "ache": "pain", "aches": "pain",
    "fever": "fever", "feverish": "fever", "temperature": "fever",
    "vomit": "vomit", "vomiting": "vomit", "throwing": "vomit",
    "cough": "cough", "coughing": "cough",
    "diarrhea": "diarrhea", "diarrhoea": "diarrhea",
    "nausea": "nausea", "nauseous": "nausea", "nauseated": "nausea",
    "old": "age", "years": "age", "age": "age",
    "race": "race", "ethnicity": "race",
    # 性别值 → sex（「are you male or female」与「identifies as female」对齐）
    "female": "sex", "male": "sex", "woman": "sex", "man": "sex",
    "sex": "sex", "gender": "sex",
}


def _normalize_tokens(text: str) -> list[str]:
    """小写、去标点、去停用词、医学同义词归一，返回内容词序列（保持顺序）。"""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = []
    for t in text.split():
        if t in STOPWORDS:
            continue
        t = SYNONYM_MAP.get(t, t)
        if t:
            toks.append(t)
    return toks


def score_relevance(question: str, fact_text: str) -> float:
    """对单个 (问句, 事实) 打 0-1 相关性分数。

    定义：事实内容词的覆盖比例 + bigram 短语命中加分。
    """
    qt = _normalize_tokens(question)
    ft = _normalize_tokens(fact_text)
    if not ft:
        return 0.0
    qset = set(qt)

    hits = sum(1 for t in ft if t in qset)
    uni = hits / len(ft)

    if len(ft) < 2 or len(qt) < 2:
        return uni  # 单内容词事实：只看覆盖

    qbig = set(zip(qt, qt[1:]))
    fbig = set(zip(ft, ft[1:]))
    if not fbig:
        return uni
    bi = len(fbig & qbig) / len(fbig)

    # unigram 为主（0.7），bigram 短语命中作加分（0.3）
    return 0.7 * uni + 0.3 * bi


def score_all(question: str, facts: list[dict]) -> list[float]:
    """对一个问句与一串事实打分，返回与 facts 同长的分数列表。"""
    return [score_relevance(question, a.get("text", "")) for a in facts]


def is_asked(score: float, threshold: float = 0.2) -> bool:
    """分数是否达到「医生问到了」的门槛。threshold 透明上报，见 M2c-b 结果。"""
    return score >= threshold


# ---------------------------------------------------------------------------
# 缓存（D14：可复现）
# ---------------------------------------------------------------------------

def build_cache(dialogues: dict[int, list[dict]], truthset: dict[int, dict],
                cache_path: Path) -> dict:
    """为所有 (dialogue, turn, fact) 预计算相关性分数并落盘。

    返回嵌套 dict：{dialogue_id(str): {turn_idx(str): [score, ...]}}，
    内层列表与 truthset[did]["truth_set"] 的 fact 顺序对齐。

    dialogues: {did: [row, ...]}（已按 turn 排序），truthset: {did: {truth_set:[...]}}。
    """
    cache: dict = {}
    for did, rows in sorted(dialogues.items()):
        ts = truthset.get(did)
        if not ts:
            continue
        facts = ts["truth_set"]
        per_turn: dict = {}
        # 按对话内【位置】（enumerate 索引）键控，与回放里 sample_behavior 的
        # turn_index=i 对齐（turn_idx 可能有跳号，不能作键）。
        for i, r in enumerate(rows):
            scores = score_all(r["doctor_utt"], facts)
            per_turn[str(i)] = scores
        cache[str(did)] = per_turn

    out = {
        "_meta": {
            "threshold": 0.2,
            "n_dialogues": len(cache),
            "method": "lexical unigram coverage + bigram + medical synonyms",
        },
        "scores": cache,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def load_cache(cache_path: Path) -> dict:
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)
