"""M3-0a 实时措辞判别器。

输入一句医生问话（纯文本），可选附上一患者发言（follow_up 判定用），
输出四维标签 + 每维置信度。

设计要点（对应 §7.10）：
  - 模型：qwen-plus（reviewer 角色），非 M0 标注器 qwen-flash —— 同源约束路径 A
  - 独立 prompt（聚焦四维 + 置信度，表述与 M0 标注 prompt 不同），四维定义与 M0 一致
  - 缓存：键 = (规范化问句, 规范化 prev)，保证同输入同输出（判据 3）
  - 延迟：单次 < 500ms（判据 4，实测另测）

核心判据「无上下文下也成立」：openness/leadingness/multi_barreled 三维
在 prev=None 时也须判对；follow_up 是唯一依赖 prev 的维度。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .llm import chat

DIMS = ["openness", "leadingness", "multi_barreled", "follow_up"]
VALID = {
    "openness": {"open", "semi", "closed"},
    "leadingness": {"neutral", "mild", "strong"},
    "multi_barreled": {"single", "multi"},
    "follow_up": {"surface", "deepen", "new", "n/a"},
}

PROMPT = """You are a real-time phrasing classifier. Classify a DOCTOR'S QUESTION from a clinical dialogue on four dimensions.

PREVIOUS PATIENT UTTERANCE (only needed to judge follow_up; empty if none):
"{prev}"

DOCTOR'S QUESTION:
"{doctor}"

Classify on four dimensions. For each, give a label AND a confidence (0-1 = how sure you are).

- openness: how much the question constrains the answer.
  * "closed": answerable with yes/no or a choice (auxiliary-verb question, "Any...", or "A or B?").
  * "semi": a wh-question expecting ONE concrete fact (a number, name, time, or list).
  * "open": invites free narrative/description with no single expected answer.
- leadingness: "neutral" (no answer implied) | "mild" (suggests a possible answer, patient can deny) | "strong" (pushes the patient to confirm, e.g. tag question "...correct?" / "...right?").
- multi_barreled: "single" (one piece of info) | "multi" (>=2 distinct pieces in one utterance).
- follow_up: relation to the PREVIOUS PATIENT UTTERANCE — "surface" (repeats/rephrases a previous question) | "deepen" (drills into what the patient just said) | "new" (new topic). If prev is empty, "n/a".

Output STRICT JSON only:
{{"openness": {{"label": "..", "confidence": 0.0}}, "leadingness": {{"label": "..", "confidence": 0.0}}, "multi_barreled": {{"label": "..", "confidence": 0.0}}, "follow_up": {{"label": "..", "confidence": 0.0}}}}"""


def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


class Discriminator:
    def __init__(self, cache_path: Path | str | None = None):
        self._cache: dict[tuple, dict] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self):
        with open(self.cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self._cache[(d["doctor"], d["prev"])] = d["out"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def _save(self, key: tuple, out: dict):
        self._cache[key] = out
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"doctor": key[0], "prev": key[1], "out": out},
                                   ensure_ascii=False) + "\n")

    def _parse(self, raw: str) -> dict | None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        out = {}
        for dim in DIMS:
            v = d.get(dim)
            if not isinstance(v, dict):
                return None
            label = str(v.get("label", "")).strip()
            if label not in VALID[dim]:
                return None
            try:
                conf = float(v.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            out[dim] = {"label": label, "confidence": max(0.0, min(1.0, conf))}
        return out

    def discriminate(self, doctor_utt: str, prev: str | None = None,
                     retries: int = 2) -> dict:
        """判四维。返回 {dim: {label, confidence, latency_ms}}。"""
        key = (_norm(doctor_utt), _norm(prev))
        if key in self._cache:
            return self._cache[key]

        prompt = PROMPT.format(prev=prev or "", doctor=doctor_utt)
        t0 = time.time()
        out = None
        for attempt in range(retries + 1):
            temp = 0.0 if attempt == 0 else 0.3
            try:
                raw = chat("reviewer", [{"role": "user", "content": prompt}],
                           temperature=temp, max_tokens=512)
                out = self._parse(raw)
                if out:
                    break
            except Exception as e:
                print(f"  [warn] discriminate: {e}")
        latency = time.time() - t0
        if out is None:
            out = {d: {"label": None, "confidence": 0.0} for d in DIMS}
        for d in DIMS:
            out[d]["latency_ms"] = round(latency * 1000, 1)
        self._save(key, out)
        return out
