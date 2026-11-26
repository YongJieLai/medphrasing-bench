"""条件概率表驱动的患者披露行为模拟器（M2 核心）。

按 D1 纪律，患者「多说 / 附和 / 漏答」等披露行为【必须】来自从真实语料估计的
条件概率表采样，而不是 system prompt 里的角色说明。

本模块只做「行为层」采样：给定医生提问的措辞标注（openness / leadingness /
multi_barreled），从条件概率表采样出患者这轮回答的披露行为标签
（relevance / definiteness / completeness）。行为规格的渲染成自然语言由
patient.py 负责（M3 真值场景）。

三个条件分布（各对应一个 M0 已验证的措辞→披露效应）：
  P(relevance     | openness)        -> 溢出率（H2）
  P(definiteness  | leadingness)     -> 附和率（H3）
  P(completeness  | multi_barreled)  -> 漏答率（H4）

参数用 Laplace (+1) 平滑避免零概率。save/load 为 JSON，参数有出处、可复现。
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 各维度的合法取值（过滤掉 n/a 与脏值）
OPENNESS = ("closed", "semi", "open")
LEADINGNESS = ("neutral", "mild", "strong")
MULTI_BARRELED = ("single", "multi")
RELEVANCE = ("on-topic", "spillover", "off-topic")
DEFINITENESS = ("exact", "vague", "agree")
COMPLETENESS = (1, 0)  # 1=答全 0=漏答


@dataclass
class ConditionalSimulator:
    """条件概率表 + 采样器。probs 结构：
    {
      "relevance|openness":      {(o, r): p, ...},
      "definiteness|leadingness":{(l, d): p, ...},
      "completeness|multi_barreled": {(m, c): p, ...},
      "_meta": {...}
    }
    """
    probs: dict = field(default_factory=dict)

    # ---------- 估计 ----------
    @classmethod
    def estimate(cls, rows: list[dict], alpha: float = 1.0) -> "ConditionalSimulator":
        """从标注行估计条件概率表（Laplace 平滑）。"""
        def cond_1d(rows, x_key, x_vals, y_key, y_vals):
            """P(y|x) 的 Laplace 估计。返回 {(x, y): p}。"""
            cnt_xy = defaultdict(int)
            cnt_x = defaultdict(int)
            for r in rows:
                x = r.get(x_key)
                y = r.get(y_key)
                if x in x_vals and y in y_vals:
                    cnt_xy[(x, y)] += 1
                    cnt_x[x] += 1
            out = {}
            for x in x_vals:
                denom = cnt_x[x] + alpha * len(y_vals)
                for y in y_vals:
                    out[(x, y)] = (cnt_xy[(x, y)] + alpha) / denom
            return out, dict(cnt_x)

        pr_rel, n_open = cond_1d(rows, "openness", OPENNESS, "relevance", RELEVANCE)
        pr_def, n_lead = cond_1d(rows, "leadingness", LEADINGNESS, "definiteness", DEFINITENESS)
        pr_cmp, n_multi = cond_1d(rows, "multi_barreled", MULTI_BARRELED, "completeness", COMPLETENESS)

        return cls(probs={
            "relevance|openness": pr_rel,
            "definiteness|leadingness": pr_def,
            "completeness|multi_barreled": pr_cmp,
            "_meta": {
                "alpha": alpha,
                "n_rows": len(rows),
                "n_by_openness": n_open,
                "n_by_leadingness": n_lead,
                "n_by_multi_barreled": n_multi,
            },
        })

    # ---------- 采样 ----------
    def sample_behavior(self, openness: str, leadingness: str,
                        multi_barreled: str, rng: np.random.Generator) -> dict:
        """采样这一轮患者的披露行为标签。措辞取值非法时回退到无条件边际。"""
        def draw(table, x, y_vals):
            if x not in [k[0] for k in table if k[0] == x]:
                # 该措辞档位在训练集无观测 -> 用无条件边际（等权 fallback）
                return y_vals[int(rng.integers(0, len(y_vals)))]
            ys = [y for y in y_vals]
            ps = [table.get((x, y), 0.0) for y in ys]
            ps = np.asarray(ps, dtype=float)
            ps = ps / ps.sum()
            return ys[int(rng.choice(len(ys), p=ps))]

        rel = draw(self.probs["relevance|openness"], openness, RELEVANCE)
        deff = draw(self.probs["definiteness|leadingness"], leadingness, DEFINITENESS)
        comp = draw(self.probs["completeness|multi_barreled"], multi_barreled, COMPLETENESS)
        return {"relevance": rel, "definiteness": deff, "completeness": comp}

    def sample_rates(self, pairs: list[dict], rng: np.random.Generator) -> dict:
        """对一批 (openness, leadingness, multi_barreled) 问句逐条采样，返回三率。"""
        n = len(pairs)
        spill = agree = incomplete = 0
        for p in pairs:
            b = self.sample_behavior(
                p.get("openness"), p.get("leadingness"), p.get("multi_barreled"), rng)
            spill += (b["relevance"] == "spillover")
            agree += (b["definiteness"] == "agree")
            incomplete += (b["completeness"] == 0)
        return {
            "spillover_rate": spill / n if n else float("nan"),
            "agree_rate": agree / n if n else float("nan"),
            "incomplete_rate": incomplete / n if n else float("nan"),
        }

    # ---------- 持久化 ----------
    @staticmethod
    def _serialize(probs: dict) -> dict:
        """tuple key -> 'a§b' 字符串，使 JSON 可序列化。"""
        out = {}
        for k, v in probs.items():
            if k == "_meta":
                out[k] = v
            elif isinstance(v, dict):
                out[k] = {f"{a}§{b}": p for (a, b), p in v.items()}
            else:
                out[k] = v
        return out

    @staticmethod
    def _deserialize(data: dict) -> dict:
        out = {}
        for k, v in data.items():
            if k == "_meta" or not isinstance(v, dict):
                out[k] = v
                continue
            out[k] = {}
            for kk, p in v.items():
                a, b = kk.split("§")
                try:
                    b = int(b)  # completeness 的 0/1 还原为 int
                except ValueError:
                    pass
                out[k][(a, b)] = p
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._serialize(self.probs), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ConditionalSimulator":
        return cls(probs=cls._deserialize(json.load(open(path, encoding="utf-8"))))
