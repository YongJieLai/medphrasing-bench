"""条件概率表驱动的患者披露行为模拟器（M2 核心）。

按 D1 纪律，患者「多说 / 附和 / 漏答」等披露行为【必须】来自从真实语料估计的
条件概率表采样，而不是 system prompt 里的角色说明。

本模块提供两种模式：

1. **1 维模式（`estimate` + 无状态 `sample_behavior`）** —— M2 早期产物，
   只条件于措辞（openness/leadingness/multi_barreled），无任何状态依赖。
   **本模式仅作为 M2b 的阴性对照（D11）保留**：它必须在 B1/B2 结构判据上失败，
   以证明 3 维模式的状态依赖不是「代码一写就对」的恒等式。

2. **3 维模式（`estimate_3d` + 有状态 `sample_behavior`）** —— M2b 主体。
   条件于（措辞 × 剩余量档 × 轮次档），并由 disclosure_ledger 提供剩余量。
   四个输出维度：
     P(relevance     | openness,       remaining_bin, turn_bin)
     P(definiteness  | leadingness,    remaining_bin, turn_bin)
     P(completeness  | multi_barreled, remaining_bin, turn_bin)
     P(info_units    | openness,       remaining_bin, turn_bin)   # 披露量

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
INFO_UNITS = tuple(range(0, 11))  # 0..10


def _bin_of(value: int, bins: list[dict]) -> str:
    """把数值映射到分档名。bins 为 [{"name","min","max"}, ...]。"""
    for b in bins:
        if b["min"] <= value <= b["max"]:
            return b["name"]
    return bins[-1]["name"]


@dataclass
class ConditionalSimulator:
    """条件概率表 + 采样器。

    probs 结构（1 维模式）：
      {"relevance|openness": {(o, r): p}, ...}
    probs 结构（3 维模式，`_meta.mode == "3d"`）：
      {"relevance": {(o, rbin, tbin): {rv: p}, ...},
       "definiteness": {...}, "completeness": {...}, "info_units": {...}}
    """
    probs: dict = field(default_factory=dict)

    # =====================================================================
    # 1 维估计（阴性对照，D11）
    # =====================================================================
    @classmethod
    def estimate(cls, rows: list[dict], alpha: float = 1.0) -> "ConditionalSimulator":
        """从标注行估计 1 维条件概率表（Laplace 平滑）。无状态。"""
        def cond_1d(rows, x_key, x_vals, y_key, y_vals):
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
                "mode": "1d",
                "alpha": alpha,
                "n_rows": len(rows),
                "n_by_openness": n_open,
                "n_by_leadingness": n_lead,
                "n_by_multi_barreled": n_multi,
            },
        })

    # =====================================================================
    # 3 维估计（M2b 主体）
    # =====================================================================
    @classmethod
    def estimate_3d(cls, samples: list[dict], config: dict, alpha: float = 1.0
                    ) -> "ConditionalSimulator":
        """从序贯样本估计 3 维条件概率表。

        samples: 每项含 openness/leadingness/multi_barreled/relevance/
                 definiteness/completeness/info_units/remaining/turn。

        稀疏 cell 用 backoff 平滑：n < backoff_threshold 的 cell 回退到该
        x 档的 1 维边际分布（再不足则回退全局边际），避免 Laplace 在稀疏
        cell 上退化成近均匀分布（均值虚高）。
        """
        turn_bins = config["turn_bins"]
        rem_bins = config["remaining_bins"]
        min_n = config.get("backoff_threshold", 30)

        def cond_3d(samples, x_key, x_vals, y_key, y_vals):
            """P(y | x, remaining_bin, turn_bin)。返回 {（x,rbin,tbin): {y:p}} 与各格计数。

            多级 backoff（按信息量降序）：
              cell(x,rbin,tbin) n≥min_n → 用 cell
              否则 (x,rbin) 边际 n≥min_n → 用该 2 维边际
              否则 x 边际（措辞维是该 y 的强预测因子）n≥min_n → 用 x 边际
              否则 rbin 边际 → 全局边际
            """
            cnt = defaultdict(lambda: defaultdict(int))
            xr_marg = defaultdict(lambda: defaultdict(int))
            r_marg = defaultdict(lambda: defaultdict(int))
            x_marg = defaultdict(lambda: defaultdict(int))
            global_marg = defaultdict(int)
            for s in samples:
                x = s.get(x_key)
                y = s.get(y_key)
                rbin = _bin_of(int(s.get("remaining", 0)), rem_bins)
                tbin = _bin_of(int(s.get("turn", 0)), turn_bins)
                if x in x_vals and y in y_vals:
                    cnt[(x, rbin, tbin)][y] += 1
                    xr_marg[(x, rbin)][y] += 1
                    r_marg[rbin][y] += 1
                    x_marg[x][y] += 1
                    global_marg[y] += 1
            out = {}
            cell_n = {}
            for x in x_vals:
                for rb in [b["name"] for b in rem_bins]:
                    for tb in [b["name"] for b in turn_bins]:
                        cell = cnt[(x, rb, tb)]
                        n = sum(cell.values())
                        cell_n[(x, rb, tb)] = n
                        if n >= min_n:
                            base = cell
                        elif sum(xr_marg[(x, rb)].values()) >= min_n:
                            base = xr_marg[(x, rb)]
                        elif sum(x_marg[x].values()) >= min_n:
                            base = x_marg[x]
                        elif sum(r_marg[rb].values()) >= min_n:
                            base = r_marg[rb]
                        else:
                            base = global_marg
                        denom = sum(base.values()) + alpha * len(y_vals)
                        out[(x, rb, tb)] = {
                            y: (base.get(y, 0) + alpha) / denom for y in y_vals
                        }
            return out, cell_n

        pr_rel, n_rel = cond_3d(samples, "openness", OPENNESS, "relevance", RELEVANCE)
        pr_def, n_def = cond_3d(samples, "leadingness", LEADINGNESS, "definiteness", DEFINITENESS)
        pr_cmp, n_cmp = cond_3d(samples, "multi_barreled", MULTI_BARRELED, "completeness", COMPLETENESS)
        pr_iu, n_iu = cond_3d(samples, "openness", OPENNESS, "info_units", INFO_UNITS)

        return cls(probs={
            "relevance": pr_rel,
            "definiteness": pr_def,
            "completeness": pr_cmp,
            "info_units": pr_iu,
            "_meta": {
                "mode": "3d",
                "alpha": alpha,
                "n_samples": len(samples),
                "turn_bins": turn_bins,
                "remaining_bins": rem_bins,
                "cell_n": {
                    "relevance": n_rel,
                    "definiteness": n_def,
                    "completeness": n_cmp,
                    "info_units": n_iu,
                },
            },
        })

    # =====================================================================
    # 3 维消融估计（M2b-1 阴性对照：remaining 抹平）
    # =====================================================================
    @classmethod
    def estimate_3d_ablated(cls, samples: list[dict], config: dict,
                            alpha: float = 1.0) -> "ConditionalSimulator":
        """remaining 抹平消融（M2b-1 §7.3 第 3 件事的阴性对照，D11 失败路径）。

        结构完全同 estimate_3d，唯一区别：info_units 维的 remaining 依赖被抹平，
        所有单元格回退到 x（openness）边际分布，等价于「remaining 系数 = 0」。
        relevance / definiteness / completeness 三维保持 3 维不变。

        用途：B2 的阴性对照。该模型在 B2（E[iu|remaining_bin] 斜率）上必须失败
        （斜率 ≈ 0 或符号不符），证明 3 维模型的残量效应不是「结构一写就对」的恒等式。
        """
        sim = cls.estimate_3d(samples, config, alpha=alpha)
        turn_bins = config["turn_bins"]
        rem_bins = config["remaining_bins"]

        # info_units 回退到 x 边际：P(iu | openness)，remaining 与 turn 均不起作用
        x_marg: dict = defaultdict(lambda: defaultdict(int))
        for s in samples:
            if s.get("openness") in OPENNESS and s.get("info_units") in INFO_UNITS:
                x_marg[s["openness"]][s["info_units"]] += 1

        ablated_iu = {}
        for x in OPENNESS:
            denom = sum(x_marg[x].values()) + alpha * len(INFO_UNITS)
            dist = {y: (x_marg[x].get(y, 0) + alpha) / denom for y in INFO_UNITS}
            for rb in [b["name"] for b in rem_bins]:
                for tb in [b["name"] for b in turn_bins]:
                    ablated_iu[(x, rb, tb)] = dict(dist)
        sim.probs["info_units"] = ablated_iu
        # 保持 mode="3d" 走 3 维采样路径；加 ablated 标记供 D14 可指认
        sim.probs["_meta"]["mode"] = "3d"
        sim.probs["_meta"]["ablated"] = True
        return sim

    # =====================================================================
    # 采样
    # =====================================================================
    def _draw(self, table: dict, key, y_vals: tuple, rng: np.random.Generator):
        """从给定 key 的条件分布抽一个 y。key 不存在则用等权边际 fallback。"""
        if key not in table:
            return y_vals[int(rng.integers(0, len(y_vals)))]
        ps = np.asarray([table[key].get(y, 0.0) for y in y_vals], dtype=float)
        ps = ps / ps.sum()
        return y_vals[int(rng.choice(len(y_vals), p=ps))]

    def sample_behavior(self, openness: str, leadingness: str, multi_barreled: str,
                        rng: np.random.Generator, remaining_n: int | None = None,
                        turn_index: int | None = None,
                        dialogue_baseline: int | None = None) -> dict:
        """采样这一轮患者的披露行为标签（含披露量 info_units）。

        3 维模式：remaining_n 与 turn_index 生效（状态依赖）。
        1 维模式：忽略状态，退化为无状态采样（阴性对照）。
        措辞取值非法时回退到无条件边际。
        """
        mode = self.probs.get("_meta", {}).get("mode", "1d")

        if mode == "3d":
            rem_bins = self.probs["_meta"]["remaining_bins"]
            turn_bins = self.probs["_meta"]["turn_bins"]
            # remaining/turn 缺失时（如 M3 初始调用），用 baseline 作第 0 轮剩余
            if remaining_n is None:
                remaining_n = dialogue_baseline if dialogue_baseline is not None else 0
            if turn_index is None:
                turn_index = 0
            rbin = _bin_of(int(remaining_n), rem_bins)
            tbin = _bin_of(int(turn_index), turn_bins)

            rel = self._draw(self.probs["relevance"], (openness, rbin, tbin), RELEVANCE, rng)
            deff = self._draw(self.probs["definiteness"], (leadingness, rbin, tbin), DEFINITENESS, rng)
            comp = self._draw(self.probs["completeness"], (multi_barreled, rbin, tbin), COMPLETENESS, rng)
            iu = self._draw(self.probs["info_units"], (openness, rbin, tbin), INFO_UNITS, rng)
            # 附和不计披露（§2.2 契约；真实数据 agree 轮 info_units 均值 0.034）
            if deff == "agree":
                iu = 0
            return {
                "relevance": rel, "definiteness": deff, "completeness": comp,
                "info_units": int(iu),
                "remaining_bin": rbin, "turn_bin": tbin,
            }

        # 1 维模式（阴性对照）
        def draw_1d(table, x, y_vals):
            if x not in [k[0] for k in table]:
                return y_vals[int(rng.integers(0, len(y_vals)))]
            ps = np.asarray([table.get((x, y), 0.0) for y in y_vals], dtype=float)
            ps = ps / ps.sum()
            return y_vals[int(rng.choice(len(y_vals), p=ps))]

        rel = draw_1d(self.probs["relevance|openness"], openness, RELEVANCE)
        deff = draw_1d(self.probs["definiteness|leadingness"], leadingness, DEFINITENESS)
        comp = draw_1d(self.probs["completeness|multi_barreled"], multi_barreled, COMPLETENESS)
        return {"relevance": rel, "definiteness": deff, "completeness": comp, "info_units": None}

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

    # =====================================================================
    # 持久化
    # =====================================================================
    @staticmethod
    def _serialize(probs: dict) -> dict:
        """嵌套 dict/tuple key -> 可 JSON 化。"""
        def _walk(obj):
            if isinstance(obj, dict):
                out = {}
                for kk, vv in obj.items():
                    kk2 = kk if not isinstance(kk, tuple) else "§".join(map(str, kk))
                    out[kk2] = _walk(vv)
                return out
            if isinstance(obj, (list, tuple)):
                return [_walk(x) for x in obj]
            return obj

        out = {}
        for k, v in probs.items():
            if k == "_meta":
                out[k] = _walk(v)
            elif k in ("relevance", "definiteness", "completeness", "info_units") \
                    and isinstance(v, dict) and v and isinstance(next(iter(v.values())), dict):
                # 3 维：(x, rbin, tbin) -> {y: p}
                out[k] = {f"{a}§{b}§{c}": inner for (a, b, c), inner in v.items()}
            elif isinstance(v, dict):
                # 1 维：(x, y) -> p
                out[k] = {f"{a}§{b}": p for (a, b), p in v.items()}
            else:
                out[k] = v
        return out

    @staticmethod
    def _deserialize(data: dict) -> dict:
        """把 _serialize 的输出还原回内存结构（tuple key / int key）。

        修掉「存盘即坏」：JSON 会把 dict 的 int key 变成字符串，读回时不还原，
        导致 info_units/completeness 的 inner key 对不上 _draw 里的 int 查询
        （.get(0) 全 miss → 概率全 0 → rng.choice 崩）。同时还原 _meta.cell_n 的
        tuple key（覆盖边界报告依赖它）。
        """
        INT_INNER = {"info_units", "completeness"}  # 这些维度的 y 取值是 int

        def _restore_meta(meta: dict) -> dict:
            out = {}
            for k, v in meta.items():
                if k == "cell_n" and isinstance(v, dict):
                    out[k] = {}
                    for dim, cells in v.items():
                        if isinstance(cells, dict):
                            out[k][dim] = {
                                tuple(kk.split("§")): n for kk, n in cells.items()
                            }
                        else:
                            out[k][dim] = cells
                else:
                    out[k] = v
            return out

        out = {}
        for k, v in data.items():
            if not isinstance(v, dict):
                out[k] = v
                continue
            if k == "_meta":
                out[k] = _restore_meta(v)
                continue
            out[k] = {}
            for kk, p in v.items():
                parts = kk.split("§")
                if len(parts) == 3:
                    # 3 维：(x, rbin, tbin) -> {y:p}
                    a, b, c = parts
                    if k in INT_INNER and isinstance(p, dict):
                        p = {int(y): prob for y, prob in p.items()}
                    out[k][(a, b, c)] = p
                elif len(parts) == 2:
                    a, b = parts
                    try:
                        b = int(b)  # completeness 0/1 还原 int
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
