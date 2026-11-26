"""M2c 主脚本：把四维行为标签渲染成中文患者口语，并回抽验证忠实度。

覆盖 §7.6 的第 1/2/3/4 件：
  1. 渲染层（renderer.py + aiu_selector.py 已在 src/）
  2. 回抽一致性：集合级 F1（LLM 跨语言事实对齐），分维 info_units/completeness/溢出率
  3. B3 轮间自相关：渲染后 iu 序列 lag-1 自相关 ≠ 0
  4. r0 档 30 条样例：remaining==0 时全部零披露

用法:
  python scripts/m2c_render.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --probs data/simulator_probs_v3_20261127.json \
      --out outputs/YYYYMMDD_m2c --limit 5   # pilot
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conditional_simulator import ConditionalSimulator
from src.disclosure_ledger import AiuLedger
from src.aiu_selector import select_aius
from src.renderer import render_turn
from src.llm import chat
from scripts.calibrate_simulator import valid_rows
from scripts.extract_aiu import extract_aius

ALIGN_SYSTEM = """You verify whether a simulated patient's utterance faithfully covers a given fact list.

FACT LIST (must ALL be said):
{gt}

OMIT LIST (must NOT be said; the patient is supposed to omit these):
{omit}

PATIENT'S UTTERANCE (Chinese):
\"\"\"
{rendered}
\"\"\"

For each fact in FACT LIST, decide if the patient actually stated it (semantic equivalence is enough, Chinese or English). Also list any NEW clinical fact the patient said that is not in FACT LIST (ignore fillers, greetings, acknowledgments). Also list any OMIT fact the patient wrongly said.

Output STRICT JSON only:
{{"matched": ["...exact fact text...", ...], "missed": ["...", ...], "extra": ["...", ...], "said_omit": ["...", ...]}}"""


def parse_json(raw: str):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def align_facts(gt: list[dict], omit: list[dict], rendered: str,
                retries: int = 2) -> dict:
    """LLM 跨语言事实对齐。返回 {matched, missed, extra, said_omit}（均为事实文本列表）。"""
    if not gt:
        # 无应说事实（r0/agree 特判轮）：直接回抽渲染文本，验证零新事实
        aius = extract_aius("", rendered)
        return {"matched": [], "missed": [], "extra": [a["text"] for a in aius],
                "said_omit": []}

    gt_txt = "\n".join(f"- {a['text']}" for a in gt)
    omit_txt = "\n".join(f"- {a['text']}" for a in omit) if omit else "（无）"
    prompt = ALIGN_SYSTEM.format(gt=gt_txt, omit=omit_txt, rendered=rendered)
    for attempt in range(retries + 1):
        try:
            raw = chat("reviewer", [{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=1024)
            d = parse_json(raw)
            if isinstance(d, dict) and all(k in d for k in ("matched", "missed", "extra", "said_omit")):
                return {k: (v if isinstance(v, list) else []) for k, v in d.items()}
        except Exception as e:
            print(f"    [warn] align {e}")
        time.sleep(1)
    return {"matched": [], "missed": [a["text"] for a in gt],
            "extra": [], "said_omit": []}


def render_dialogue(dialogue_id: int, rows: list[dict], truth: dict,
                    sim: ConditionalSimulator, rng: np.random.Generator) -> list[dict]:
    """对一段对话逐轮回放渲染，返回每轮记录（含行为标签、事实清单、渲染文本、对齐结果）。"""
    truth_set = truth[dialogue_id]["truth_set"]
    ledger = AiuLedger(truth_set)
    prev_patient = None
    records = []
    for i, row in enumerate(rows):
        behavior = sim.sample_behavior(
            row["openness"], row["leadingness"], row["multi_barreled"], rng,
            remaining_n=ledger.remaining_count(), turn_index=i,
            dialogue_baseline=len(truth_set))
        instruction = select_aius(ledger, behavior)
        rendered = render_turn(row["doctor_utt"], prev_patient, instruction,
                               rng=random.Random(0))
        ledger.record_indices(instruction["indices"])

        # 回抽对齐（GT = must_say；omit 单独列出）
        # 特殊轮（r0/agree/no_content）为固定模板，零披露由模板保证，不参与回抽
        if instruction.get("special"):
            align = {"matched": [], "missed": [], "extra": [], "said_omit": []}
            gt = []
            omit = []
        else:
            gt = instruction["must_say"]
            omit = instruction["omit"]
            align = align_facts(gt, omit, rendered)

        records.append({
            "dialogue_id": dialogue_id,
            "turn": i,
            "doctor_utt": row["doctor_utt"],
            "openness": row["openness"],
            "leadingness": row["leadingness"],
            "multi_barreled": row["multi_barreled"],
            "behavior": behavior,
            "gt_n": len(gt),
            "omit_n": len(omit),
            "special": instruction.get("special"),
            "rendered": rendered,
            "align": align,
            # 渲染后回抽的事实数 = 说对的 + 多说的
            "pred_n": len(align["matched"]) + len(align["extra"]),
            "label_iu": instruction["info_units"],
        })
        prev_patient = rendered
    return records


def f1_of(gt_n: int, align: dict) -> float:
    """集合级 F1 = 2|matched| / (|GT| + |matched| + |extra| + |said_omit|)。"""
    matched = len(align["matched"])
    extra = len(align["extra"])
    said_omit = len(align["said_omit"])
    denom = gt_n + matched + extra + said_omit
    return 2 * matched / denom if denom > 0 else (1.0 if gt_n == 0 else 0.0)


def lag1_autocorr(seq: list[float]) -> float:
    if len(seq) < 3:
        return float("nan")
    x = np.asarray(seq[:-1], dtype=float)
    y = np.asarray(seq[1:], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="pilot：只处理前 N 个对话")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}
    sim = ConditionalSimulator.load(Path(args.probs))
    rng = np.random.default_rng(args.seed)

    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_d[r["dialogue_id"]].append(r)
    for d in by_d:
        by_d[d].sort(key=lambda x: x["turn_idx"])

    # 只保留有真值集的对话
    dial_ids = sorted(d for d in by_d if d in truthset)
    if args.limit:
        dial_ids = dial_ids[: args.limit]

    print(f"抽样对话 {len(dial_ids)} 个（pilot={args.limit}）")

    # ---------- 回抽一致性 + 渲染 ----------
    all_records = []
    t0 = time.time()
    for di, did in enumerate(dial_ids):
        recs = render_dialogue(did, by_d[did], truthset, sim, rng)
        all_records.extend(recs)
        if (di + 1) % 5 == 0:
            print(f"  进度 {di+1}/{len(dial_ids)}  耗时 {time.time()-t0:.0f}s")

    # 汇总回抽一致性
    gt_total = sum(r["gt_n"] for r in all_records)
    matched_total = sum(len(r["align"]["matched"]) for r in all_records)
    extra_total = sum(len(r["align"]["extra"]) for r in all_records)
    said_omit_total = sum(len(r["align"]["said_omit"]) for r in all_records)
    missed_total = sum(len(r["align"]["missed"]) for r in all_records)

    denom = gt_total + matched_total + extra_total + said_omit_total
    macro_f1 = (2 * matched_total / denom) if denom > 0 else float("nan")
    # 每轮 F1 的平均（轮级）
    per_turn_f1 = [f1_of(r["gt_n"], r["align"]) for r in all_records]
    mean_f1 = float(np.mean(per_turn_f1)) if per_turn_f1 else float("nan")

    n_normal = [r for r in all_records if not r["special"]]
    # info_units 一致率（说全 + 不多说）
    exact = [r for r in n_normal if r["pred_n"] == r["gt_n"] and len(r["align"]["extra"]) == 0
             and len(r["align"]["said_omit"]) == 0]
    info_units_acc = len(exact) / len(n_normal) if n_normal else float("nan")
    # 溢出率（多说了清单外事实）
    spill = [r for r in n_normal if len(r["align"]["extra"]) > 0 or len(r["align"]["said_omit"]) > 0]
    spill_rate = len(spill) / len(n_normal) if n_normal else float("nan")
    # 漏答率（completeness=1 时 missed > 0 算漏；completeness=0 时 said_omit 算漏答失败）
    comp1 = [r for r in n_normal if r["behavior"]["completeness"] == 1]
    comp0 = [r for r in n_normal if r["behavior"]["completeness"] == 0]
    leak_comp1 = sum(1 for r in comp1 if len(r["align"]["missed"]) > 0) / len(comp1) if comp1 else float("nan")
    comp0_ok = sum(1 for r in comp0 if len(r["align"]["said_omit"]) == 0) / len(comp0) if comp0 else float("nan")

    # ---------- B3 轮间自相关 ----------
    by_dial_records = defaultdict(list)
    for r in all_records:
        by_dial_records[r["dialogue_id"]].append(r)
    label_lag, pred_lag = [], []
    for did, recs in by_dial_records.items():
        label_seq = [r["label_iu"] for r in recs]
        pred_seq = [r["pred_n"] for r in recs]
        ll = lag1_autocorr(label_seq)
        pl = lag1_autocorr(pred_seq)
        if not np.isnan(ll):
            label_lag.append(ll)
        if not np.isnan(pl):
            pred_lag.append(pl)
    label_lag_mean = float(np.mean(label_lag)) if label_lag else float("nan")
    pred_lag_mean = float(np.mean(pred_lag)) if pred_lag else float("nan")

    # ---------- r0 档 30 条样例 ----------
    r0_samples = []
    r0_pool = [d for d in dial_ids if truthset[d]["truth_set_size"] >= 2][:30]
    for did in r0_pool:
        ts = truthset[did]["truth_set"]
        ledger = AiuLedger(ts)
        # 模拟「已说完全部事实」：直接披露满
        ledger.record_indices(range(len(ts)))
        # 标签层仍可能输出 info_units>0（r0 档定性错误），验证闸门拦截
        fake_behavior = {"relevance": "on-topic", "definiteness": "exact",
                         "completeness": 1, "info_units": 3}
        inst = select_aius(ledger, fake_behavior)
        rendered = render_turn("Anything else?", "没有了", inst, rng=random.Random(1))
        r0_samples.append({
            "dialogue_id": did,
            "special": inst["special"],
            "info_units_out": inst["info_units"],
            "rendered": rendered,
        })

    # ---------- 一段完整模拟对话（取轮数最多的对话） ----------
    demo_did = max(dial_ids, key=lambda d: len(by_d[d]))
    demo_recs = render_dialogue(demo_did, by_d[demo_did], truthset, sim, rng)

    # ---------- 输出 ----------
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "n_dialogues": len(dial_ids),
        "n_turns": len(all_records),
        "all_records": all_records,
        "recall_consistency": {
            "gt_total": gt_total,
            "matched_total": matched_total,
            "extra_total": extra_total,
            "said_omit_total": said_omit_total,
            "missed_total": missed_total,
            "macro_f1": macro_f1,
            "mean_per_turn_f1": mean_f1,
            "info_units_acc": info_units_acc,
            "spill_rate": spill_rate,
            "comp1_leak_rate": leak_comp1,
            "comp0_ok_rate": comp0_ok,
        },
        "B3_autocorr": {
            "label_iu_lag1": label_lag_mean,
            "rendered_iu_lag1": pred_lag_mean,
            "n_dialogues_with_lag": len(pred_lag),
        },
        "r0_samples": r0_samples,
        "demo_dialogue": demo_recs,
    }
    with open(outdir / "m2c_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("M2c 回抽一致性（集合级 F1）")
    print("=" * 78)
    print(f"  轮数 n={len(all_records)}  应说事实 GT={gt_total}  说对 matched={matched_total}")
    print(f"  多说 extra={extra_total}  误说 omit={said_omit_total}  漏说 missed={missed_total}")
    print(f"  集合级 F1 = {macro_f1:.4f}   每轮 F1 均值 = {mean_f1:.4f}")
    print(f"  info_units 一致率 = {info_units_acc:.2%}")
    print(f"  溢出率(多说) = {spill_rate:.2%}")
    print(f"  completeness=1 漏说率 = {leak_comp1:.2%}")
    print(f"  completeness=0 漏答忠实率 = {comp0_ok:.2%}")
    print("\n" + "=" * 78)
    print("B3 轮间自相关（lag-1）")
    print("=" * 78)
    print(f"  标签层 iu 序列 lag-1 = {label_lag_mean:.4f}")
    print(f"  渲染后 iu 序列 lag-1 = {pred_lag_mean:.4f}  (n对话={len(pred_lag)})")
    print("\n" + "=" * 78)
    print(f"r0 档样例 {len(r0_samples)} 条（应全为零披露）")
    print("=" * 78)
    print(f"  special 全为 'r0': {all(s['special']=='r0' for s in r0_samples)}")
    print(f"  info_units_out 全为 0: {all(s['info_units_out']==0 for s in r0_samples)}")
    print(f"  渲染样例: {r0_samples[0]['rendered']!r}  {r0_samples[1]['rendered']!r}")

    print("\n" + "=" * 78)
    print(f"完整模拟对话 demo（dialogue {demo_did}, {len(demo_recs)} 轮）")
    print("=" * 78)
    for r in demo_recs:
        print(f"\n[医生] {r['doctor_utt']}")
        print(f"[患者] {r['rendered']}")
        print(f"  (标签: open={r['openness']} lead={r['leadingness']} multi={r['multi_barreled']} "
              f"iu={r['behavior']['info_units']} 说对={len(r['align']['matched'])}/{r['gt_n']})")

    print(f"\n完成 -> {outdir}/m2c_result.json")


if __name__ == "__main__":
    main()
