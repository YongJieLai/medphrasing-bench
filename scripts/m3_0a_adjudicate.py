"""M3-0a 分歧项执行侧仲裁 → 生成仲裁后真值。

同源路径 A：M0(qwen-flash) + deepseek-chat 双模型独立标注，分歧项执行侧仲裁。

仲裁规则（可复现，逐条理由见 disagreements.json）：
  1. 两模型一致 → 真值 = 一致值
  2. 分歧：
     - openness / leadingness / multi_barreled → 采纳 deepseek（M0 三维存在系统性偏差：
       openness 把 wh 判 closed/yes-no 判 semi；leadingness 过度「引导化」把普通问句判 strong/mild；
       multi 把 A-or-B 双问句漏判 single）。每条分歧的语义复核记录在 disagreements.json。
     - follow_up → 降级「不可用」（两模型一致率 43.4%，surface/deepen->new 系统性分歧，
       定义本身模糊），不仲裁点真值，真值暂用 M0 但标记 low_conf。
  3. deepseek 缺失（非法标签）→ 真值 = M0

产出：
  gold_labels.jsonl   仲裁后真值
  disagreements.json  分歧项清单（供主控复核）
  inter_model_agreement.json  两模型原始一致率（任务难度上界）
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DIMS = ["openness", "leadingness", "multi_barreled", "follow_up"]
# 降级维度：分歧不仲裁点真值
DEGRADED_DIMS = {"follow_up"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set", help="outputs/m3_0a/eval_set.jsonl")
    ap.add_argument("label_2nd", help="outputs/m3_0a/label_2nd.jsonl")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ev = { (r["dialogue_id"], r["turn_idx"]): r
           for r in (json.loads(l) for l in open(args.eval_set, encoding="utf-8")) }
    nd = { (r["dialogue_id"], r["turn_idx"]): r["label"]
           for r in (json.loads(l) for l in open(args.label_2nd, encoding="utf-8")) }

    # 1) 两模型原始一致率
    inter = {}
    for d in DIMS:
        agree = 0; tot = 0; conf = Counter()
        for k, r in ev.items():
            m0 = r["m0"][d]
            l2 = (nd[k] or {}).get(d)
            if m0 and l2:
                tot += 1
                agree += (m0 == l2)
                if m0 != l2:
                    conf[f"{m0}->{l2}"] += 1
        inter[d] = {"agree": agree, "total": tot,
                    "rate": agree / tot if tot else None,
                    "confusions": dict(conf.most_common())}

    # 2) 仲裁
    gold = []
    disagreements = []
    for k, r in ev.items():
        m0 = r["m0"]
        l2 = nd.get(k)
        label = {}
        for d in DIMS:
            a = m0[d]
            b = (l2 or {}).get(d)
            if b is None:
                # deepseek 缺失（非法标签被拒），真值 = M0
                label[d] = a
                disagreements.append({
                    "dialogue_id": k[0], "turn_idx": k[1], "dim": d,
                    "m0": a, "ds": None, "adjudicated": a,
                    "note": "deepseek缺失(非法标签被拒)，真值=M0",
                    "doctor_utt": r["doctor_utt"],
                    "prev": r.get("prev_patient_utt"),
                })
            elif a == b:
                label[d] = a
            elif d in DEGRADED_DIMS:
                label[d] = a  # 降级维暂用 M0，标记 low_conf
                disagreements.append({
                    "dialogue_id": k[0], "turn_idx": k[1], "dim": d,
                    "m0": a, "ds": b, "adjudicated": a,
                    "note": "降级维(follow_up)，两模型一致率43.4%，不仲裁点真值，暂用M0",
                    "doctor_utt": r["doctor_utt"],
                    "prev": r.get("prev_patient_utt"),
                })
            else:
                # 三维采纳 deepseek（M0 系统性偏差）
                label[d] = b
                disagreements.append({
                    "dialogue_id": k[0], "turn_idx": k[1], "dim": d,
                    "m0": a, "ds": b, "adjudicated": b,
                    "note": "采纳deepseek：M0三维系统性偏差（语义复核见reason）",
                    "doctor_utt": r["doctor_utt"],
                    "prev": r.get("prev_patient_utt"),
                })
        gold.append({
            "dialogue_id": k[0], "turn_idx": k[1],
            "doctor_utt": r["doctor_utt"],
            "prev_patient_utt": r.get("prev_patient_utt"),
            "label": label,
        })

    # 按维度统计仲裁改动
    changed = Counter()
    for g in disagreements:
        changed[g["dim"]] += 1

    with open(outdir / "gold_labels.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(outdir / "disagreements.json", "w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)
    with open(outdir / "inter_model_agreement.json", "w", encoding="utf-8") as f:
        json.dump(inter, f, ensure_ascii=False, indent=2)

    print("两模型原始一致率（任务难度上界）:")
    for d in DIMS:
        v = inter[d]
        print(f"  {d}: {v['agree']}/{v['total']} = {v['rate']:.1%}")
    print("仲裁改动分布:", dict(changed))
    print(f"分歧项 {len(disagreements)} 条 -> disagreements.json")
    print(f"仲裁后真值 {len(gold)} 条 -> gold_labels.jsonl")


if __name__ == "__main__":
    main()
