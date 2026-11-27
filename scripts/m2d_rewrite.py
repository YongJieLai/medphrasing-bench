"""M2d 改写与校验（§7.8 第 1/2 件）：把 holdout 里的问句在 open <-> closed 之间改写。

覆盖：
  1. 划分 holdout（seed=42, 20% 对话，与 replay_validation.py 同口径）
  2. 取 holdout 中 open 问句全量改写为 closed；closed 问句抽样（对等数量）改写为 open
     —— 用 builder 角色（qwen-flash），约束：语义等价 / 信息量不变 / 只改句式
  3. 用 M0 标注器（annotator 角色）判改写后问句的 openness，报判对率
     —— 判据：closed 判对率 >= 0.90，不过即停
  4. 产出 20 条「原文/改写」对供人工核对语义等价

用法:
  python scripts/m2d_rewrite.py data/annot_all_flash.jsonl \
      --truthset data/aiu_truthset_v2_20261126.json \
      --out outputs/YYYYMMDD_m2d --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import chat
from scripts.calibrate_simulator import valid_rows

REWRITE_PROMPT = """You are rewriting a doctor's question for a research study on how question phrasing affects how much a patient discloses.

Rules:
1. Keep the SAME meaning — ask about exactly the same thing(s) the original asked.
2. Keep the SAME information content — do NOT add any new thing the doctor did not ask about, and do NOT drop anything the doctor did ask about.
3. ONLY change the phrasing style.

Target style: {target}

Original question:
"{question}"

Output ONLY the rewritten question (one sentence, in English). No quotes, no explanation, no extra text."""

TARGET_CLOSED = ('a closed question — answerable with yes/no or a choice. '
                 'Start with an auxiliary verb (Do/Does/Did/Is/Are/Have/Has/Can/Could/Would/Should) '
                 'or "Any...", or offer "A or B?". Ask the same thing the original asked.')

TARGET_OPEN = ('an open question — invite a free narrative or description with no single expected answer '
               '(e.g. "Tell me about...", "Describe...", "What brings you in today?", "How are you feeling?"). '
               'Ask the same thing the original asked.')

CHECK_PROMPT = """Classify how much this doctor's question CONSTRAINS the answer.

- "closed": answerable with yes/no or a choice. Starts with an auxiliary verb (Do/Does/Did/Is/Are/Have/Has/Can/Could/Would/Should) or "Any...", or offers "A or B?".
- "semi": a wh-question expecting ONE concrete fact (a number, name, time, or list).
- "open": invites a free narrative/description with no single expected answer.

Question: "{question}"

Output ONLY one word: "closed", "semi", or "open"."""


def split_holdout(valid: list[dict], truthset: dict, seed: int, frac: float = 0.2):
    """按 dialogue_id 划分 holdout（与 replay_validation.py 同口径）。"""
    by_d: dict[int, list[dict]] = defaultdict(list)
    for r in valid:
        by_d[r["dialogue_id"]].append(r)
    for d in by_d:
        by_d[d].sort(key=lambda x: x["turn_idx"])
    dial_ids = sorted(d for d in by_d if d in truthset)
    rng = np.random.default_rng(seed)
    rng.shuffle(dial_ids)
    n_hold = int(len(dial_ids) * frac)
    hold_ids = set(dial_ids[:n_hold])
    return by_d, hold_ids


def rewrite_one(q: str, target: str) -> str:
    prompt = REWRITE_PROMPT.format(target=target, question=q)
    try:
        out = chat("builder", [{"role": "user", "content": prompt}], temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__ {e}"
    out = (out or "").strip().strip('"').strip("'")
    return out


def check_one(q: str) -> str:
    prompt = CHECK_PROMPT.format(question=q)
    try:
        out = chat("annotator", [{"role": "user", "content": prompt}], temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return f"__ERROR__ {e}"
    out = (out or "").strip().lower().strip('"').strip("'")
    if "open" in out and "closed" not in out and "semi" not in out:
        return "open"
    if "semi" in out:
        return "semi"
    if "closed" in out:
        return "closed"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--truthset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="改写数量上限（0=全部 open）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8")]
    valid = valid_rows(rows)
    truthset = {t["dialogue_id"]: t for t in json.load(open(args.truthset, encoding="utf-8"))}
    by_d, hold_ids = split_holdout(valid, truthset, args.seed)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "rewrites.json"

    # 断点续跑：读已有结果
    if out_path.exists():
        done = json.load(open(out_path, encoding="utf-8"))
        pairs = done["pairs"]
        done_keys = {(p["dialogue_id"], p["turn_idx"], p["direction"]) for p in pairs}
    else:
        pairs = []
        done_keys = set()

    # 收集 open 轮（全量）与 closed 轮（抽样对等数量）
    open_rows = []
    closed_rows = []
    for d in sorted(hold_ids):
        for r in by_d[d]:
            if r["openness"] == "open":
                open_rows.append((d, r))
            elif r["openness"] == "closed":
                closed_rows.append((d, r))

    n_open = len(open_rows)
    rng = np.random.default_rng(args.seed)
    closed_idx = rng.choice(len(closed_rows), size=min(n_open, len(closed_rows)), replace=False)
    closed_sample = [closed_rows[i] for i in sorted(closed_idx)]

    todo = []
    for d, r in open_rows:
        todo.append((d, r, "open->closed"))
    for d, r in closed_sample:
        todo.append((d, r, "closed->open"))

    if args.limit:
        todo = todo[: args.limit]

    print(f"holdout 对话 {len(hold_ids)}；open 轮 {n_open}，closed 抽样 {len(closed_sample)} 条")
    print(f"待改写 {len(todo)} 条（已跳过已完成的 {len(done_keys)} 条）")

    t0 = time.time()
    for i, (d, r, direction) in enumerate(todo):
        key = (d, r["turn_idx"], direction)
        if key in done_keys:
            continue
        target = TARGET_CLOSED if direction == "open->closed" else TARGET_OPEN
        rewritten = rewrite_one(r["doctor_utt"], target)
        judged = check_one(rewritten) if not rewritten.startswith("__ERROR__") else "error"
        pairs.append({
            "dialogue_id": d, "turn_idx": r["turn_idx"], "direction": direction,
            "original": r["doctor_utt"], "original_openness": r["openness"],
            "rewritten": rewritten, "judged_openness": judged,
        })
        if (i + 1) % 25 == 0:
            print(f"  进度 {i+1}/{len(todo)}  耗时 {time.time()-t0:.0f}s")

    # 判对率
    o2c = [p for p in pairs if p["direction"] == "open->closed" and not p["rewritten"].startswith("__ERROR__")]
    c2o = [p for p in pairs if p["direction"] == "closed->open" and not p["rewritten"].startswith("__ERROR__")]
    o2c_ok = sum(1 for p in o2c if p["judged_openness"] == "closed")
    c2o_ok = sum(1 for p in c2o if p["judged_openness"] == "open")
    o2c_rate = o2c_ok / len(o2c) if o2c else float("nan")
    c2o_rate = c2o_ok / len(c2o) if c2o else float("nan")
    n_err = sum(1 for p in pairs if p["rewritten"].startswith("__ERROR__"))

    result = {
        "seed": args.seed,
        "n_holdout_dialogues": len(hold_ids),
        "n_open_rows": n_open,
        "n_closed_sampled": len(closed_sample),
        "check": {
            "open_to_closed": {"n": len(o2c), "judged_closed": o2c_ok, "rate": o2c_rate},
            "closed_to_open": {"n": len(c2o), "judged_open": c2o_ok, "rate": c2o_rate},
            "n_rewrite_errors": n_err,
        },
        "pairs": pairs,
    }
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 20 条人工核对样例（open->closed 10 条 + closed->open 10 条）
    o2c_ok_pairs = [p for p in o2c if p["judged_openness"] == "closed"]
    c2o_ok_pairs = [p for p in c2o if p["judged_openness"] == "open"]
    rng2 = np.random.default_rng(args.seed + 1)
    sample_o = [o2c_ok_pairs[i] for i in rng2.choice(len(o2c_ok_pairs), size=min(10, len(o2c_ok_pairs)), replace=False)]
    sample_c = [c2o_ok_pairs[i] for i in rng2.choice(len(c2o_ok_pairs), size=min(10, len(c2o_ok_pairs)), replace=False)]
    verify = sample_o + sample_c

    print("\n" + "=" * 78)
    print("M2d 改写校验")
    print("=" * 78)
    print(f"open->closed 判对率 = {o2c_rate:.3f} (n={len(o2c)}, 判为 closed {o2c_ok})  "
          f"阈值 0.90 → {'✅' if o2c_rate >= 0.90 else '❌ 停，回主控'}")
    print(f"closed->open 判对率 = {c2o_rate:.3f} (n={len(c2o)}, 判为 open {c2o_ok})")
    print(f"改写失败(空/异常) {n_err} 条")
    print(f"\n20 条人工核对样例已写入 {outdir}/rewrites_verify.json")
    json.dump(verify, open(outdir / "rewrites_verify.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n完成 -> {out_path}")


if __name__ == "__main__":
    main()
