"""解析 MTS-Dialog parquet → 逐轮 (speaker, utterance) 对齐的对话列表。

用法:
    python scripts/parse_mtsdialog.py --parquet-dir /path/to/parquets --out data/mtsdialog_parsed.json

输出 JSON: [ [["doctor","..."],["patient","..."],...], ... ]  (1501 个对话)
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_turns(dialogue: str) -> list[list[str]]:
    turns = []
    for line in dialogue.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(Doctor|Patient|Guest_family|Guest):\s*(.*)", line, re.I)
        if not m:
            continue
        speaker = "doctor" if m.group(1).lower().startswith("doctor") else "patient"
        text = m.group(2).strip()
        if text:
            turns.append([speaker, text])
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", default="data/mtsdialog_parsed.json")
    args = ap.parse_args()

    d = Path(args.parquet_dir)
    df = pd.concat(
        [pd.read_parquet(d / f"{split}.parquet") for split in ["train", "validation", "test"]],
        ignore_index=True,
    )
    parsed = [parse_turns(x) for x in df["dialogue"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    print(f"parsed {len(parsed)} dialogues -> {out}")


if __name__ == "__main__":
    main()
