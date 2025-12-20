import json
import sys
import re
import os

from pathlib import Path
from typing import Dict, List, Iterator
from xopen import xopen

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def extract_gold_idx(filename: str) -> int:
    m = re.search(r"gold_at_(\d+)_", filename)
    return int(m.group(1)) if m else None


def read_jsonl(path: Path) -> Iterator[Dict]:
    with xopen(path, "r") as f:
        for line in f:
            yield json.loads(line)


def build_summary(folder_path: str) -> List[Dict]:
    folder = Path(folder_path)
    results = []

    for file_path in folder.glob("gold_at_*.jsonl.gz"):
        gold_idx = extract_gold_idx(file_path.name)

        scores = []
        tokens = []

        for record in read_jsonl(file_path):
            scores.append(record["score"])
            tokens.append(record["num_prompt_tokens"])

        results.append({
            "file": file_path.name,
            "gold_idx": gold_idx,
            "avg_score": sum(scores) / len(scores),
            "avg_num_prompt_tokens": sum(tokens) / len(tokens),
            "num_rows": len(scores),
        })

    return sorted(results, key=lambda x: x["gold_idx"])


def run() -> None:
    summary: List[Dict] = build_summary(folder_path="./results/gold_idx_change/num_idxs_20/gemma-3-4b-it")
    
    out_dir = Path("./data/summaries/gold_idx_change/num_idxs_20/gemma-3-4b-it")
    os.makedirs(out_dir, exist_ok=True)

    out_file = out_dir / "summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    run()
