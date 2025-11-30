import json
import uuid
import sys

from pathlib import Path
from typing import List, Dict
from xopen import xopen

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from common.consts import NQ_DATASET_FILE_PATH


def _count_lines(path: Path) -> int:
    total = 0
    with xopen(path, "rt") as f:
        for total, _ in enumerate(f, 1):
            pass
    return total


def _create_new_data(data_path: Path) -> List[Dict]:
    total_lines = _count_lines(path=data_path)
    print(f"start creating new data: {total_lines=}")

    data_with_uuid = []
    with xopen(data_path, "rt") as f:
        for i, line in enumerate(f, 1):
            print(f"start procesing {i}/{total_lines}")
            
            line = line.strip()
            entry = json.loads(line)

            hasanswer = False
            for ctx in entry["ctxs"]:
                if ctx["hasanswer"]:
                    hasanswer = True
                    break

            if not hasanswer:
                print(f"Skipping {i}: {hasanswer=}")
                continue

            entry["id"] = str(uuid.uuid4())
            data_with_uuid.append(entry)
    
    print(f"new data creating is done: {len(data_with_uuid)=}")
    return data_with_uuid


def _save_new_data(new_data: List[Dict], out_path: Path) -> None:
    print(f"start saving new data: {out_path=}")
    
    with xopen(out_path, "wt") as f:
        for i, entry in enumerate(new_data, 1):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Progress: {i}/{len(new_data)}")
    
    print(f"new data saving is done")


def _validate_results(out_path: Path) -> bool:
    if not out_path.exists():
        print(f"File does not exist: {out_path}")
        return

    size_bytes = out_path.stat().st_size

    if size_bytes == 0:
        print(f"File exists but is EMPTY: {out_path}")
        return

    size_mb = size_bytes / (1024 * 1024)

    print(f"file exists: {out_path}")
    print(f"compressed size: {size_bytes:,} bytes ({size_mb:.2f} MB)")


def run() -> None:
    data_path = Path("./data/nq-open-contriever-msmarco-retrieved-documents.jsonl")
    print(f"add_uuid_to_base_data - starting: {data_path=}")

    data_with_uuid = _create_new_data(data_path=data_path)

    out_path = NQ_DATASET_FILE_PATH
    _save_new_data(new_data=data_with_uuid, out_path=out_path)

    _validate_results(out_path=out_path)


if __name__ == "__main__":
    run()