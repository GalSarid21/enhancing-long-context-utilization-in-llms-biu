import json

from pathlib import Path
from typing import List, Optional, Dict
from xopen import xopen


def _read_ids_list(path: Optional[Path] = None) -> List[str]:
    path = path or Path("./data/ids/common_cross.json")
    print(f"_read_ids_list - started: {path=}")
    
    with xopen(path, "rt") as f:
        ids = json.load(f)
    
    print(f"_read_ids_list - finished: {ids=}")
    return ids


def _read_filtered_dataset(ids: List[str], path: Optional[Path] = None) -> List[Dict]:
    path = path or Path("./data/nq-open-with-uuid.jsonl.gz")
    print(f"_read_filtered_dataset - started: {ids=}, {path=}")
    
    filtered_data = []
    with xopen(path, "rt") as f:
        for line in f:
            line = line.strip()
            line = json.loads(line)
            if line["id"] in ids:
                filtered_data.append(line)
    
    print(f"_read_filtered_dataset - finished: {len(filtered_data)=}")
    return filtered_data


def _export_filtered_data(filtered_data: List[Dict], path: Optional[Path] = None) -> None:
    path = path or Path("./data/nq-open-with-uuid-367.jsonl.gz")
    print(f"_export_filtered_data - started: {len(filtered_data)=}, {path=}")
    
    with xopen(path, "wt") as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"_export_filtered_data - finished")


def run() -> None:
    ids = _read_ids_list()
    filtered_data = _read_filtered_dataset(ids=ids)
    _export_filtered_data(filtered_data=filtered_data)


if __name__ == "__main__":
    run()