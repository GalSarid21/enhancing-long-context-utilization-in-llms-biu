import asyncio
import json
import sys

from pathlib import Path
from typing import List, Optional, Dict
from xopen import xopen

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.wrappers import HfTokenizer


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


async def _get_model_num_docs(models: List[str], data: List[Dict]) -> Dict:
    max_tokens_tresh = 0.98 * 131_072
    print(f"_get_model_num_docs - started: {max_tokens_tresh=}")
    models_min_docs = {}

    for model in models:
        print(f"_get_model_num_docs - processing: {model=}")
        tok = HfTokenizer(model=model)
        min_docs = 1000

        for d in data:
            total_tokens = 0
            for doc_cnt, ctx in enumerate(d["ctxs"], 1):
                tokens_cnt = await tok.count_tokens(prompt=f"Document [{doc_cnt}](Title: {ctx['title']}) {ctx['text']}\n")
                total_tokens += tokens_cnt
                if total_tokens >= max_tokens_tresh:
                    break
            if doc_cnt < min_docs:
                min_docs = doc_cnt
        
        print(f"_get_model_num_docs - {model} finished: {min_docs=}")
        models_min_docs[model] = min_docs

    print(f"_get_model_num_docs - finished: {models_min_docs=}")
    return models_min_docs


def run() -> None:
    ids: List[str] = _read_ids_list()
    filtered_data: List[Dict] = _read_filtered_dataset(ids=ids)
    _export_filtered_data(filtered_data=filtered_data)

    models = ["google/gemma-3-4b-it", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"]
    asyncio.run(
        _get_model_num_docs(models=models, data=filtered_data)
    )


if __name__ == "__main__":
    run()
