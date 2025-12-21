from pathlib import Path


NQ_DATASET_FILE_PATH = Path("./data/nq-open-with-uuid-367.jsonl.gz")
MAX_DOCS = 1000
MAX_DOC_LEN = 510
SYSTEM_LEN = 2000
MODEL_DOCS_MAPPINGS = {
    "gemma-3-4b-it": 685,
    "Llama-3.2-3B-Instruct": 695,
    "Qwen3-4B-Instruct-2507": 670
}
