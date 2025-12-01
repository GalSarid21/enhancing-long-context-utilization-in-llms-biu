from pydantic import BaseModel


class Configs(BaseModel):
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.95
    temperature: float = 0.01
    top_p: float = 0.95
    max_tokens: int = 256
    results_folder: str = "results/gold_idx_change"
