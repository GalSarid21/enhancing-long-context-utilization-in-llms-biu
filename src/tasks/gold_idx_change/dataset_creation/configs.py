from pydantic import BaseModel
from pathlib import Path


class Configs(BaseModel):
    num_idxs: int = 30
    dataset_base_dir: Path = Path("./data/datasets")