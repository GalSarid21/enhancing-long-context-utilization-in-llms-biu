from pydantic import BaseModel


class Configs(BaseModel):
    num_idxs: int = 30
    dataset_folder: str = "data/datasets"
