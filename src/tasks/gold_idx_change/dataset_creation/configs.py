from pydantic import BaseModel


class Configs(BaseModel):
    num_idxs: int = 20
    dataset_folder: str = "data/datasets"
