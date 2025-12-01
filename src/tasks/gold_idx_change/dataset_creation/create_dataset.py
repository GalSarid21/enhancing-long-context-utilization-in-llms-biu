import os
import json

from typing import Dict
from xopen import xopen

import src.helpers.nq_data as nq_helper

from src.entities.dto import TaskResultsDTO
from src.tasks.abstract import AbstractTask
from src.entities.enums import Status
from src.entities.experiments.gold_idx_change.data import GoldIdxChangeExperimentData
from src.tasks.gold_idx_change.dataset_creation.configs import Configs


class GoldIdxChangeDatasetCreation(AbstractTask):

    def __init__(self, args):
        super().__init__(args)
        self._configs = Configs()

    async def load_data(self) -> GoldIdxChangeExperimentData:
        data = await nq_helper.get_golden_idx_change_data(prompting_mode=self._prompting_mode,
                                                          model_name=self._model,
                                                          num_idxs=self._configs.num_idxs)
        return data
    
    async def run(self, data: GoldIdxChangeExperimentData) -> TaskResultsDTO:
        # results is data since the perpuse of this task is create a dataset file
        return TaskResultsDTO(status=Status.SUCCESS, results=data.model_dump())

    async def log_results(self, results: Dict) -> None:
        dataset_dir = self._base_dir / self._configs.dataset_folder / f"num_idxs_{self._configs.num_idxs}"
        os.makedirs(dataset_dir, exist_ok=True)

        experiments = results.get("experiments")
        if not experiments:
            raise ValueError("GoldIdxChangeDatasetCreation must have 'experiments' entry")

        for experiment in experiments:
            dataset_file_name = experiment["name"]
            dataset_file_path = f"{dataset_file_name}.jsonl.gz"
            
            with xopen(dataset_file_path, "wt") as f:
                f.write(json.dumps(experiment["data"], ensure_ascii=False) + "\n")
