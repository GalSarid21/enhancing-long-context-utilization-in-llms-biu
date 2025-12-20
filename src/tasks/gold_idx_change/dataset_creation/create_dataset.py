import json
import logging

from argparse import Namespace
from typing import AsyncIterator
from xopen import xopen

import src.helpers.nq_data as nq_helper

from src.entities.dto import TaskResultsDTO
from src.tasks.abstract import AbstractTask
from src.entities.enums import Status
from src.entities.experiments.gold_idx_change.data import SingleIdxData
from src.tasks.gold_idx_change.dataset_creation.configs import Configs


logger = logging.getLogger(__name__)


class GoldIdxChangeDatasetCreation(AbstractTask):

    def __init__(self, args: Namespace):
        self._configs = Configs()
        super().__init__(args)

    async def run(self) -> TaskResultsDTO:
        logger.info(f"run - started: {self._dataset_dir=}")
    
        try:
            gold_idx_data = await nq_helper.get_golden_idx_change_data(prompting_mode=self._prompting_mode,
                                                                       model_name=self._model,
                                                                       num_idxs=self._configs.num_idxs)
            if isinstance(gold_idx_data, AsyncIterator):
                async for idx_data in gold_idx_data:
                    await self._log_single_idx_data(idx_data=idx_data)
            else:
                await self._log_single_idx_data(idx_data=gold_idx_data[0])

            res_dto = TaskResultsDTO(status=Status.SUCCESS)
            logger.info(f"run - finished: {res_dto=}")
            return res_dto

        except Exception as e:
            logger.exception(f"run - failed: {e}")
            return TaskResultsDTO(status=Status.FAILURE, error=str(e))
    
    async def _log_single_idx_data(self, idx_data: SingleIdxData) -> None:
        dataset_path = self._dataset_dir / f"{idx_data.name}.jsonl.gz"
        logger.info(f"_log_single_idx_data - logging datset: {dataset_path=}")

        with xopen(dataset_path, "wt") as f:
            for data in idx_data.data:
                f.write(json.dumps(data.model_dump(), ensure_ascii=False) + "\n")
