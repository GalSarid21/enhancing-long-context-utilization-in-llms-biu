import logging
import torch
import json

from argparse import Namespace
from abc import ABC, abstractmethod

from src.entities.enums import PromptingMode
from src.entities.base import BaseDataClass, BaseTaskResults
from src.entities.dto import TaskResultsDTO
from src.wrappers import HfTokenizer


logger = logging.getLogger(__name__)


class AbstractTask(ABC):

    def __init__(self, args: Namespace) -> None:
        self._results_dir = None
        self._results_file_name = None

        self._prompting_mode = PromptingMode(args.prompting_mode)
        self._tokenizer = HfTokenizer(args.model)

        self._log_env_resources()

    def _log_env_resources(self) -> None:
        if torch.cuda.is_available():
            msg = f"HW type: GPU | HW name: {torch.cuda.get_device_name(0)}"
        else:
            import platform
            msg = f"HW type: CPU | HW name: {platform.processor()}"
        logger.info(msg)

    @abstractmethod
    async def load_data(self) -> BaseDataClass:
        pass

    @abstractmethod
    async def run(self, data: BaseDataClass) -> TaskResultsDTO:
        pass

    async def log_results(self, results: BaseTaskResults) -> None:
        res_path = f"{self._results_dir}/{self._results_file_name}"
        with open(res_path, "w") as f:
            json.dump(f, results.model_dump(), indent=2)
