import logging
import torch

from pathlib import Path
from argparse import Namespace
from typing import Optional, Set
from abc import ABC, abstractmethod

from src.entities.enums import PromptingMode
from src.entities.base import BaseDataClass
from src.entities.dto import TaskResultsDTO
from src.wrappers import HfTokenizer


logger = logging.getLogger(__name__)


class AbstractTask(ABC):

    def __init__(self, args: Namespace) -> None:
        self._prompting_mode = PromptingMode(args.prompting_mode)
        
        self._model: str = args.model
        self._model_short_name: str = self._model.split("/")[-1]

        base_dir_str = args.base_dir or ".//"
        self._base_dir = Path(base_dir_str)

        self._tokenizer = HfTokenizer(model=self._model)

        self._log_env_resources()

    def _log_env_resources(self) -> None:
        if torch.cuda.is_available():
            msg = f"HW type: GPU | HW name: {torch.cuda.get_device_name(0)}"
        else:
            import platform
            msg = f"HW type: CPU | HW name: {platform.processor()}"
        logger.info(msg)

    @abstractmethod
    async def run(self, data: BaseDataClass) -> TaskResultsDTO:
        pass

    async def _get_results_dir_files(self, dir: Path, return_file_names_only: Optional[bool] = False) -> Set:
        existing_files = [
            f"{file.name.rsplit('_', 1)[0]}.jsonl.gz"
            if return_file_names_only else file
            for file in dir.iterdir()
            if file.is_file() and file.name.endswith(".jsonl.gz")
        ]
        logger.info(f"run - found {len(existing_files)} files: {dir=}, {existing_files=}")
        return set(existing_files)
