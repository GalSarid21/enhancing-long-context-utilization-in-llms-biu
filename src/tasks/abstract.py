import logging
import torch
import os

from pathlib import Path
from argparse import Namespace
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

        self._dataset_dir: Path = self._base_dir / self._configs.dataset_folder
        if self._prompting_mode in PromptingMode.get_multiple_docs_modes():
            self._dataset_dir = self._dataset_dir / f"num_idxs_{self._configs.num_idxs}" / self._model_short_name
        os.makedirs(self._dataset_dir, exist_ok=True)

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
