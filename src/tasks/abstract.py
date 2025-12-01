import logging
import torch

from pathlib import Path
from argparse import Namespace
from abc import ABC, abstractmethod

from src.entities.enums import PromptingMode
from src.entities.base import BaseDataClass
from src.entities.dto import TaskResultsDTO


logger = logging.getLogger(__name__)


class AbstractTask(ABC):

    def __init__(self, args: Namespace) -> None:
        self._prompting_mode = PromptingMode(args.prompting_mode)
        
        base_dir_str = args.base_dir or ".//"
        self._base_dir = Path(base_dir_str)
        
        self._model = args.model
        self._model_short_name = self._model.split("/")[-1]

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
