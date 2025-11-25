import logging
import torch

from abc import ABC, abstractmethod

from src.entities.base import BaseDataClass, BaseTaskResults


logger = logging.getLogger(__name__)


class AbstractTask(ABC):

    def __init__(self) -> None:
        self._log_env_resources()

    def _log_env_resources(self) -> None:
        if torch.cuda.is_available():
            msg = f"HW type: GPU | HW name: {torch.cuda.get_device_name(0)}"
        else:
            import platform
            msg = f"HW type: CPU | HW name: {platform.processor()}"
        logger.info(msg)

    @abstractmethod
    @property
    def results_dir(self) -> str:
        pass

    @abstractmethod
    @property
    def results_file_name(self) -> str:
        pass

    @abstractmethod
    async def load_data(self) -> BaseDataClass:
        pass

    @abstractmethod
    async def run(self) -> BaseTaskResults:
        pass
