from typing import Any
from abc import ABC, abstractmethod


class AbstractTask(ABC):

    @abstractmethod
    async def load_data(self) -> Any:
        pass

    @abstractmethod
    async def run(self) -> Any:
        pass
