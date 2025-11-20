from abc import ABC, abstractmethod


class AbstractTask(ABC):

    @abstractmethod
    async def load_data(self) -> 
