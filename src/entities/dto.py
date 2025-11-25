from pydantic import BaseModel

from src.entities.enums import Status
from src.entities.base import BaseTaskResults


class TaskResultsDTO(BaseModel):
    status: Status
    results: BaseTaskResults
