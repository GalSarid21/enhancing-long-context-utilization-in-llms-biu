from pydantic import BaseModel
from typing import Dict

from src.entities.enums import Status


class TaskResultsDTO(BaseModel):
    status: Status
    results: Dict
