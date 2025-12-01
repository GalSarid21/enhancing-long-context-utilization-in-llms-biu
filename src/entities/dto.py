from pydantic import BaseModel
from typing import Optional

from src.entities.enums import Status


class TaskResultsDTO(BaseModel):
    status: Status
    error: Optional[str] = None
