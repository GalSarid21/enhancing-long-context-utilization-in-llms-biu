from pydantic import BaseModel
from typing import Optional, List

from src.entities.enums import Metric


class SingleQuestionResult(BaseModel):
    question_id: str
    question: Optional[str] = None
    model_answer: Optional[str] = None
    answers: Optional[List[str]] = None
    score: float
    num_prompt_tokens: int


class SingleExperimentResults(BaseModel):
    # name example: gold_at_0, closedbook, etc.
    name: str
    metric: Metric
    results: List[SingleQuestionResult]
