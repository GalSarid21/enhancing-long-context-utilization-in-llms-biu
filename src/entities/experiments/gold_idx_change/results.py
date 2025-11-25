from pydantic import BaseModel
from typing import Optional, List

from src.entities.base import BaseTaskResults
from src.entities.enums import PromptingMode, Metric


class SingleQuestionResult(BaseModel):
    question_id: str
    model_answer: str
    score: float
    num_prompt_tokens: int


class SingleIdxResults(BaseModel):
    # name example: gold_at_0, closedbook, etc.
    name: str
    metric: Metric
    results: List[SingleQuestionResult]


class GoldIdxChangeExperimentResults(BaseTaskResults):
    model: str
    num_documents: int
    prompting_mode: PromptingMode
    experiments: List[SingleIdxResults]
    start_time: Optional[str] = None
    end_time: Optional[str] = None
