from pydantic import BaseModel
from typing import Optional, List

from src.entities.enums import PromptingMode, Metric


class SingleQuestionResult(BaseModel):
    question_id: str
    question: Optional[str] = None
    model_answer: Optional[str] = None
    answers: Optional[List[str]] = None
    score: float
    num_prompt_tokens: int


class SingleIdxResults(BaseModel):
    # name example: gold_at_0, closedbook, etc.
    name: str
    metric: Metric
    results: List[SingleQuestionResult]


class GoldIdxChangeExperimentResults(BaseModel):
    model: str
    num_documents: int
    prompting_mode: PromptingMode
    experiments: List[SingleIdxResults]
    start_time: Optional[str] = None
    end_time: Optional[str] = None
