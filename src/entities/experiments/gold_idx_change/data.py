from pydantic import BaseModel
from typing import List, Optional

from src.entities.document import Document


class SingleQuestionRawData(BaseModel):
    question_id: str
    question: str
    answers: List[str]
    docuemnts: Optional[List[Document]] = None
    gold_docs: Optional[List[Document]] = None


class SingleQuestionData(BaseModel):
    question_id: str
    question: str
    answers: List[str]
    documents: Optional[List[Document]] = None


class SingleIdxData(BaseModel):
    # name example: gold_at_0, closedbook, etc.
    name: str
    data: List[SingleQuestionData]


class GoldIdxChangeExperimentData(BaseModel):
    experiments: List[SingleIdxData]
