from pydantic import BaseModel
from typing import TypeVar, Optional, Type, Dict, Any
from copy import deepcopy


T = TypeVar("T")


# Adapted from the nelson-liu/lost-in-the-middle repository
# Original source: https://github.com/nelson-liu/lost-in-the-middle
# Licensed under the MIT License
# The pydantic.dataclass type was changed to pydantic.BaseModel
class Document(BaseModel):
    title: str
    text: str
    id: str
    score: float
    hasanswer: bool
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))

    def to_dict(self) -> Dict[str, Any]:
        field_names = list(self.__annotations__.keys())
        return {field: getattr(self, field) for field in field_names}
