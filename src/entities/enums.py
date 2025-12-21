from typing import List, Generator
from enum import Enum


class TaskType(str, Enum):
    EXPERIMENT = "experiment"
    DATASET_CREATION = "dataset_creation"


class TaskName(str, Enum):
    GOLD_IDX_CHANGE = "gold_idx_change"
    NUM_DOCS_INCREMENT = "num_docs_increment"


class PromptingMode(str, Enum):
    CLOSEDBOOK = "closedbook"
    OPENBOOK = "openbook"
    OPENBOOK_RANDOM = "openbook_random"
    # baseline is a prompt with the question and the golden-document only
    BASELINE = "baseline"

    @staticmethod
    def get_ctx_modes() -> List["PromptingMode"]:
        return [PromptingMode.OPENBOOK, PromptingMode.OPENBOOK_RANDOM, PromptingMode.BASELINE]
    
    @staticmethod
    def get_multiple_docs_modes() -> List["PromptingMode"]:
        return [PromptingMode.OPENBOOK, PromptingMode.OPENBOOK_RANDOM]


class Metric(str, Enum):
    BEST_SUBSPAN_EM = "best_subspan_em"


class Status(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class GoldLocation(str, Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"
