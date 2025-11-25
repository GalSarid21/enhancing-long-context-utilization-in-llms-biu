from typing import List, Generator
from enum import Enum


class TaskType(str, Enum):
    GOLD_IDX_CHANGE_EXPERIMENT = "gold_idx_change_experiment"
    DATASET_CREATION = "dataset_creation"


class PromptingMode(str, Enum):
    CLOSEDBOOK = "closedbook"
    OPENBOOK = "openbook"
    OPENBOOK_RANDOM = "openbook_random"
    # baseline is a prompt with the question and the golden-document only
    BASELINE = "baseline"

    @staticmethod
    def get_ctx_modes() -> List["PromptingMode"]:
        return [PromptingMode.OPENBOOK, PromptingMode.OPENBOOK_RANDOM, PromptingMode.BASELINE]


class Metric(str, Enum):
    BEST_SUBSPAN_EM = "best_subspan_em"
