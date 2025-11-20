from enum import Enum


class TaskType(Enum, str):
    GOLD_IDX_CHANGE_EXPERIMENT = "gold_idx_change_experiment"
    DATASET_CREATION = "dataset_creation"


class PromptingMode(Enum, str):
    CLOSEDBOOK = "closedbook"
    OPENBOOK = "openbook"
    OPENBOOK_RANDOM = "openbook_random"
    # baseline is a prompt with the question and the golden-document only
    BASELINE = "baseline"
