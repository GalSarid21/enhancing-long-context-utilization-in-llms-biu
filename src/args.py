from argparse import ArgumentParser, Namespace

from src.entities.enums import PromptingMode
from src.entities.enums import TaskType, TaskName


def read_cli_env_args() -> Namespace:
    parser = ArgumentParser("")

    parser.add_argument(
        "--task_type",
        help="task type to run",
        choices=[TaskType.DATASET_CREATION.value, TaskType.EXPERIMENT.value],
        type=str
    )

    parser.add_argument(
        "--task_name",
        help="task name to run",
        choices=[name for name in TaskName],
        type=str
    )

    parser.add_argument(
        "--model",
        help="HF model repo to use in experiment",
        type=str
    )

    parser.add_argument(
        "--prompting_mode",
        help="prompting type to use in experiment",
        type=str
    )

    parser.add_argument(
        "--base_dir",
        help="app base dir",
        type=str
    )

    parser.add_argument(
        "--max_model_len",
        help="maximum tokens the model can get as input",
        type=int
    )

    parser.add_argument(
        "--gold_location",
        help="gold doc relative location in increase number of documents experiment",
        type=int
    )

    return parser.parse_args()
