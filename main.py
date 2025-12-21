import logging
import asyncio
import json
import sys

from argparse import Namespace

from src.args import read_cli_env_args
from src.entities.dto import TaskResultsDTO
from src.entities.enums import TaskType, TaskName
from src.tasks.abstract import AbstractTask
from src.tasks.gold_idx_change import GoldIdxChangeDatasetCreation, GoldIdxChangeExperiment
from src.tasks.num_docs_increnet import NumDocsIncrementExperiment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)


logger = logging.getLogger(__name__)
PROG_NAME = "Enhancing Long-Context Utilization in LLMs"


async def main(args: Namespace) -> None:
    logger.info("main - started")

    task: AbstractTask = await _get_running_task(task_type=args.task_type,
                                                 task_name=args.task_name,
                                                 args=args)
    logger.info(f"main - created task obj: {task=}")

    res_dto: TaskResultsDTO = await task.run()
    logger.info(f"main - task running completed: {res_dto=}")


async def _get_running_task(task_type: str, task_name: str, args: Namespace) -> AbstractTask:
    task_type = TaskType(task_type)
    task_name = TaskName(task_name)

    if task_name == TaskName.GOLD_IDX_CHANGE:
        if task_type == TaskType.DATASET_CREATION:
            return GoldIdxChangeDatasetCreation(args=args)
        if task_type == TaskType.EXPERIMENT:
            return GoldIdxChangeExperiment(args=args)
    
    if task_name == TaskName.NUM_DOCS_INCREMENT:
        if task_type == TaskType.EXPERIMENT:
            return NumDocsIncrementExperiment(args=args)


if __name__ == "__main__":
    try:
        args = read_cli_env_args()
        logger.info(
            f"starting a {PROG_NAME} run with the following args:\n"
            f"{json.dumps(vars(args), indent=2)}"
        )
        asyncio.run(main(args=args))
    
    except KeyboardInterrupt:
        logger.info("the run was interuppted by the user. exit gracefully.")
        sys.exit(130)
    
    except Exception as e:
        logger.exception(msg=str(e))
        sys.exit(1)
