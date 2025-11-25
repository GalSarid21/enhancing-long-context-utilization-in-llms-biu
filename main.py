import logging
import asyncio
import json
import sys

from argparse import Namespace

from src.args import read_cli_env_args
from src.entities.dto import TaskResultsDTO
from src.entities.base import BaseDataClass
from src.entities.enums import TaskType, TaskName
from src.tasks.abstract import AbstractTask
from src.tasks.gold_idx_change import GoldIdxChangeDatasetCreation, GoldIdxChangeExperiment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)


logger = logging.getLogger(__name__)
PROG_NAME = "Enhancing Long-Context Utilization in LLMs"


async def main(args: Namespace) -> None:
    logger.info("main - started")

    task: AbstractTask = await _get_running_task(task_type=args.task_type, task_name=args.task_name)
    logger.info(f"main - created task obj: {task=}")

    data: BaseDataClass = await task.load_data()
    logger.info(f"main - data was loaded: {data=}")

    res_dto: TaskResultsDTO = await task.run(data=data)
    logger.info(f"main - task running completed: {res_dto=}")

    await task.log_results(results=res_dto.results)
    logger.info(f"main - results file logged successfully")
    logger.info("main - finished")


async def _get_running_task(task_type: str, task_name: str) -> AbstractTask:
    task_type = TaskType(task_type)
    task_name = TaskName(task_name)

    if task_name == TaskName.GOLD_IDX_CHANGE:
        if task_type == TaskType.DATASET_CREATION:
            return GoldIdxChangeDatasetCreation()
        if task_type == TaskType.EXPERIMENT:
            return GoldIdxChangeExperiment()


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
