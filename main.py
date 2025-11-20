import logging
import asyncio
import json
import sys

from argparse import Namespace


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)


logger = logging.getLogger(__name__)
PROG_NAME = "Enhancing Long-Context Utilization in LLMs"


async def main(args: Namespace) -> None:
    pass


if __name__ == "__main__":
    try:
        args = None
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

