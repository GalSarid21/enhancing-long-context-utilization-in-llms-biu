import pytest
import sys

from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.entities.experiments.gold_idx_change.data import SingleQuestionRawData, GoldIdxChangeExperimentData
from src.entities.enums import PromptingMode
from common.nq_data import read_data_file, get_golden_idx_change_data


class TestNqData:
    @pytest.mark.asyncio
    async def test_read_data_file_openbook(self) -> None:
        raw_data: List[SingleQuestionRawData] = await read_data_file(prompting_mode=PromptingMode.OPENBOOK, num_examples=10)
        pass
