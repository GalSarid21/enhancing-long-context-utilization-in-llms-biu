import pytest
import json

from assertpy import assert_that
from pathlib import Path
from typing import List, Dict, Optional

from src.entities.experiments.gold_idx_change.data import SingleQuestionRawData, GoldIdxChangeExperimentData
from src.entities.enums import PromptingMode
from src.helpers.nq_data import read_data_file, get_golden_idx_change_data


class TestNqData:
    @pytest.mark.asyncio
    async def test_read_data_file_openbook(self) -> None:
        await self._test_read_data_file(prompting_mode="openbook")
    
    @pytest.mark.asyncio
    async def test_read_data_file_closedbook(self) -> None:
        await self._test_read_data_file(prompting_mode="closedbook")
    
    @pytest.mark.asyncio
    async def test_read_data_file_baseline(self) -> None:
        await self._test_read_data_file(prompting_mode="baseline")

    async def _test_read_data_file(self, prompting_mode: str, num_examples: Optional[int] = 3) -> None:
        raw_data: List[SingleQuestionRawData] = await read_data_file(prompting_mode=PromptingMode(prompting_mode), num_examples=num_examples)        
        src_data_payload: Dict = await self._read_results_data(file_name=f"read_data_file_{prompting_mode}.json")
        src_data = [SingleQuestionRawData(**payload) for payload in src_data_payload]
        assert_that(raw_data).is_equal_to(src_data)

    @pytest.mark.asyncio
    async def test_get_golden_idx_change_data_openbook(self) -> None:
        await self._test_get_golden_idx_change_data(prompting_mode="openbook")

    
    @pytest.mark.asyncio
    async def test_get_golden_idx_change_data_closedbook(self) -> None:
        await self._test_get_golden_idx_change_data(prompting_mode="closedbook", num_idxs=None)
    
    @pytest.mark.asyncio
    async def test_get_golden_idx_change_data_baseline(self) -> None:
        await self._test_get_golden_idx_change_data(prompting_mode="baseline", num_idxs=None)

    async def _test_get_golden_idx_change_data(
        self,
        prompting_mode: str,
        model_name: Optional[str] = "Qwen/Qwen3-4B-Instruct-2507",
        num_idxs: Optional[int] = 3,
        path: Optional[str] = None,
        num_examples: Optional[int] = 5
    ) -> None:
    
        raw_data: GoldIdxChangeExperimentData = await get_golden_idx_change_data(prompting_mode=PromptingMode(prompting_mode),
                                                                                 num_examples=num_examples,
                                                                                 model_name=model_name,
                                                                                 num_idxs=num_idxs,
                                                                                 path=path)      
        src_data_payload: Dict = await self._read_results_data(file_name=f"get_golden_idx_change_data_{prompting_mode}.json")
        src_data = GoldIdxChangeExperimentData(**src_data_payload)
        assert_that(raw_data).is_equal_to(src_data)

    async def _read_results_data(self, file_name: str) -> Dict:
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        data_path = current_dir / "results" / file_name

        with open(data_path, "r") as f:
            data = json.load(f)
        
        return data
