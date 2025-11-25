import pytest
import json

from assertpy import assert_that
from pathlib import Path
from typing import List, Dict, Optional

from src.entities.experiments.gold_idx_change.data import SingleQuestionRawData, GoldIdxChangeExperimentData
from src.entities.enums import PromptingMode
from src.helpers.nq_data import read_data_file, get_golden_idx_change_data
from src.wrappers import HfTokenizer


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
    async def test_get_golden_idx_change_data_openbook(self, hf_tokenizer: HfTokenizer) -> None:
        await self._test_get_golden_idx_change_data(prompting_mode="openbook",
                                                    hf_tokenizer=hf_tokenizer,
                                                    min_prompt_tokens=115_000)
    
    @pytest.mark.asyncio
    async def test_get_golden_idx_change_data_openbook_bad_init(self, hf_tokenizer: HfTokenizer) -> None:
        with pytest.raises(Exception) as ex_info:
            await self._test_get_golden_idx_change_data(prompting_mode="openbook", hf_tokenizer=hf_tokenizer)
        assert_that(str(ex_info.value)).is_equal_to("tokenizer needs 'min_prompt_tokens' to set number of prompt documents")
    
    @pytest.mark.asyncio
    async def test_get_golden_idx_change_data_closedbook(self) -> None:
        await self._test_get_golden_idx_change_data(prompting_mode="closedbook", golden_idxs=None)
    
    @pytest.mark.asyncio
    async def test_get_golden_idx_change_data_baseline(self) -> None:
        await self._test_get_golden_idx_change_data(prompting_mode="baseline", golden_idxs=None)

    async def _test_get_golden_idx_change_data(
        self,
        prompting_mode: str,
        min_prompt_tokens: Optional[int] = None,
        hf_tokenizer: Optional[HfTokenizer] = None,
        num_examples: Optional[int] = 5,
        golden_idxs: List[int] = [0,2,4]
    ) -> None:
    
        raw_data: GoldIdxChangeExperimentData = await get_golden_idx_change_data(prompting_mode=PromptingMode(prompting_mode),
                                                                                 num_examples=num_examples,
                                                                                 gold_idxs=golden_idxs,
                                                                                 tokenizer=hf_tokenizer,
                                                                                 min_prompt_tokens=min_prompt_tokens)        
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
