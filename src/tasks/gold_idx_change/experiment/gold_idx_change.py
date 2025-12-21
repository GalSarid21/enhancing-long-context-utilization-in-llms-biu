import logging
import torch
import json
import os

from xopen import xopen
from typing import Set, Optional
from pathlib import Path
from argparse import Namespace
from datetime import datetime, timezone

from src.metrics import best_subspan_em
from src.wrappers import vLLM
from src.entities.dto import TaskResultsDTO
from src.entities.enums import Metric, Status, PromptingMode
from src.tasks.abstract import AbstractTask
from src.prompt_builder import PromptBuilder
from src.tasks.gold_idx_change.experiment.configs import Configs
from src.entities.experiments.data import SingleQuestionData
from src.entities.experiments.results import SingleExperimentResults, SingleQuestionResult


logger = logging.getLogger(__name__)


class GoldIdxChangeExperiment(AbstractTask):
    
    def __init__(self, args: Namespace) -> None:
        self._configs = Configs()
        super().__init__(args)

        self._dataset_dir: Path = self._base_dir / self._configs.dataset_folder
        if self._prompting_mode in PromptingMode.get_multiple_docs_modes():
            self._dataset_dir = self._dataset_dir / f"num_idxs_{self._configs.num_idxs}" / self._model_short_name
        os.makedirs(self._dataset_dir, exist_ok=True)

        self._res_dir = self._base_dir / self._configs.results_folder / f"num_idxs_{self._configs.num_idxs}" / self._model_short_name
        os.makedirs(self._res_dir, exist_ok=True)

        self._prompt_builder = PromptBuilder(
            prompting_mode=self._prompting_mode,
            tokenizer=self._tokenizer,
            max_tokens=self._configs.max_tokens
        )

        self._llm: vLLM = self._load_llm(args=args)

    def _load_llm(self, args: Namespace) -> vLLM:
        return vLLM(
            model=args.model,
            dtype=self._configs.dtype,
            top_p=self._configs.top_p,
            num_gpus=torch.cuda.device_count(),
            max_tokens=self._configs.max_tokens,
            temperature=self._configs.temperature,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=self._configs.gpu_memory_utilization
        )

    async def run(self) -> TaskResultsDTO:
        logger.info(f"run - started: {self._dataset_dir=}")
        
        try:
            if self._prompting_mode in PromptingMode.get_multiple_docs_modes():
                existing_res_files = await self._get_results_dir_files(dir=self._res_dir, return_file_names_only=True)
                processing_files = await self._get_results_dir_files(dir=self._dataset_dir)

                for file_path in processing_files:
                    if file_path.name in existing_res_files:
                        logger.info(f"run - skipping existing file: {file_path.name=}")

                    else:
                        logger.info(f"run - processing: {file_path.name=}")
                        res = await self._process_single_dataset(dataset_path=file_path)
                        await self._log_single_idx_data(idx_data=res)
            else:
                file_path = self._dataset_dir / f"{self._prompting_mode.value}.jsonl.gz"
                logger.info(f"run - processing: {file_path=}")
                res = await self._process_single_dataset(dataset_path=file_path)
                await self._log_single_idx_data(idx_data=res)

            res_dto = TaskResultsDTO(status=Status.SUCCESS)
            logger.info(f"run - finished: {res_dto=}")
            return res_dto
        
        except Exception as e:
            logger.exception(f"run - failed: {e}")
            return TaskResultsDTO(status=Status.FAILURE, error=str(e))
    
    async def _process_single_dataset(self, dataset_path: Path) -> SingleExperimentResults:
        prompts = []
        single_idx_data = []
        
        with xopen(dataset_path, "rt") as f:
            for line in f:
                payload = json.loads(line)
                single_question_data = SingleQuestionData(**payload)
                single_idx_data.append(single_question_data)
                
                prompt = await self._prompt_builder.build(question=single_question_data.question,
                                                          documents=single_question_data.documents)
                prompts.append(prompt)
        
        preds = await self._llm.generate_batch(prompts=prompts)
        
        sigle_question_res_list = []
        for question_data, pred, prompt in zip(single_idx_data, preds, prompts):

            res = SingleQuestionResult(question_id=question_data.question_id,
                                       question=question_data.question,
                                       answers=question_data.answers,
                                       model_answer=pred,
                                       score=best_subspan_em(prediction=pred, ground_truths=question_data.answers),
                                       num_prompt_tokens=await self._tokenizer.count_tokens(prompt=prompt))
            
            sigle_question_res_list.append(res)
        
        name = dataset_path.as_posix().split("/")[-1].split(".")[0]
        return SingleExperimentResults(name=name, metric=Metric.BEST_SUBSPAN_EM, results=sigle_question_res_list)
        
    async def _log_single_idx_data(self, idx_data: SingleExperimentResults) -> None:
        res_file_name = f"{idx_data.name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl.gz"
        res_path = self._res_dir / res_file_name
        logger.info(f"_log_single_idx_data - logging res file: {res_path=}")

        with xopen(res_path, "wt") as f:
            for data in idx_data.results:
                f.write(json.dumps(data.model_dump(), ensure_ascii=False) + "\n")
    
    async def _get_results_dir_files(self, dir: Path, return_file_names_only: Optional[bool] = False) -> Set:
        existing_files = [
            f"{file.name.rsplit('_', 1)[0]}.jsonl.gz"
            if return_file_names_only else file
            for file in dir.iterdir()
            if file.is_file() and file.name.endswith(".jsonl.gz")
        ]
        logger.info(f"run - found {len(existing_files)} files: {dir=}, {existing_files=}")
        return set(existing_files)
