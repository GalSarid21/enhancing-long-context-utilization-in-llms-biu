import logging
import torch
import json
import os

from datetime import datetime, timezone
from argparse import Namespace
from typing import List
from xopen import xopen
from copy import deepcopy

from common.consts import MAX_DOCS, MAX_DOC_LEN, SYSTEM_LEN
from src.metrics import best_subspan_em
from src.wrappers import vLLM
from src.entities.dto import TaskResultsDTO
from src.entities.enums import Metric, Status, GoldLocation
from src.tasks.abstract import AbstractTask
from src.prompt_builder import PromptBuilder
from src.helpers.nq_data import read_data_file
from src.entities.experiments.data import SingleQuestionRawData
from src.entities.experiments.results import SingleQuestionResult, SingleExperimentResults
from src.tasks.num_docs_increnet.experiment.configs import Configs


logger = logging.getLogger(__name__)


class NumDocsIncrementExperiment(AbstractTask):
    
    def __init__(self, args: Namespace) -> None:
        self._configs = Configs()
        super().__init__(args)

        self._gold_location = GoldLocation(args.gold_location)

        self._res_dir = (
            self._base_dir / 
            self._configs.results_folder / 
            f"doc_step_{self._configs.docs_step_size}" /
            self._model_short_name /
            f"gold_location_{self._gold_location}"
        )
        os.makedirs(self._res_dir, exist_ok=True)

        self._prompt_builder = PromptBuilder(
            prompting_mode=self._prompting_mode,
            tokenizer=self._tokenizer,
            max_tokens=self._configs.max_tokens
        )
    
    async def run(self) -> TaskResultsDTO:
        logger.info(f"run - started: {self._dataset_dir=}")
        try:
            raw_data: List[SingleQuestionRawData] = await read_data_file(prompting_mode=self._prompting_mode)
    
            for n_docs in range(0, MAX_DOCS, self._configs.docs_step_size):
                logger.info(f"run - start processing {n_docs} docs")

                llm = await self._load_llm(n_docs=n_docs)

                prompts = []
                for data in raw_data:
                    gold_doc = data.gold_docs[0]
                    docs = data.documents[:n_docs]
                    docs_cpy = deepcopy(docs)

                    insert_idx = None
                    if self._gold_location == GoldLocation.START:
                        insert_idx = 0
                    elif self._gold_location == GoldLocation.END:
                        insert_idx = len(docs_cpy)
                    else:
                        insert_idx = len(docs_cpy) // 2
                    docs_cpy.insert(insert_idx, gold_doc)

                    prompt = await self._prompt_builder.build(question=data.question, documents=docs_cpy)
                    prompts.append(prompt)
                
                preds = await llm.generate_batch(prompts=prompts)

                sigle_question_res_list = []
                for question_data, pred, prompt in zip(raw_data, preds, prompts):

                    res = SingleQuestionResult(question_id=question_data.question_id,
                                               question=question_data.question,
                                               answers=question_data.answers,
                                               model_answer=pred,
                                               score=best_subspan_em(prediction=pred, ground_truths=question_data.answers),
                                               num_prompt_tokens=await self._tokenizer.count_tokens(prompt=prompt))
                    
                    sigle_question_res_list.append(res)

                single_experiment_results = SingleExperimentResults(
                    name=f"num_docs_{n_docs}",
                    metric=Metric.BEST_SUBSPAN_EM,
                    results=sigle_question_res_list
                )

                await self._log_single_idx_data(experiment_res=single_experiment_results)

            res_dto = TaskResultsDTO(status=Status.SUCCESS)
            logger.info(f"run - finished: {res_dto=}")
            return res_dto

        except Exception as e:
            logger.exception(f"run - failed: {e}")
            return TaskResultsDTO(status=Status.FAILURE, error=str(e))
    
    async def _load_llm(self, n_docs: int) -> vLLM:
        max_model_len = (n_docs * MAX_DOC_LEN) + SYSTEM_LEN
        
        return vLLM(
            model=self._model,
            dtype=self._configs.dtype,
            top_p=self._configs.top_p,
            num_gpus=torch.cuda.device_count(),
            max_tokens=self._configs.max_tokens,
            temperature=self._configs.temperature,
            max_model_len=max_model_len,
            gpu_memory_utilization=self._configs.gpu_memory_utilization
        )
    
    async def _log_single_idx_data(self, experiment_res: SingleExperimentResults) -> None:
        res_file_name = f"{experiment_res.name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl.gz"
        res_path = self._res_dir / res_file_name
        logger.info(f"_log_single_idx_data - logging res file: {res_path=}")

        with xopen(res_path, "wt") as f:
            for data in experiment_res.results:
                f.write(json.dumps(data.model_dump(), ensure_ascii=False) + "\n")