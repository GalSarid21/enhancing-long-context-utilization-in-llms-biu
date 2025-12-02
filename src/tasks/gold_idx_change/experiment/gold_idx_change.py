import logging
import torch
import json

from xopen import xopen
from pathlib import Path
from argparse import Namespace
from datetime import datetime, timezone

from src.metrics import best_subspan_em
from src.wrappers import vLLM, HfTokenizer
from src.entities.dto import TaskResultsDTO
from src.entities.enums import Metric, Status, PromptingMode
from src.tasks.abstract import AbstractTask
from src.prompt_builder import PromptBuilder
from src.tasks.gold_idx_change.experiment.configs import Configs
from src.entities.experiments.gold_idx_change.data import SingleQuestionData
from src.entities.experiments.gold_idx_change.results import SingleIdxResults, SingleQuestionResult


logger = logging.getLogger(__name__)


class GoldIdxChangeExperiment(AbstractTask):
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._configs = Configs()

        self._tokenizer = HfTokenizer(model=self._model)

        self._prompt_builder = PromptBuilder(
            prompting_mode=self._prompting_mode,
            tokenizer=self._tokenizer
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
                for file_path in self._dataset_dir.iterdir():
                    if file_path.is_file():
                        res = await self._process_single_dataset(dataset_path=file_path)
                        await self._log_single_idx_data(idx_data=res)
            else:
                res = await self._process_single_dataset(dataset_path=file_path)
                await self._log_single_idx_data(idx_data=res)

            res_dto = TaskResultsDTO(status=Status.SUCCESS)
            logger.info(f"run - finished: {res_dto=}")
        
        except Exception as e:
            logger.exception(f"run - failed: {e}")
            return TaskResultsDTO(status=Status.FAILURE, error=str(e))
    
    async def _process_single_dataset(self, dataset_path: Path) -> SingleIdxResults:
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
                                       score=best_subspan_em(prediction=pred, ground_truths=question_data.questions),
                                       num_prompt_tokens=await self._tokenizer.count_tokens(prompt))
            
            sigle_question_res_list.append(res)
        
        name = dataset_path.as_posix().split("/")[-1].split(".")[0]
        return SingleIdxResults(name=name, metric=Metric.BEST_SUBSPAN_EM, results=sigle_question_res_list)
        
    async def _log_single_idx_data(self, idx_data: SingleIdxResults) -> None:
        res_file_name = f"{idx_data.name}_{idx_data.metric}_{datetime.now(timezone.utc).timestamp()}.jsonl.gz"
        res_path = self._configs.results_folder / res_file_name
        logger.info(f"_log_single_idx_data - logging res file: {res_path=}")

        with xopen(res_path, "wt") as f:
            for data in idx_data.results:
                f.write(json.dumps(data.model_dump(), ensure_ascii=False) + "\n")
