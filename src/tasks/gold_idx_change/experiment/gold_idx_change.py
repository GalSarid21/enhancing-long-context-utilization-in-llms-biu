import torch
import json

from xopen import xopen
from pathlib import Path
from argparse import Namespace
from datetime import datetime, timezone

from src.metrics import best_subspan_em
from src.wrappers import vLLM, HfTokenizer
from src.entities.dto import TaskResultsDTO
from src.tasks.abstract import AbstractTask
from src.prompt_builder import PromptBuilder
from src.tasks.gold_idx_change.experiment.configs import Configs
from src.entities.experiments.gold_idx_change.data import SingleQuestionData
from src.entities.experiments.gold_idx_change.results import SingleIdxResults, SingleQuestionResult


class GoldIdxChangeExperiment(AbstractTask):
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._configs = Configs()

        self._tokenizer = HfTokenizer(model=self._model)

        self._prompt_builder = PromptBuilder(
            prompting_mode=self._prompting_mode,
            tokenizer=self._tokenizer
        )

        self._llm = self._load_llm(args=args)

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

    def _get_results_file_name(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d')}_{datetime.now(timezone.utc).timestamp()}.json"

    async def run(self) -> TaskResultsDTO:
        return super().run()
    
    async def _process_single_dataset(self, dataset_path: Path) -> SingleIdxResults:
        prompts = []
        single_idx_data = []
        
        with xopen(dataset_path, "rt") as f:
            for line in f:
                payload = json.loads(line)
                single_question_data = SingleQuestionData(**payload)
                prompt = await self._prompt_builder.build(question=single_question_data.question,
                                                          documents=single_question_data.documents)
                prompts.append(prompt)
                single_idx_data.append(single_question_data)
        
        preds = await self._llm.generate_batch(prompts=prompts)
        
        sigle_question_res_lise = []
        for question_data, pred, prompt in zip(single_idx_data, preds, prompts):

            res = SingleQuestionResult(question_id=question_data.question_id,
                                       score=best_subspan_em(prediction=pred, ground_truths=question_data.questions),
                                       num_prompt_tokens=await self._tokenizer.count_tokens(prompt))
            
            sigle_question_res_lise.append(res)
        
        return SingleIdxResults()
        

