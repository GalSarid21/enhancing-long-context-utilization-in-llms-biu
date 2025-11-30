import torch

from argparse import Namespace
from datetime import datetime, timezone

from src.wrappers import vLLM, HfTokenizer
from src.entities.dto import TaskResultsDTO
from src.tasks.abstract import AbstractTask
from src.prompt_builder import PromptBuilder
from src.tasks.gold_idx_change.experiment.configs import Configs
from src.entities.experiments.gold_idx_change.data import GoldIdxChangeExperimentData


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

        self._results_dir = self._configs.results_dir
        self._results_file_name = self._set_results_file_name()

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

    def _set_results_file_name(self) -> str:
        f"gold_idx_change_{datetime.now().strftime('%Y%m%d')}_{datetime.now(timezone.utc).timestamp()}.json"

    @property
    def results_dir(self) -> str:
        return self._configs.results_dir

    @property
    def results_file_name(self) -> str:
        return self._results_file_name

    async def load_data(self) -> GoldIdxChangeExperimentData:
        return super().load_data()
    
    async def run(self, data: GoldIdxChangeExperimentData) -> TaskResultsDTO:
        return super().run()
