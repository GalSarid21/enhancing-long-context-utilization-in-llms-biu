import torch

from argparse import Namespace
from datetime import datetime, timezone

from src.wrappers import HfTokenizer, vLLM
from src.tasks.abstract import AbstractTask
from src.entities.enums import PromptingMode
from src.prompt_builder import PromptBuilder
from src.tasks.gold_idx_change_experimant.configs import Configs
from src.entities.experiments.gold_idx_change.data import GoldIdxChangeExperimentData
from src.entities.experiments.gold_idx_change.results import GoldIdxChangeExperimentResults


class GoldIdxChangeExperiment(AbstractTask):
    
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self._configs = Configs()
        
        self._prompting_mode = PromptingMode(args.prompting_mode)
        self._tokenizer = HfTokenizer(args.model)

        self._prompt_builder = PromptBuilder(
            prompting_mode=self._prompting_mode,
            tokenizer=self._tokenizer
        )

        self._llm = self._load_llm(args=args)

        self._results_file_name = self._set_results_file_name()

    def _load_llm(self, args: Namespace) -> vLLM:
        return vLLM(
            model=args.model,
            dtype=self._configs.dtype,
            top_p=self._configs.top_p,
            num_gpus=torch.cuda.device_count(),
            max_tokens=args.max_tokens,
            temperature=self._configs.temperature,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=self._configs.gpu_memory_utilization
        )

    def _set_results_file_name(self) -> str:
        f"gold_idx_change_{datetime.now().strftime("%Y%m%d")}_{datetime.now(timezone.utc).timestamp()}.json"

    @property
    def results_dir(self) -> str:
        return self._configs.results_dir

    @property
    def results_file_name(self) -> str:
        return self._results_file_name

    async def load_data(self) -> GoldIdxChangeExperimentData:
        return super().load_data()
    
    async def run(self, data: GoldIdxChangeExperimentData) -> GoldIdxChangeExperimentResults:
        return super().run()
