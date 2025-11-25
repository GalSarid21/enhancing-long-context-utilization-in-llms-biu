import src.helpers.nq_data as nq_helper

from src.tasks.abstract import AbstractTask
from src.entities.experiments.gold_idx_change.data import GoldIdxChangeExperimentData


class GoldIdxChangeDatasetCreation(AbstractTask):

    def __init__(self, args):
        super().__init__(args)
        self._min_prompt_tokens = args.min_prompt_tokens

    async def load_data(self) -> GoldIdxChangeExperimentData:
        nq_helper.get_golden_idx_change_data(prompting_mode=self._prompting_mode,
                                             tokenizer=self._tokenizer,
                                             min_prompt_tokens=self._min_prompt_tokens,
                                             )