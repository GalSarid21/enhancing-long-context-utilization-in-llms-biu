import logging
import json

from pathlib import Path
from typing import Optional, Dict,List
from xopen import xopen
from copy import deepcopy

from src.entities.experiments.gold_idx_change.data import (
    GoldIdxChangeExperimentData,
    SingleQuestionRawData,
    SingleQuestionData,
    SingleIdxData
)
from src.entities.document import Document
from src.entities.enums import PromptingMode


logger = logging.getLogger(__name__)


async def get_golden_idx_change_data(
    gold_idxs: List[int],
    prompting_mode: PromptingMode,
    path: Optional[str] = None,
    num_examples: Optional[int] = None
) -> GoldIdxChangeExperimentData:

    logger.info(f"get_golden_idx_change_data - started: {gold_idxs=}, {prompting_mode=}, {path=}, {num_examples=}")
    raw_data = await read_data_file(prompting_mode=prompting_mode, path=path, num_examples=num_examples)

    experiments = []
    for gold_idx in gold_idxs:

        idx_data = []
        for data in raw_data:
            documents_cpy = deepcopy(data.docuemnts)
            # TODO: create shuffle logic that is based on external document ids order (for stability)
            data.gold_docs.sort(key=lambda obj: obj.score)
            documents_cpy.insert(gold_idx, data.gold_docs[0])

            idx_data.append(
                SingleQuestionData(
                    question_id=data.question_id,
                    question=data.question,
                    answers=data.answers,
                    documents=documents_cpy
                )
            )

        experiments.append(SingleIdxData(name=f"gold_at_{gold_idx}", data=idx_data))

    gold_idx_change_data = GoldIdxChangeExperimentData(experiments=experiments)
    logger.info(f"get_golden_idx_change_data - finished: {len(gold_idx_change_data.experiments)=}, {gold_idx_change_data=}")
    return gold_idx_change_data


async def read_data_file(prompting_mode: PromptingMode, path: Optional[str] = None, num_examples: Optional[int] = None) -> List[SingleQuestionRawData]:
    data_path = await _get_data_path(path=path)
    logger.info(f"read_data_file - started: {data_path=}, {prompting_mode=}, {num_examples=}")

    raw_data = []
    with xopen(data_path, "rt") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                logger.warning(f"read_data_file - skipping line empty line number {i}")
                continue

            line_as_dict = json.loads(line)
            question_data_payload = await _get_single_question_raw_data_payload(raw_data=line_as_dict, prompting_mode=prompting_mode)
            raw_question_data = SingleQuestionRawData(**question_data_payload)
            raw_data.append(raw_question_data)
            
            if num_examples and len(raw_data) == num_examples:
                raw_data[:num_examples]
                break
    
    logger.info(f"read_data_file - finished: {len(raw_data)=}, {raw_data=}")
    return raw_data


async def _get_data_path(path: Optional[str] = None) -> Path:
    if path:
        data_path = Path(path)
    else:
        project_root = Path(__file__).resolve().parents[1]
        data_path = project_root / "scripts" / "add_uuid_to_base_data" / "nq-open-with-uuid.jsonl.gz"

    return data_path


async def _get_single_question_raw_data_payload(raw_data: Dict, prompting_mode: PromptingMode) -> Dict:
    question_data_payload = {
        "question_id": raw_data["id"],
        "question": raw_data["question"],
        "answers": raw_data["answers"]
    }

    if prompting_mode in PromptingMode.get_ctx_modes():
        gold_docs = []
        docs = []

        for ctx in raw_data["ctxs"]:
            document = Document.from_dict(ctx)

            if document.hasanswer:
                gold_docs.append(document)
            else:
                docs.append(document)

        question_data_payload["gold_docs"] = gold_docs
        if prompting_mode != PromptingMode.BASELINE:
            question_data_payload["docuemnts"] = docs[:5] # TEST

    return question_data_payload
