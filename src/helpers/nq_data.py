import logging
import json

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Optional, Dict,List
from xopen import xopen
from copy import deepcopy

from common.consts import NQ_DATASET_FILE_PATH, MODEL_DOCS_MAPPINGS

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
    prompting_mode: PromptingMode,
    model_name: Optional[str] = None,
    num_idxs: Optional[int] = None,
    path: Optional[str] = None,
    num_examples: Optional[int] = None,
    model: Optional[str] = None
) -> GoldIdxChangeExperimentData:

    logger.info(f"get_golden_idx_change_data - started: {num_idxs=}, {prompting_mode=}, {model_name=}, {path=}, {num_examples=}")
    raw_data: List[SingleQuestionRawData] = await read_data_file(prompting_mode=prompting_mode, path=path, num_examples=num_examples)
    experiments = []

    if num_idxs and prompting_mode in PromptingMode.get_ctx_modes():
        experiments = await _create_experiments_with_full_documents_list(num_idxs=num_idxs,
                                                                         raw_data=raw_data,
                                                                         model_name=model_name)
    # when there are no gold indcies - we're working in closedbook or baseline
    else:
        experiments = await _create_experiments_with_partial_documents_list(prompting_mode=prompting_mode, raw_data=raw_data)

    gold_idx_change_data = GoldIdxChangeExperimentData(experiments=experiments, model=model)
    logger.info(f"get_golden_idx_change_data - finished: {len(gold_idx_change_data.experiments)=}")
    return gold_idx_change_data


async def _generate_gold_idxs(num_idxs: int, num_docs: int) -> AsyncIterator[int]:
    await _validate_create_gold_idx_list_args(num_idxs=num_idxs, num_docs=num_docs)

    if num_idxs == 1:
        yield 0
        return

    step = (num_docs - 1) / (num_idxs - 1)

    for i in range(num_idxs):
        yield round(i * step)


async def _validate_create_gold_idx_list_args(num_docs: int, num_idxs: int) -> None:
    if not isinstance(num_docs, int) or not isinstance(num_idxs, int):
        raise TypeError("num_docs and num_idxs must be integers")
    if num_docs <= 0:
        raise ValueError("num_docs must be > 0")
    if num_idxs <= 0:
        raise ValueError("num_idxs must be > 0")
    if num_idxs > num_docs:
        raise ValueError("num_idxs cannot exceed num_docs")


async def _create_experiments_with_full_documents_list(
    num_idxs: int,
    raw_data: List[SingleQuestionRawData],
    model_name: str,
    log_steps: int = 100
) -> List[SingleIdxData]:

    if log_steps <= 0 or log_steps >= len(raw_data):
        raise ValueError(f"'log_steps' mus be in range [1,{len(raw_data)-1}]")

    model_short_name = model_name.split("/")[-1]

    num_docs = MODEL_DOCS_MAPPINGS.get(model_short_name)
    if not num_docs:
        raise RuntimeError(f"num docs mapping failed! model={model_short_name}")
    
    logger.info(f"MODEL_DOCS_MAPPINGS: {num_docs=}")

    experiments = []
    async for gold_idx in _generate_gold_idxs(num_idxs=num_idxs, num_docs=num_docs):
        logger.info(f"IDX: {gold_idx=}")

        idx_data = []
        for i, data in enumerate(raw_data, 1):
            if log_steps and i % log_steps == 0:
                logger.info(f"PROCESSING: {i}/{len(raw_data)}")

            if not data.documents:
                logger.warning(f"question has no docs: {data.question_id=}")
                continue
            if not data.gold_docs:
                logger.warning(f"question has no gold docs: {data.question_id=}")
                continue
            
            documents_cpy: List[Document] = deepcopy(data.documents)
            documents_cpy = documents_cpy[:num_docs]

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
    
    return experiments


async def _create_experiments_with_partial_documents_list(
    prompting_mode: PromptingMode,
    raw_data: List[SingleQuestionRawData],
    log_steps: int = 100
) -> List[SingleIdxData]:

    if log_steps <= 0 or log_steps >= len(raw_data):
        raise ValueError(f"'log_steps' mus be in range [1,{len(raw_data)-1}]")

    idx_data = []

    for i, data in enumerate(raw_data, 1):
        if log_steps and i % log_steps == 0:
            logger.info(f"PROCESSING: {i}/{len(raw_data)}")
        documents = None
        
        if prompting_mode == PromptingMode.BASELINE:
            data.gold_docs.sort(key=lambda obj: obj.score)
            documents = [data.gold_docs[0]]
        
        idx_data.append(
            SingleQuestionData(
                question_id=data.question_id,
                question=data.question,
                answers=data.answers,
                documents=documents
            )
        )

    return [SingleIdxData(name=prompting_mode.value, data=idx_data)]


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
    
    logger.info(f"read_data_file - finished: {len(raw_data)=}")
    return raw_data


async def _get_data_path(path: Optional[str] = None) -> Path:
    return Path(path) if path else NQ_DATASET_FILE_PATH


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
        question_data_payload["documents"] = docs

    return question_data_payload
