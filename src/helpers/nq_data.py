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
from src.wrappers import HfTokenizer


logger = logging.getLogger(__name__)


async def get_golden_idx_change_data(
    prompting_mode: PromptingMode,
    tokenizer: Optional[HfTokenizer] = None,
    min_prompt_tokens: Optional[int] = None,
    gold_idxs: Optional[List[int]] = None,
    path: Optional[str] = None,
    num_examples: Optional[int] = None
) -> GoldIdxChangeExperimentData:

    if tokenizer and not min_prompt_tokens:
        raise Exception("tokenizer needs 'min_prompt_tokens' to set number of prompt documents")

    logger.info(f"get_golden_idx_change_data - started: {gold_idxs=}, {prompting_mode=}, {path=}, {num_examples=}")
    raw_data: List[SingleQuestionRawData] = await read_data_file(prompting_mode=prompting_mode, path=path, num_examples=num_examples)
    experiments = []

    if gold_idxs:
        experiments = await _create_experiments_with_full_documents_list(gold_idxs=gold_idxs,
                                                                         raw_data=raw_data,
                                                                         tokenizer=tokenizer,
                                                                         min_prompt_tokens=min_prompt_tokens)
    # when there are no gold indcies - we're working in closedbook or baseline
    else:
        experiments = await _create_experiments_with_partial_documents_list(prompting_mode=prompting_mode, raw_data=raw_data)

    gold_idx_change_data = GoldIdxChangeExperimentData(experiments=experiments)
    logger.info(f"get_golden_idx_change_data - finished: {len(gold_idx_change_data.experiments)=}, {gold_idx_change_data=}")
    return gold_idx_change_data


async def _create_gold_idx_list() -> List[int]:
    pass



async def _create_experiments_with_full_documents_list(
    gold_idxs: List[int],
    raw_data: List[SingleQuestionRawData],
    min_prompt_tokens: int,
    tokenizer: HfTokenizer
) -> List[SingleIdxData]:

    experiments = []
    for gold_idx in gold_idxs:

        idx_data = []
        for data in raw_data:
            documents_cpy: List[Document] = deepcopy(data.docuemnts)
            
            total_tokens = 0
            for num_docs, doc in enumerate(documents_cpy, 1):
                title_tokens: int = tokenizer.count_tokens(prompt=doc.title)
                text_tokens: int = tokenizer.count_tokens(prompt=doc.text)
                total_tokens += (title_tokens + text_tokens)
                if total_tokens >= min_prompt_tokens:
                    break
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


async def _create_experiments_with_partial_documents_list(prompting_mode: PromptingMode, raw_data: List[SingleQuestionRawData]) -> List[SingleIdxData]:
    idx_data = []

    for data in raw_data:
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
    
    logger.info(f"read_data_file - finished: {len(raw_data)=}, {raw_data=}")
    return raw_data


async def _get_data_path(path: Optional[str] = None) -> Path:
    if path:
        data_path = Path(path)
    else:
        project_root = await _get_project_root()
        # set default path if nothing else is provided
        data_path = project_root / "scripts" / "add_uuid_to_base_data" / "nq-open-with-uuid.jsonl.gz"

    return data_path


async def _get_project_root() -> Path:
    """We identify the root as the folder which has 'pyproject.toml' in it"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("pyproject.toml not found")


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
