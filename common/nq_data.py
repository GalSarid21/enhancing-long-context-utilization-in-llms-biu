from entities.document import Document
from entities.enums import PromptingMode
from src.wrappers import HfTokenizer

from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Any
from random import shuffle
from xopen import xopen
from copy import deepcopy
from tqdm import tqdm

import logging
import shutil
import gzip
import json
import os


def read_files_by_num_docs(
    folder_path: str,
    prompting_mode: PromptingMode,
    num_examples: Optional[int] = None
) -> Dict[str, Dict[str, Union[List[str], List[List[Document]]]]]:
    """
    Reads all jsonl files from input folder and creats a file postfix based
    data object (used for gold_idx_change experiment).
    """
    folder_path = Path(folder_path)
    logging.info(f"Creating documents from {folder_path} folder...")

    jsonl_files = list(folder_path.glob("*.jsonl"))
    files_data = {}

    if prompting_mode is PromptingMode.CLOSEDBOOK:
        questions, answers, _ = read_file(
            # we need only one file since we don't use documents and this
            # is what changes between files
            file_path=jsonl_files[0],
            prompting_mode=prompting_mode,
            num_examples=num_examples
        )
        files_data.update({
            prompting_mode.value: {
                "questions": questions,
                "answers": answers
            }
        })

    else:
        for jsonl in jsonl_files:
            questions, answers, documents = read_file(
                file_path=jsonl, prompting_mode=prompting_mode, num_examples=num_examples
            )
            # stem should look like the following:
            # nq-open-10_total_documents_gold_at_0
            # hence - split("_documents_")[-1] - will
            # create a short name such as "gold_at_0"
            file_short_name = jsonl.stem.split("_documents_")[-1]
            files_data.update({
                file_short_name: {
                    "questions": questions,
                    "answers": answers,
                    "documents": documents
                }
            })

    return files_data


def read_files_by_gold_idx(
    folder_paths: List[str],
    prompting_mode: PromptingMode,
    gold_idx: int,
    num_examples: Optional[int] = None
) -> Dict[str, Dict[str, Union[List[str], List[List[Document]]]]]:
    """
    Reads a jsonl file that ends with `gold_idx` postfix from input folder
    and creats a file prefix based data object (used for num_docs_change
    experiment).
    """
    suffix = f"gold_at_{gold_idx}.jsonl"
    file_paths = [
        next(file for file in Path(folder).glob(f"*{suffix}"))
        for folder in folder_paths
    ]
    logging.info(f"Creating documents from relevant gold_idx={gold_idx} files...")

    files_data = {}
    for file_path in file_paths:
        questions, answers, documents = read_file(file_path, prompting_mode, num_examples)
        # stem should look like the following:
        # nq-open-10_total_documents_gold_at_0
        # hence - replace("nq-open-", "") will remove the "nq-open-" and
        # split("_gold_")[0] - will create a short name such as
        # "10_total_documents"
        file_short_name =\
            file_path.stem.replace("nq-open-", "").split("_gold_")[0]
        files_data.update({
            file_short_name: {
                "questions": questions,
                "answers": answers,
                "documents": documents
            }
        })

    return files_data


def read_file(
    file_path: str,
    prompting_mode: PromptingMode,
    num_examples: Optional[int] = None
) -> Tuple[List[str], List[List[str]], List[List[Document]]]:
    """
    Reads NQ dataset file using `file_path` and creates a list of lists
    of documents with a matching list of questions.
    """
    all_questions = []
    all_documents = []
    all_answers = []

    with open(file_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # get example's question
            question = input_example["question"]
            all_questions.append(question)
            # get example's answers
            answers = input_example["answers"]
            all_answers.append(answers)
            if prompting_mode is PromptingMode.CLOSEDBOOK:
                # closedbook doesn not need context document - 
                # we're returning and empty list
                continue
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if prompting_mode is PromptingMode.OPENBOOK_RANDOM:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            all_documents.append(documents)
            if num_examples and len(all_documents) == num_examples:
                all_questions = all_questions[:num_examples]
                answers = answers[:num_examples]
                break

    return all_questions, all_answers, all_documents


def serialize(obj: Any) -> Dict[str, str]:
    if isinstance(obj, Document):
        return obj.to_dict()
    elif (
        isinstance(obj, list) and
        isinstance(obj[0], list) and
        isinstance(obj[0][0], Document)
    ):
        return [
            [doc.to_dict() for doc in sublist]
            for sublist in obj
        ]
    else:
        return obj


def download_files_by_num_docs(
    num_docs: int,
    dst_dir: str,
    src_dir: str
) -> str:
    """
    Downloads the NQ dataset filesfrom the lost-in-the-middle local cloned git
    repo at `src_dir`, if it has a `num_docs` prefix` and saves it under
    `dst_dir`.

    Returns the new folder that was created during downloading.

    Skipping any existing `src_file` from `src_dir` during the download
    process.
    """
    logging.info(f"Downloading NQ Data [num_docs={num_docs}]...")
    # lost in the middle suppose to have one folder
    # for each number of test documents
    data_folder = [
        f for f in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, f))
            and f.startswith(str(num_docs))
    ][0]

    src_folder = f"{src_dir}/{data_folder}"
    dst_folder = f"{dst_dir}/{data_folder}"
    os.makedirs(dst_folder, exist_ok=True)

    download_folder_files(dst_folder, src_folder)
    return dst_folder


def download_files_by_gold_idx(
    gold_idx: int,
    dst_dir: str,
    src_dir: str
) -> List[str]:
    """
    Downloads the NQ dataset files that has a `gold_idx` posfix from the
    lost-in-the-middle local cloned git repo at `src_dir` and saves it under
    `dst_dir`.

    Returns the new folder that was created during downloading.

    Skipping any existing `src_file` from `src_dir` during the download
    process.
    """
    logging.info(f"Downloading NQ Data [gold_idx={gold_idx}]...")
    data_folders = [
        f for f in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, f))
    ]
    dst_folders = []

    for data_folder in data_folders:
        src_folder = f"{src_dir}/{data_folder}"
        dst_folder = f"{dst_dir}/{data_folder}"
        os.makedirs(dst_folder, exist_ok=True)
        postfix = f"gold_at_{gold_idx}.jsonl.gz"
        download_folder_files(dst_folder, src_folder, postfix)
        dst_folders.append(dst_folder)

    return dst_folders


def download_folder_files(
    dst_folder: str,
    src_folder: str,
    postfix: Optional[str] = None
) -> None:

    for file_name in os.listdir(src_folder):
        if file_name.endswith(".gz"):
            src_file = f"{src_folder}/{file_name}"
            dst_file = f"{dst_folder}/{file_name.replace('.gz', '')}"
            if (
                postfix is not None and
                not src_file.endswith(postfix)
            ):
                logging.info(
                    f"Skipping existing file " +
                    f"[file doesn't end with {postfix} postfix]: {src_file}"
                )
                continue

            if not os.path.exists(dst_file):
                with(
                    gzip.open(src_file, "rb") as f_in,
                    open(dst_file, "wb") as f_out
                ):
                    logging.info(f"Downloading file: {src_file} to: {dst_file}")
                    shutil.copyfileobj(f_in, f_out)

            else:
                logging.info(f"Skipping existing file [file exists]: {src_file}")


def read_all_files_src(
    model_name: str,
    num_files: Optional[int] = None,
    files_path: Optional[str] = "nq-open-contriever-msmarco-retrieved-documents.jsonl.gz",
    min_tokens: Optional[int] = 120_000,
    max_tokens: Optional[int] = 127_000
) -> List[Dict[str, Union[str, float]]]:

    tokenizer = HfTokenizer(model_name)
    all_files = []

    with xopen(files_path, "r") as f:
        for line in tqdm(f):
            res = json.loads(line)
            new_entry = {"question": res["question"], "answers": res["answers"], "documents": []}
            total_tokens = 0
            for i, elm in enumerate(res["ctxs"], 1):
                prompt = f"Document [{i}](Title: {elm['title']}) {elm['text']}"
                if elm["hasanswer"]:
                    if new_entry.get("doc_with_answer"):
                        continue
                    else:
                        new_entry["doc_with_answer"] = elm
                else:
                    new_entry["documents"].append(elm)
                tokens = tokenizer.tokenize(prompt)
                total_tokens += len(tokens)
                if total_tokens >= max_tokens:
                    break
            if total_tokens < min_tokens:
                continue
            new_entry["total_tokens"] = total_tokens
            if not new_entry.get("doc_with_answer"):
                continue
            all_files.append(new_entry)
            if num_files and len(all_files) == num_files:
                break

    return all_files
