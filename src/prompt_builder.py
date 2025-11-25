from src.entities.enums import PromptingMode
from src.entities.document import Document
from common.utils import get_messages_list
from src.wrappers import HfTokenizer

from typing import List, Tuple, Optional, Dict


class PromptBuilder:
    _WRITING_INSTRUCTIONS = "Write a high-quality answer for the given question using only the provided search results"
    _RELEVANCY_DESCLAIMER = "(some of which might be irrelevant)"
    _LENGTH_RESTRICTION_TEMPLATE = "Keep your answer short according to the max_new_tokens={max_tokens} limitation."
    _TEMPLATE_MAPPING = {}

    def __init__(
        self,
        prompting_mode: PromptingMode,
        tokenizer: HfTokenizer,
        max_tokens: int
    ) -> None:

        self._init_template_mappings(max_tokens=max_tokens)
        self._tokenizer = tokenizer
        self._prompting_mode = prompting_mode
        self._system, self._user_template = self._get_prompt_components()

    def _init_template_mappings(self, max_tokens: int) -> Dict:
        length_restriction = self._LENGTH_RESTRICTION_TEMPLATE.format(max_tokens=max_tokens)

        self._TEMPLATE_MAPPING = {
            PromptingMode.OPENBOOK: {
                "system": f"{self._WRITING_INSTRUCTIONS} {self._RELEVANCY_DESCLAIMER}. {length_restriction}",
                "user": [
                    "Search Results:\n{search_results}",
                    "Question: {question}",
                    "Answer:"
                ]
            },
            PromptingMode.OPENBOOK_RANDOM: {
                "system": f"{self._WRITING_INSTRUCTIONS} {self._RELEVANCY_DESCLAIMER}. The search results are ordered randomly. {length_restriction}",
                "user": [
                    "Search Results:\n{search_results}",
                    "Question: {question}",
                    "Answer:"
                ]
            },
            PromptingMode.BASELINE: {
                "system": f"{self._WRITING_INSTRUCTIONS}. All the search results are relevant and contain parts of the answer or all of it. {length_restriction}",
                "user": [
                    "Search Results:\n{search_results}",
                    "Question: {question}",
                    "Answer:"
                ]
            },
            PromptingMode.CLOSEDBOOK: {
                "system": f"Write a high-quality answer for the given question. {length_restriction}",
                "user": [
                    "Question: {question}",
                    "Answer:"
                ]
            }
        }


    def build(
        self,
        question: str,
        documents: Optional[List[Document]] = None
    ) -> str:

        if self._prompting_mode is PromptingMode.CLOSEDBOOK:
            user_prompt = self._user_template.format(question=question)
        else:
            search_results = self._format_documents(documents)
            user_prompt = self._user_template.format(
                search_results=search_results,
                question=question
            )

        if not self._tokenizer.is_chat_model:
            prompt = f"{self._system}\n\n{user_prompt}"
            return prompt

        messages = get_messages_list(user=user_prompt, system=self._system)
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt

    def _get_prompt_components(self) -> Tuple[str, str]:
        prompt_template_parts = self._TEMPLATE_MAPPING[self._prompting_mode]
        syetem = prompt_template_parts["system"]
        user_parts = prompt_template_parts["user"]
        user_template = "\n\n".join(user_parts)
        return syetem, user_template

    def _format_documents(self, documents: List[Document]) -> str:
        return "\n".join(
            f"Document [{document_index}](Title: {document.title}) {document.text}"
            for document_index, document in enumerate(documents, 1)
        )
