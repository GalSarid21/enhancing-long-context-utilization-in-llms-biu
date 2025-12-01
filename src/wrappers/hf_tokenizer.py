from common.utils import get_messages_list

from transformers import AutoTokenizer, TensorType
from typing import List, Dict, Optional, Any


class HfTokenizer:

    def __init__(self, model: str) -> None:
        self._model = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._is_chat_model = self._tokenizer.chat_template is not None


    @property
    def model(self) -> str:
        return self._model

    @property
    def eos_token(self) -> str:
        return self._tokenizer.eos_token

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def is_chat_model(self) -> bool:
        return self._is_chat_model


    async def tokenize(
        self,
        text: str,
        pair: Optional[str] = None,
        add_special_tokens: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Exposing the 'tokenize' method with its HF signature.
        """
        return self._tokenizer.tokenize(
            text=text,
            pair=pair,
            add_special_tokens=add_special_tokens,
            **kwargs
        )

    async def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        **tokenizer_kwargs: Any
    ) -> (str | List[int]):
        """
        Exposing the 'apply_chat_template' method with its HF signature.
        """
        return self._tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            **tokenizer_kwargs
        )

    async def count_tokens(
        self,
        prompt_with_inst_tokens: Optional[bool] = False,
        **kwargs
    ) -> int:
        """
        Function to count tokens using the 'apply_chat_template' method,
        meaning that we consider the prompt special tokens being added
        by the tokenizer in our count.

        Gets **kwargs input to handle both messages (List[Dict[str, str]])
        and prompt (str) inputs correctly.

        If `prompt_with_inst_tokens` is not None, that means we want to
        tokenize the existing prompt without adding the instruction tokens
        again.

        Raises an exception if none of those two variables are being passed.
        """
        prompt = kwargs.get("prompt")
        if prompt is not None:
            if prompt_with_inst_tokens is True:
                tokenized_prompt = self.tokenize(prompt)
                return len(tokenized_prompt)
            messages = get_messages_list(user=prompt)
        
        elif kwargs.get("messages"):
            messages = kwargs.get("messages")

        else:
            raise Exception(
                "'count_tokens' function must get either 'prompt' or " +
                "'messages' input variables."
            )

        tokenized_messages = self.apply_chat_template(messages)
        return len(tokenized_messages)
