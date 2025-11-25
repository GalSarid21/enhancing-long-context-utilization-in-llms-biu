import pytest

from src.wrappers import HfTokenizer


@pytest.fixture
def hf_tokenizer() -> HfTokenizer:
    return HfTokenizer(model="Qwen/Qwen3-0.6B")
