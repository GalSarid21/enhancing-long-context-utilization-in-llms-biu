# Adapted from the nelson-liu/lost-in-the-middle repository
# Original source: https://github.com/nelson-liu/lost-in-the-middle
# Licensed under the MIT License
# Modifications may have been made to suit specific project needs.
from typing import List
import string
import regex


def normalize_sentence(s: str) -> str:
    """
    Normalization from the SQuAD evaluation script.
    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    # our addition to normalize unicode chars
    # currently - not in use to preven bias
    def decode_unicode(text):
        try:
            return text.encode('utf-8').decode('unicode_escape')
        except UnicodeDecodeError:
            return text

    # s = decode_unicode(s)
    s = lower(s)
    s = remove_punc(s)
    s = remove_articles(s)
    s = white_space_fix(s)
    return s


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_sentence(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_sentence(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0
