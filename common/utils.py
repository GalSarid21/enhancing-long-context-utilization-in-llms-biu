import random

from typing import List, Dict, Optional

from src.entities.document import Document


def get_messages_list(
    user: str,
    system: Optional[str] = None,
    apply_ici: Optional[bool] = True
) -> List[Dict[str, str]]:

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if apply_ici is True and system is not None:
        messages.append({"role": "system", "content": system})
    return messages


def reposition_with_shuffle(
    lst: List[Document],
    source_index: int,
    target_index: int
) -> None:

    element = lst.pop(source_index)
    random.shuffle(lst)
    lst.insert(target_index, element)
