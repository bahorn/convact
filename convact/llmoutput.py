import re
from enum import Enum, auto


class BooleanResponse(Enum):
    YES = auto()
    NO = auto()
    UNKNOWN = auto()


def normalize(word: str) -> str:
    """
    Map a word to lower case and remove special characters.
    """
    new = re.sub(
        '[^A-Za-z0-9]+', '', word
    )
    new = new.lower()
    return new


def map_llm_response_to_binary(message: str) -> BooleanResponse:
    """
    Map the LLMs output to a BooleanResponse.
    """

    new = normalize(message)

    if 'yes' in new:
        return BooleanResponse.YES
    if 'no' in new:
        return BooleanResponse.NO

    return BooleanResponse.UNKNOWN
