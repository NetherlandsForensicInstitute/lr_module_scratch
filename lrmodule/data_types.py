from collections import namedtuple
from enum import Enum
from typing import NamedTuple


class MarkType(Enum):
    FIRING_PIN_IMPRESSION = "firing_pin_impression"
    BREECH_PIN_IMPRESSION = "breech_pin_impression"
    APERTURE_SHEER = "aperture_sheer"


class ScoreType(Enum):
    CMC = "cmc"
    ACCF = "accf"


class ModelSettings(NamedTuple):
    mark_type: MarkType
    score_type: ScoreType


# Group the LLR percentile and best-estimate data in a tuple
LLR_data = namedtuple("LLR_data", ["low_percentile", "best_estimate", "high_percentile"])
