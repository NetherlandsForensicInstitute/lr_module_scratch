from enum import Enum
from typing import NamedTuple


class MarkType(Enum):
    FIRING_PIN_IMPRESSION = 1
    BREECH_PIN_IMPRESSION = 2
    APERTURE_SHEER = 3


class ScoreType(Enum):
    CMC = 1
    ACCF = 2


class ModelSettings(NamedTuple):
    mark_type: MarkType
    score_type: ScoreType
