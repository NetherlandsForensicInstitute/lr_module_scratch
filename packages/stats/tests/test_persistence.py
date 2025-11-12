from pathlib import Path

import pytest

from lrmodule import ModelSettings
from lrmodule.data_types import MarkType, ScoreType
from lrmodule.lrsystem import load_lrsystem
from lrmodule.persistence import load_model, save_model


def test_persistence():
    settings = ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF)

    # not implemented
    with pytest.raises(Exception):
        load_model(settings, "dataset_id", Path("/"))

    # not implemented
    lrsystem = load_lrsystem(settings)
    with pytest.raises(Exception):
        save_model(lrsystem, settings, "dataset_id", Path("/"))
