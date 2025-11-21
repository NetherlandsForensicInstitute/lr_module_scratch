from pathlib import Path

import pytest
from lir.lrsystems.lrsystems import LRSystem

from lrmodule import ModelSettings
from lrmodule.data_types import MarkType, ScoreType
from lrmodule.lrsystem import load_lrsystem
from lrmodule.persistence import load_model, save_model


MODEL_STORAGE_PATH = Path(__file__).parent / "test_model_storage"


def test_persistence():
    settings = ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF)

    # not implemented
    with pytest.raises(Exception):
        load_model(settings, "dataset_id", Path("/"))

    # not implemented
    lrsystem = load_lrsystem(settings)
    with pytest.raises(Exception):
        save_model(lrsystem, settings, "dataset_id", Path("/"))


def test_serialize_trained_lr_system(trained_lr_system: LRSystem):
    """Check that a trained LR system can be serialized."""
    # Given that we have a trained LR system
    expected_deserialized_lr_system = trained_lr_system
    settings = ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF)
    mark_type = settings.mark_type.value
    score_type = settings.score_type.value

    # When we serialize the LR system
    save_model(trained_lr_system, settings, MODEL_STORAGE_PATH)

    # There should be a file we can load
    model_file_path = MODEL_STORAGE_PATH / mark_type / score_type / "model.pkl"
    assert model_file_path.exists()