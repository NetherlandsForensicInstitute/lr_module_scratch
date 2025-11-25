import os
import shutil
from pathlib import Path
from pickle import UnpicklingError
from unittest import mock

import pytest
from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem

from lrmodule import ModelSettings
from lrmodule.data_types import MarkType, ScoreType
from lrmodule.persistence import load_model, save_model, _get_model_filename


MODEL_STORAGE_PATH = Path(__file__).parent / "test_model_storage"


@pytest.fixture(autouse=True)
def clear_test_model_storage_directory():
    """Clean up 'test_model_storage' directory before running each test.

    This ensures a fresh environment for each test. The generated artifacts are not
    cleaned up after each test to allow easy debugging of the generated pickle files.
    """
    if MODEL_STORAGE_PATH.exists():
        shutil.rmtree(MODEL_STORAGE_PATH)
    MODEL_STORAGE_PATH.mkdir(parents=True)


def test_serialize_trained_lr_system(trained_lr_system: LRSystem):
    """Check that a trained LR system can be serialized."""
    # Given that we have a trained LR system
    settings = ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF)
    mark_type = settings.mark_type.value
    score_type = settings.score_type.value

    # When we serialize the LR system
    save_model(trained_lr_system, settings, MODEL_STORAGE_PATH)

    # There should be a file we can load
    model_filename = _get_model_filename(settings)
    model_file_path = MODEL_STORAGE_PATH / model_filename
    assert model_file_path.exists()


def test_deserialize_trained_lr_system(trained_lr_system: LRSystem, sample_feature_data: FeatureData):
    """Check that a deserialized, trained LR system yields exactly the same results."""
    # Given that we have a certain LR system serialized
    settings = ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF)
    save_model(trained_lr_system, settings, MODEL_STORAGE_PATH)

    # When the model is deserialized
    deserialized_model = load_model(settings, MODEL_STORAGE_PATH)

    # The deserialized model and the model it originated from should be of the same type of LR system
    assert type(trained_lr_system) == type(trained_lr_system)

    # The calculated LLR output should be identical to the LR system output of the serialized model
    expected_llr_data = trained_lr_system.apply(sample_feature_data)
    deserialized_model_data = deserialized_model.apply(sample_feature_data)

    assert deserialized_model_data == expected_llr_data


@pytest.mark.parametrize('mark_type,score_type', [
    (MarkType.FIRING_PIN_IMPRESSION, ScoreType.CMC),  # other score type
    (MarkType.BREECH_PIN_IMPRESSION, ScoreType.ACCF),  # other mark type
    (MarkType.BREECH_PIN_IMPRESSION, ScoreType.CMC),  # other mark and other score type
])
def test_deserialize_inexistent_lr_system(trained_lr_system: LRSystem, mark_type: MarkType, score_type: ScoreType):
    """Check that an appropriate error is raised when there is no serialized model."""
    # Given that the LR model storage directory is empty
    assert os.listdir(MODEL_STORAGE_PATH) == []

    # Given that we have a serialized model for a given type of `ModelSettings`
    settings = ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF)
    save_model(trained_lr_system, settings, MODEL_STORAGE_PATH)
    assert len(os.listdir(MODEL_STORAGE_PATH)) == 1

    # When we try to deserialize a model for a different type of `ModelSettings`
    other_settings = ModelSettings(mark_type, score_type)

    # An exception should be raised mentioning that we can't find that particular deserialized LR model
    with pytest.raises(FileNotFoundError) as exception_info:
        load_model(other_settings, MODEL_STORAGE_PATH)

    # The exception should mention no models found for the requested mark/score types
    assert "No model found for mark type" in str(exception_info.value)
    assert other_settings.mark_type.value in str(exception_info.value)
    assert other_settings.score_type.value in str(exception_info.value)


def test_deserialize_from_invalid_pickle_file(trained_lr_system: LRSystem):
    """Check that an appropriate error is raised when unable to unpickle serialized model."""
    # Given that we have a serialized model for a given type of `ModelSettings`
    settings = ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF)
    save_model(trained_lr_system, settings, MODEL_STORAGE_PATH)

    with mock.patch('pickle.load', side_effect=UnpicklingError("Some pickle error")):
        # When pickle can't load the given file, we expect an appropriate error to be raised
        with pytest.raises(RuntimeError) as exception_info:
            load_model(settings, MODEL_STORAGE_PATH)

        assert "Could not load model from .pkl file for mark type" in str(exception_info.value)
        assert settings.mark_type.value in str(exception_info.value)
        assert settings.score_type.value in str(exception_info.value)
