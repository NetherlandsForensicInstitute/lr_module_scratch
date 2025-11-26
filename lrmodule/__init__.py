from pathlib import Path

import numpy as np
from lir.config.lrsystem_architectures import specific_source
from lir.data.models import FeatureData, LLRData
from lir.lrsystems.lrsystems import LRSystem

from lrmodule import persistence
from lrmodule.data_types import ModelSettings
from lrmodule.lrsystem import get_trained_model


def get_model(settings: ModelSettings, training_data: FeatureData, model_storage_path: Path | None) -> LRSystem:
    """
    Obtain a model by loading it from disk, or by fitting it from training data.

    :param settings: model settings
    :param training_data: training data
    :param model_storage_path: path where trained LR models are stored
    :return: a fitted LR system
    """
    model = None if not model_storage_path else persistence.load_model(settings, model_storage_path)
    if not model:
        model = get_trained_model(settings, training_data)
        if model_storage_path:
            persistence.save_model(model, settings, model_storage_path)
    return model


def calculate_llrs(
    features: np.ndarray, settings: ModelSettings, training_data: FeatureData, model_storage_path: Path | None
) -> LLRData:
    """Calculate LLRs after fitting a model with a training set."""
    model = get_model(settings, training_data, model_storage_path)
    return model.apply(FeatureData(features=features))


# create an alias for the specific source system, since the architecture is identical but the name is misleading
# in the current application
binary_lrsystem = specific_source
