from pathlib import Path

import numpy as np
from lir.data.models import FeatureData, LLRData
from lir.lrsystems.lrsystems import LRSystem

from lrmodule import persistence
from lrmodule.data import get_dataset_id
from lrmodule.data_types import ModelSettings
from lrmodule.lrsystem import get_trained_model


def get_model(settings: ModelSettings, dataset: FeatureData, cache_dir: Path) -> LRSystem:
    """TODO: docstr."""
    dataset_id = get_dataset_id(dataset)
    model = persistence.load_model(settings, dataset_id, cache_dir)
    if not model:
        model = get_trained_model(settings, dataset)
        persistence.save_model(model, settings, dataset_id, cache_dir)
    return model


def calculate_llrs(
    features: np.ndarray, settings: ModelSettings, training_data: FeatureData, cache_dir: Path
) -> LLRData:
    """TODO: docstr."""
    model = get_model(settings, training_data, cache_dir)
    return model.apply(FeatureData(features=features))
