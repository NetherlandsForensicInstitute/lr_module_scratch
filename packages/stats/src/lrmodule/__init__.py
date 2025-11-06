from pathlib import Path

import numpy as np
from lir.data.models import DataSet

from lrmodule import persistence
from lrmodule.data import get_dataset_id
from lrmodule.lrsystem import train_model, ScratchLrSystem
from lrmodule.models import ModelSettings


def get_model(settings: ModelSettings, dataset: DataSet, cache_dir: Path) -> ScratchLrSystem:
    model = persistence.load_model(settings, get_dataset_id(dataset), cache_dir)
    if not model:
        model = train_model(settings, dataset)
        persistence.save_model(model, cache_dir)
    return model


def calculate_llrs(features: np.ndarray, settings: ModelSettings, dataset: DataSet, cache_dir: Path) -> np.ndarray:
    model = get_model(settings, dataset, cache_dir)
    llrs, labels, meta = model.apply(features, labels=None, meta=np.array((features.shape[0], 1)))
    return llrs
