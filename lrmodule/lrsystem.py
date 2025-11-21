import importlib
from pathlib import Path

import confidence
from lir.config.base import _expand
from lir.config.lrsystem_architectures import parse_lrsystem
from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem

from lrmodule import resources as resources_package
from lrmodule.data_types import ModelSettings


def load_lrsystem(settings: ModelSettings) -> LRSystem:
    """
    Load the LR system for the given model settings.

    :param settings: the model settings
    :return: an LRSystem object
    """
    filename = f"lrsystem_{settings.mark_type.value}_{settings.score_type.value}.yaml"
    path = importlib.resources.files(resources_package) / filename  # ty: ignore[unresolved-attribute]
    if not path.exists():
        raise FileNotFoundError(path)
    cfg = confidence.loadf(path)
    return parse_lrsystem(_expand([], cfg), Path("lrsystem_output"))


def get_trained_model(settings: ModelSettings, training_data: FeatureData) -> LRSystem:
    """
    Return a trained LR system.

    :param settings: settings
    :param training_data: training data
    :return: a trained LR system object
    """
    return load_lrsystem(settings).fit(training_data)
