from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem

from lrmodule.data_types import ModelSettings


def train_model(settings: ModelSettings, training_data: FeatureData) -> LRSystem:
    """Train a model from data."""
    raise NotImplementedError
