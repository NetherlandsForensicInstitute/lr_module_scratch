from lir.data.models import DataSet
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.transform.pipeline import Pipeline

from lrmodule.data_types import ModelSettings


class ScratchLrSystem(BinaryLRSystem):
    def __init__(self, settings: ModelSettings):
        super().__init__(name="scratch", pipeline=Pipeline([]))
        self.settings = settings
        self.dataset_id = None


def train_model(settings: ModelSettings, data: DataSet) -> ScratchLrSystem:
    raise NotImplementedError
