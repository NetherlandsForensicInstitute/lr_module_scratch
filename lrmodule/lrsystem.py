from lir.data.models import DataSet
from lir.lrsystems.lrsystems import Pipeline
from lir.lrsystems.specific_source import SpecificSourceSystem

from lrmodule.data_types import ModelSettings


class ScratchLrSystem(SpecificSourceSystem):
    def __init__(self, settings: ModelSettings):
        super().__init__(name="scratch", pipeline=Pipeline([]))
        self.settings = settings
        self.dataset_id = None


def train_model(settings: ModelSettings, data: DataSet) -> ScratchLrSystem:
    raise NotImplementedError
