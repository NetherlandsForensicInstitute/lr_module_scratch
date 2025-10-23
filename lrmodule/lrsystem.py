from lir.data.models import DataStrategy
from lir.lrsystems.specific_source import SpecificSourceSystem


def train_model(settings: dict[str, str], data: DataStrategy) -> SpecificSourceSystem:
    raise NotImplementedError