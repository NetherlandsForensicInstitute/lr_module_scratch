import logging
from collections.abc import Iterable
from enum import StrEnum

from lir.data.models import DataStrategy, FeatureData

LOG = logging.getLogger(__name__)


class TestTrainSplit(StrEnum):
    NOT_USED = "n"
    TRAIN = "t"
    TEST = "v"


class PredefinedCrossValidation(DataStrategy):
    """Return a series of train/test sets for a predefined cross-validation setup."""

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Return a series of train/test sets for a predefined cross-validation setup."""
        for split in range(len(instances.split[0])):
            training_data = instances[instances.split[:, split] == TestTrainSplit.TRAIN.value]  # type: ignore
            test_data = instances[instances.split[:, split] == TestTrainSplit.TEST.value]  # type: ignore
            yield training_data, test_data
