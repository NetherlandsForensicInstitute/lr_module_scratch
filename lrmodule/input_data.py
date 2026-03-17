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
        role_assignments = instances.split  # type: ignore
        for split in range(role_assignments.shape[1]):
            training_data = instances[role_assignments[:, split] == TestTrainSplit.TRAIN.value]
            test_data = instances[role_assignments[:, split] == TestTrainSplit.TEST.value]
            yield training_data, test_data
