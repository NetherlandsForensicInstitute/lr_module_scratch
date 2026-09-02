import logging
from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import Any

import numpy as np
from lir.data.models import DataStrategy, FeatureData
from lir.util import check_type
from sklearn.model_selection import KFold

LOG = logging.getLogger(__name__)


class TestTrainSplit(StrEnum):
    NOT_USED = "n"
    TRAIN = "t"
    TEST = "v"


class PredefinedCrossValidation(DataStrategy):
    """Return a series of train/test sets for a predefined cross-validation setup."""

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Return a series of train/test sets for a predefined cross-validation setup."""
        role_assignments = [instances.split1, instances.split2, instances.split3]  # type: ignore
        for split in range(3):
            training_data = instances[role_assignments[split] == TestTrainSplit.TRAIN.value]
            test_data = instances[role_assignments[split] == TestTrainSplit.TEST.value]
            yield training_data, test_data


class CrossValidationPairs(DataStrategy):
    """K-fold cross-validation for paired instances."""

    def __init__(self, folds: int, shuffle: bool | None = None, random_state: int | None = None):
        if shuffle is None:
            shuffle = random_state is not None
        self._kf = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)

    def apply(self, instances: FeatureData) -> Iterator[tuple[FeatureData, FeatureData]]:
        """Perform *k*-fold cross-validation on paired instances."""
        source_ids = instances.source_ids
        if source_ids is None or len(source_ids.shape) != 2 or source_ids.shape[1] != 2:  # noqa: PLR2004
            raise ValueError(f"expected two-column source_ids; shape found: {getattr(source_ids, 'shape', None)}")

        sources = np.unique(check_type(np.ndarray, source_ids))

        for train_source_index, _ in self._kf.split(sources):
            train_sources = set(sources[train_source_index])

            def _is_in_training(value: Any) -> bool:
                return value in train_sources

            source_membership = np.vectorize(_is_in_training)(instances.source_ids)
            training_instances = np.all(source_membership, axis=1)
            test_instances = np.all(~source_membership, axis=1)

            yield instances[training_instances], instances[test_instances]
