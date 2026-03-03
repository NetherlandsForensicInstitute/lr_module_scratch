import logging
from collections.abc import Iterable
from enum import StrEnum
from functools import cache
from pathlib import Path

from lir.data.io import search_path
from lir.data.models import DataStrategy, FeatureData
from lir.datasets.feature_data_csv import ExtraField, FeatureDataCsvParser

LOG = logging.getLogger(__name__)


class TestTrainSplit(StrEnum):
    NOT_USED = "n"
    TRAIN = "t"
    TEST = "v"


SPLIT_COLUMNS = ["split1", "split2", "split3"]


class ScratchCsvReader(FeatureDataCsvParser):
    def __init__(self, input_file_path: Path | str):
        """Read and represent Scratch specific input data, as corresponding instances.

        The data might include n-fold cross validation splits, where each fold has a train/test split.
        This class provides access to iterate over the available folds and the corresponding train and test splits.
        """
        super().__init__(
            source_id_column=["weapon1", "weapon2"],
            label_column="hypothesis",
            extra_fields=[
                ExtraField("split", SPLIT_COLUMNS, str),
            ],
            message_prefix=f"{input_file_path}: ",
        )

        self.file_path = Path(input_file_path)

    @cache
    def get_instances(self) -> FeatureData:
        """Read K-fold cross validation CSV input data to a list of K corresponding subsets of test/train folds.

        In the CSV file, subsets of the data are indicated by the "split<N>" column. For example, 3-fold cross
        validation is represented through columns 'split1', 'split2' and 'split3' which indicate if the data in
        this subset belongs to the test split ("t"), train split ("v") or is not used ("n").

        The FeatureData instances are created from all columns that are not part of the expected columns:
        'weapon1', 'weapon2', 'hypothesis', 'split<N>' (for all N). The 'hypothesis' column is used as label.
        The remaining columns are treated as features. This means that the pipeline in which this data is used
        should filter out any non-relevant feature columns before training or evaluating a model.
        """
        path = search_path(self.file_path)
        LOG.debug(f"parsing CSV file: {self.file_path} as {path}")
        with open(path) as f:
            return self._parse_file(f)


class PredefinedCrossValidation(DataStrategy):
    """Return a series of train/test sets for a predefined cross-validation setup."""

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Return a series of train/test sets for a predefined cross-validation setup."""
        for split in range(len(SPLIT_COLUMNS)):
            training_data = instances[instances.split[:, split] == TestTrainSplit.TRAIN.value]  # type: ignore
            test_data = instances[instances.split[:, split] == TestTrainSplit.TEST.value]  # type: ignore
            yield training_data, test_data
