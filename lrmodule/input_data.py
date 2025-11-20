from collections import defaultdict
from collections.abc import Iterator
from enum import StrEnum
from functools import cache
from pathlib import Path

import pandas as pd
from lir.data.data_strategies import DataStrategy
from lir.data.models import FeatureData


class TestTrainSplit(StrEnum):
    NOT_USED = "n"
    TRAIN = "t"
    TEST = "v"


class ScratchData(DataStrategy):
    def __init__(self, input_file_path: Path):
        """Read and represent Scratch specific input data, as corresponding instances.

        The data might include n-fold cross validation splits, where each fold has a train/test split.
        This class provides access to iterate over the available folds and the corresponding train and test splits.
        """
        self.file_path = input_file_path

    def _read_instances_from_file(self):
        """Read K-fold cross validation CSV input data to a list of K corresponding subsets of test/train folds.

        In the CSV file, subsets of the data are indicated by the "split<N>" column. For example, 3-fold cross
        validation is represented through columns 'split1', 'split2' and 'split3' which indicate if the data in
        this subset belongs to the test split ("t"), train split ("v") or is not used ("n").
        """
        df = pd.read_csv(self.file_path)

        # Ensure all expected columns are present
        expected_columns = ["weapon1", "weapon2", "hypothesis", "split1"]
        if not all(column in df.columns for column in expected_columns):
            raise ValueError(
                f"Missing one of the expected columns: {', '.join(set(expected_columns) - set(df.columns))}"
            )

        if "accf" in df.columns:
            feature_columns = ["accf"]
        elif "ccf" in df.columns:
            feature_columns = ["ccf"]
        else:
            raise ValueError("Missing one of the expected feature columns: 'accf' or 'ccf'")

        # Find all columns regarding the prepared folds, named 'split*' ('split1', 'split2', etc.)
        fold_column_names = [c for c in df.columns if c.startswith("split")]

        label_column = ["hypothesis"]

        # Group the folds by the column name, i.e. 'split1', 'split2', etc.
        df_with_subsets = df.melt(
            id_vars=label_column + feature_columns,
            value_vars=fold_column_names,
            var_name="subset",
            value_name="test_train_split",
        )

        subsets = []

        # Loop over each subset
        for _, folds in df_with_subsets.groupby("subset"):
            # Filter out the data marked as "not used"
            test_train_folds = folds[folds.test_train_split != TestTrainSplit.NOT_USED]

            # Loop over 'train' / 'test' folds for the current subset
            subset_folds = defaultdict()

            for test_or_train_indicator, raw_data in test_train_folds.groupby("test_train_split"):
                # The `test_or_train_indicator` refers to the role of this data
                # in the current fold; belonging to either the 'test' or 'train' split.
                features = raw_data[feature_columns].to_numpy(dtype=float).flatten()
                labels = raw_data[label_column].to_numpy(dtype=int).flatten()

                subset_folds[test_or_train_indicator] = FeatureData(features=features, labels=labels)

            subsets.append((subset_folds[TestTrainSplit.TRAIN], subset_folds[TestTrainSplit.TEST]))

        return subsets

    @cache
    def _get_instances(self):
        """Read instances from file only once."""
        return self._read_instances_from_file()

    def __iter__(self) -> Iterator[tuple[FeatureData, FeatureData]]:
        """Allow iteration by looping over the resulting train/test split(s)."""
        yield from self._get_instances()
