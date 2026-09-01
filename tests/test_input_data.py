from pathlib import Path

import numpy as np
from lir.data.models import FeatureData
from lir.datasets.feature_data_csv import ExtraField, FeatureDataCsvParser
from numpy import array

from lrmodule.input_data import PredefinedCrossValidation


def test_input_data_to_instances():
    """Check that input data is correctly parsed to instances (having multiple folds)."""
    # Arrange
    input_file = Path(__file__).parent / "fixtures/input_data/train_test_data.csv"
    dataset = FeatureDataCsvParser(
        open_file_fn=lambda: open(input_file),
        hypothesis_column="hypothesis",
        source_id_column=["weapon1", "weapon2"],
        extra_fields=[
            ExtraField("split1", "split1", str),
            ExtraField("split2", "split2", str),
            ExtraField("split3", "split3", str),
        ],
    ).get_instances()
    strategy = PredefinedCrossValidation()

    # The following train/test splits for the given data_subsets are expected
    subset_1 = [
        FeatureData(hypothesis=array([1, 0]), features=array([[60.1234, 10, 21], [63.1234, 16, 20]])),
        FeatureData(hypothesis=array([1, 0]), features=array([[20.1234, 11, 42], [10.1234, 6, 34]])),
    ]

    subset_2 = [
        FeatureData(hypothesis=array([1, 0]), features=array([[20.1234, 11, 42], [10.1234, 6, 34]])),
        FeatureData(hypothesis=array([1, 0]), features=array([[60.1234, 10, 21], [63.1234, 16, 20]])),
    ]

    subset_3 = [
        FeatureData(
            hypothesis=array([1, 1, 0, 0]),
            features=array([[60.1234, 10, 21], [20.1234, 11, 42], [10.1234, 6, 34], [63.1234, 16, 20]]),
        ),
        FeatureData(hypothesis=array([0]), features=array([[9.1234, 2, 12]])),
    ]

    # Act
    split = np.column_stack([dataset.split1, dataset.split2, dataset.split3])  # type: ignore
    assert split.shape == (5, 3), "role assignment shape should match the input data"
    assert np.all(split[:, 0] == np.array(["t", "v", "v", "n", "t"]))

    data_subsets = list(strategy.apply(dataset))

    # Assert
    # The fixture contains 3 subsets of data (3-fold cross validation)
    assert len(data_subsets) == 3  # noqa: PLR2004 (magic number)

    for i, ((actual_train, actual_test), (expected_train, expected_test)) in enumerate(
        zip(data_subsets, [subset_1, subset_2, subset_3])
    ):
        assert FeatureData(features=actual_train.features, hypothesis=actual_train.hypothesis) == expected_train
        assert FeatureData(features=actual_test.features, hypothesis=actual_test.hypothesis) == expected_test
