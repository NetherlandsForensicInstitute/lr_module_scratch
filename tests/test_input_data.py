from pathlib import Path

import numpy as np
import pytest
from lir.data.models import FeatureData
from lir.datasets.feature_data_csv import ExtraField, FeatureDataCsvParser
from numpy import array

from lrmodule.input_data import CrossValidationPairs, PredefinedCrossValidation


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


def test_cross_validation_pairs():
    """Check that CrossValidationPairs correctly splits paired data into k folds."""
    # Arrange
    source_ids = array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [0, 3],
        [1, 4],
        [2, 5],
        [0, 4],
        [1, 5],
        [2, 3],
    ])
    labels = array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    features = np.arange(24).reshape(12, 2).astype(float)
    data = FeatureData(features=features, hypothesis=labels, source_ids=source_ids)
    strategy = CrossValidationPairs(folds=3, random_state=42)

    # Act
    splits = list(strategy.apply(data))

    # Assert: correct number of folds
    assert len(splits) == 3  # noqa: PLR2004

    for train, test in splits:
        # No source overlap between train and test
        assert train.source_ids is not None
        assert test.source_ids is not None

        train_sources = set(train.source_ids.flatten())
        test_sources = set(test.source_ids.flatten())
        assert train_sources & test_sources == set(), "sources should not overlap between train and test"

        # All pairs in train have both sources in training set
        for pair in train.source_ids:
            assert pair[0] in train_sources
            assert pair[1] in train_sources

        # All pairs in test have both sources in test set
        for pair in test.source_ids:
            assert pair[0] in test_sources
            assert pair[1] in test_sources


def test_cross_validation_pairs_invalid_source_ids():
    """Check that CrossValidationPairs raises an exception for invalid source_ids shapes."""
    features = np.arange(12).reshape(6, 2).astype(float)
    labels = array([1, 1, 0, 0, 0, 0])
    strategy = CrossValidationPairs(folds=3, random_state=42)

    source_ids_1col = array([[0], [1], [2], [3], [4], [5]])
    data_1col = FeatureData(features=features, hypothesis=labels, source_ids=source_ids_1col)
    with pytest.raises(ValueError, match="expected two-column source_ids"):
        list(strategy.apply(data_1col))
