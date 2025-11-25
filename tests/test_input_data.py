from pathlib import Path

from lir.data.models import FeatureData
from lrmodule.input_data import ScratchData
from numpy import array


def test_input_data_to_instances():
    """Check that input data is correctly parsed to instances (having multiple folds)."""
    # Arrange
    input_file = Path(__file__).parent / "fixtures/input_data/train_test_data.csv"
    parsed_input_data = ScratchData(input_file)

    # The following train/test splits for the given data_subsets are expected
    subset_1 = {
        "train": FeatureData(labels=array([1, 0]), features=array([[60.1234, 10, 21], [63.1234, 16, 20]])),
        "test": FeatureData(labels=array([1, 0]), features=array([[20.1234, 11, 42], [10.1234, 6, 34]])),
    }

    subset_2 = {
        "train": FeatureData(labels=array([1, 0]), features=array([[20.1234, 11, 42], [10.1234, 6, 34]])),
        "test": FeatureData(labels=array([1, 0]), features=array([[60.1234, 10, 21], [63.1234, 16, 20]])),
    }

    subset_3 = {
        "train": FeatureData(
            labels=array([1, 1, 0, 0]),
            features=array([[60.1234, 10, 21], [20.1234, 11, 42], [10.1234, 6, 34], [63.1234, 16, 20]]),
        ),
        "test": FeatureData(labels=array([0]), features=array([[9.1234, 2, 12]])),
    }

    # Act
    data_subsets = list(parsed_input_data)

    # Assert
    # The fixture contains 3 subsets of data (3-fold cross validation)
    assert len(data_subsets) == 3  # noqa: PLR2004 (magic number)
    assert data_subsets == [
        (subset_1["train"], subset_1["test"]),
        (subset_2["train"], subset_2["test"]),
        (subset_3["train"], subset_3["test"]),
    ]
