import numpy as np
from lir.data.models import FeatureData

from lrmodule.data import get_dataset_id


def test_dataset_id():
    assert get_dataset_id(FeatureData(features=np.arange(10))) == get_dataset_id(FeatureData(features=np.arange(10)))
    assert get_dataset_id(FeatureData(features=np.arange(10))) != get_dataset_id(FeatureData(features=np.ones(10)))


if __name__ == "__main__":
    test_dataset_id()
