from lir.data.models import FeatureData
from lir.lrsystems import LRSystem
from lir.util import check_type

from lrmodule import Path, get_lr_system, get_reference_data


def test_get_lr_system_loads_trained_model(model_folder: Path):
    """Check that the `get_lr_system` function can load a trained LR system."""
    lr_system = get_lr_system(model_folder)

    # We should get an LR system object
    assert lr_system is not None
    check_type(LRSystem, lr_system)


def test_get_reference_data(model_folder: Path):
    """Check that the `get_reference_data` function can load reference data."""
    reference_data = get_reference_data(model_folder)

    # We should get reference data
    assert reference_data is not None
    check_type(FeatureData, reference_data)
