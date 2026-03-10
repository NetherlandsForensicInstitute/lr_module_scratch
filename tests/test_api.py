import os

import confidence
import pytest
from lir.data.models import FeatureData
from lir.lrsystems import LRSystem
from lir.main import initialize_experiments
from lir.util import check_type

from lrmodule import Path, get_lr_system, get_reference_data

# TEST_FOLDER is a folder that contains packages (folders) with everything relevant for that specific model
# (e.g. reference data, trained model, etc.).
TEST_FOLDER = Path(__file__).parent / 'saved_models'
os.makedirs(TEST_FOLDER, exist_ok=True)

# Generate models at import time so that TEST_FOLDER is populated before @pytest.mark.parametrize
# evaluates its argument list during collection.
_yaml_file = Path(__file__).parent.parent / 'validation.yaml'
_cfg = confidence.Configuration(confidence.loadf(_yaml_file), {'output_path': TEST_FOLDER})
_exps, _ = initialize_experiments(_cfg)
for _exp in _exps.values():
    _exp.run()


# Test for every folder in TEST_FOLDER that the `get_lr_system` function can load a trained LR system and that it
# returns an LR system object.
@pytest.mark.parametrize("model_folder", [f for f in TEST_FOLDER.iterdir() if f.is_dir()])
def test_get_lr_system_loads_trained_model(model_folder: Path):
    """Check that the `get_lr_system` function can load a trained LR system."""
    lr_system = get_lr_system(model_folder)

    # We should get an LR system object
    assert lr_system is not None
    check_type(LRSystem, lr_system)


@pytest.mark.parametrize("model_folder", [f for f in TEST_FOLDER.iterdir() if f.is_dir()])
def test_get_reference_data(model_folder: Path):
    """Check that the `get_reference_data` function can load reference data."""
    reference_data = get_reference_data(model_folder)

    # We should get reference data
    assert reference_data is not None
    check_type(FeatureData, reference_data)
