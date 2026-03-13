import os
from pathlib import Path

import confidence
import pytest
from lir.data.models import FeatureData
from lir.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalData
from lir.lrsystems.lrsystems import LRSystem
from lir.main import initialize_experiments


# TEST_FOLDER is a folder that contains packages (folders) with everything relevant for that specific model
# (e.g. reference data, trained model, etc.).
TEST_FOLDER = Path(__file__).parent / 'saved_models'


@pytest.fixture(scope='session', params=[
    'aperture_shear',
    'firing_pin_impression',
    'breech_face_impression',
])
def model_folder(request) -> Path:
    TEST_FOLDER.mkdir(exist_ok=True, parents=True)

    # Generate models at import time so that TEST_FOLDER is populated before @pytest.mark.parametrize
    # evaluates its argument list during collection.
    yaml_file = Path('models') / request.param / 'validation.yaml'
    yaml_update = {
        'output_path': TEST_FOLDER / request.param,
        'input_file_path': str(Path('data') / f'{request.param}.csv'),
    }
    cfg = confidence.Configuration(confidence.loadf(yaml_file), yaml_update)
    exps, _ = initialize_experiments(cfg)
    for exp in exps.values():
        exp.run()
    return TEST_FOLDER / request.param / 'model'
