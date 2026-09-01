from pathlib import Path

import pytest

from lrmodule import get_validation_experiment

# TEST_FOLDER is a folder that contains packages (folders) with everything relevant for that specific model
# (e.g. reference data, trained model, etc.).
TEST_FOLDER = Path(__file__).parent / "saved_models"


@pytest.fixture(
    scope="session",
    params=[
        "aperture_shear",
        "firing_pin_impression",
        "breech_face_impression",
    ],
)
def model_folder(request) -> Path:
    TEST_FOLDER.mkdir(exist_ok=True, parents=True)
    output_path = TEST_FOLDER / request.param

    # Generate models at import time so that TEST_FOLDER is populated before @pytest.mark.parametrize
    # evaluates its argument list during collection.
    return get_validation_experiment(request.param, Path("data") / f"{request.param}.csv", output_path)
