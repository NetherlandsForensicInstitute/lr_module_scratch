from pathlib import Path

import confidence
from lir.main import initialize_experiments


def test_validation_yaml(tmpdir: Path):
    """Test if the validation.yaml file can be parsed correctly.
    
    Does not test correctness of the content, only that it can be parsed without errors.
    Running the whole setup will take too long for a unit test.
    """
    validation_file = Path(__file__).parent.parent / "validation.yaml"
    cfg = confidence.Configuration(confidence.loadf(validation_file), {'output_path': tmpdir})
    setup, _ = initialize_experiments(cfg)

