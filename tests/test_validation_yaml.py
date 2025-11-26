from pathlib import Path

import confidence
from lir.config.experiment_strategies import parse_experiments_setup


def test_validation_yaml():
    """Test if the validation.yaml file can be parsed correctly.
    
    Does not test correctness of the content, only that it can be parsed without errors.
    Running the whole setup will take too long for a unit test.
    """
    validation_file = Path(__file__).parent.parent / "validation.yaml"
    setup, _ = parse_experiments_setup(confidence.loadf(validation_file))

