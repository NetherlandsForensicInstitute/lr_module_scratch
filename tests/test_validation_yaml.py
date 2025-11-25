from pathlib import Path

import confidence
from lir.config.experiment_strategies import parse_experiments_setup


def test_validation_yaml():
    validation_file = Path(__file__).parent.parent / "validation.yaml"
    setup, _ = parse_experiments_setup(confidence.loadf(validation_file))
    for _, experiment in setup.items():
        experiment.run()
