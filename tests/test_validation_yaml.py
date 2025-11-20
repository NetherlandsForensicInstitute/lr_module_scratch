from pathlib import Path

import confidence
import pytest
from lir.config.experiment_strategies import parse_experiments_setup


def test_validation_yaml():
    validation_file = Path(__file__).parent.parent / "validation.yaml"
    setup, _ = parse_experiments_setup(confidence.loadf(validation_file))
    for name, experiment in setup.items():
        # TODO: once the experiments are working and data are accessible, this should not raise an exception
        # Alternatively, run this test with fake data
        with pytest.raises(Exception):
            experiment.run()

