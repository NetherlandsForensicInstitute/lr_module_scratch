import pickle
from pathlib import Path

import confidence
from lir.config.lrsystem_architectures import specific_source
from lir.data.models import FeatureData
from lir.datasets.feature_data_csv import FeatureDataCsvFileParser
from lir.experiments import Experiment
from lir.lrsystems.lrsystems import LRSystem
from lir.main import initialize_experiments


def get_lr_system(lr_system_folder: Path, file_name: str = "model.pkl") -> LRSystem:
    """
    Load a trained LR system from disk from a given folder.

    It is expected that the folder contains a file named "model.pkl" (or another name specified by file_name),
    which is a pickled LRSystem object. The function loads this object and returns it.

    The system is returned as an instance of the LRSystem class. This class provides an apply method, which can be used
    to calculate LLRs for given features. These features should be contained in a FeatureData object from lir.

    Example usage:
    ```
    from lir.data.models import FeatureData

    lr_system = get_lr_system(Path("path/to/lr_system_folder/"))

    # Create three instances of features, each with two feature values.
    features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    feature_data = FeatureData(features=features)
    llr_data = lr_system.apply(feature_data)
    ```
    """
    with (lr_system_folder / file_name).open("rb") as f:
        return pickle.load(f)  # noqa: S301


def get_reference_data(lr_system_folder: Path, file_name: str = "reference_data.csv") -> FeatureData:
    """Load reference data from disk.

    It is expected that the folder contains a file named "reference_data.csv" (or another name specified by file_name),
    which is a CSV file containing the reference data. The function loads this data and returns it as a FeatureData
    object.

    If the data has `n` features, the CSV file should have `n+1` columns. One of the columns should be named
    "hypothesis", and contain the labels for the data. The other columns should contain the feature values.

    The data is returned as an instance of the FeatureData class from lir. This class has a features and a labels
    attribute, which can be used to access the feature values and the labels, respectively.
    """
    reference_data_file = lr_system_folder / file_name
    return FeatureDataCsvFileParser(file=reference_data_file, label_column="hypothesis").get_instances()


def get_validation_experiment(model_name: str, training_data_path: Path, output_path: Path) -> tuple[Experiment, Path]:
    """
    Return an `Experiment` that builds and validates a model.

    The `lr_system_folder` will be created in `output_path` under the name "model".
    """
    yaml_file = Path(__file__).parent / "models" / model_name / "validation.yaml"
    yaml_update = {
        "output_path": str(output_path),
        "input_file_path": str(training_data_path),
    }
    cfg = confidence.Configuration(confidence.loadf(yaml_file), yaml_update)
    exps, _ = initialize_experiments(cfg)
    return exps["model"], output_path / "model"


# create an alias for the specific source system, since the architecture is identical but the name is misleading
# in the current application
binary_lrsystem = specific_source
