import os
import pickle
from pathlib import Path

from lir.lrsystems.lrsystems import LRSystem

from lrmodule.data_types import ModelSettings


def _get_model_filename(settings: ModelSettings) -> str:
    """Construct model filename based on mark and score type."""
    mark_type = settings.mark_type.value
    score_type = settings.score_type.value

    return f"{mark_type}_{score_type}_model.pkl"


def load_model(settings: ModelSettings, model_storage_path: Path) -> LRSystem:
    """Load previously cached model."""
    model_filename = _get_model_filename(settings)
    model_file_path = model_storage_path / model_filename

    mark_type = settings.mark_type.value
    score_type = settings.score_type.value

    if not model_file_path.exists():
        raise FileNotFoundError(f"No model found for mark type '{mark_type}', score type: '{score_type}'.")

    try:
        with open(model_file_path, "rb") as f:
            # It is assumed exclusively `LRSystem` models will be loaded, which are considered safe
            return pickle.load(f)  # noqa: S301
    except Exception:
        raise RuntimeError(
            f"Could not load model from .pkl file for mark type '{mark_type}', score type: '{score_type}'"
        )


def save_model(model: LRSystem, settings: ModelSettings, model_storage_path: Path) -> None:
    """Save a model to disk."""
    model_filename = _get_model_filename(settings)
    model_file_path = model_storage_path / model_filename

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        f.write(pickle.dumps(model))
