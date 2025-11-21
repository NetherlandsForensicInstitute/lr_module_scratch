import os
import pickle
from hashlib import sha256
from pathlib import Path

from lir.lrsystems.lrsystems import LRSystem

from lrmodule.data_types import ModelSettings


def load_model(settings: ModelSettings, model_storage_path: Path) -> LRSystem:
    """Load previously cached model."""
    mark_type = settings.mark_type
    score_type = settings.score_type

    model_file_path = model_storage_path / f"{mark_type}_{score_type}.pkl"
    if not model_file_path.exists():
        raise FileNotFoundError(f"No model found for mark type '{mark_type}', score type: '{score_type}'.")

    try:
        return pickle.load(model_file_path)
    except Exception:
        raise RuntimeError(
            f"Could not load model from .pkl file "
            f"for mark type '{mark_type}', score type: '{score_type}'"
        )


def save_model(model: LRSystem, settings: ModelSettings, model_storage_path: Path) -> None:
    """Save a model to disk."""
    mark_type = settings.mark_type.value
    score_type = settings.score_type.value

    model_file_path = model_storage_path / mark_type / score_type / "model.pkl"

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, 'wb') as f:
        f.write(pickle.dumps(model))

