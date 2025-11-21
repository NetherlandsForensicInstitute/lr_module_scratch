from hashlib import sha256
from pathlib import Path

from lir.lrsystems.lrsystems import LRSystem

from lrmodule.data_types import ModelSettings


# def _get_model_dirname(settings: ModelSettings, dataset_id: str) -> str:
#     h = sha256()
#     h.update(str(settings).encode("utf8"))
#     h.update(dataset_id.encode("utf8"))
#     return h.hexdigest()


def load_model(settings: ModelSettings, dataset_id: str, model_storage_path: Path) -> LRSystem | None:
    """Load previously cached model."""
    mark_type = settings.mark_type
    score_type = settings.score_type

    model_file_path = model_storage_path / f"{mark_type}_{score_type}.pkl"
    if not model_file_path.exists():
        raise FileNotFoundError(f"No model found for mark type '{mark_type}', score type: '{score_type}'.")

    try:

    raise NotImplementedError


def save_model(model: LRSystem, settings: ModelSettings, dataset_id: str, model_storage_path: Path) -> None:
    """Save a model to disk."""
    _ = model_storage_path / _get_model_dirname(settings, dataset_id)
    raise NotImplementedError
