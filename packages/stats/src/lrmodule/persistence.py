from hashlib import sha256
from pathlib import Path

from lir.lrsystems.lrsystems import LRSystem

from lrmodule.data_types import ModelSettings


def _get_model_dirname(settings: ModelSettings, dataset_id: str) -> str:
    h = sha256()
    h.update(str(settings).encode("utf8"))
    h.update(dataset_id.encode("utf8"))
    return h.hexdigest()


def load_model(settings: ModelSettings, dataset_id: str, cache_dir: Path) -> LRSystem | None:
    """Load previously cached model."""
    # model_path = cache_dir / _get_model_dirname(settings, dataset_id) / "model.pkl"
    raise NotImplementedError


def save_model(model: LRSystem, settings: ModelSettings, dataset_id: str, cache_dir: Path) -> None:
    """Save a model to disk."""
    # model_dir = cache_dir / _get_model_dirname(settings, dataset_id)
    raise NotImplementedError
