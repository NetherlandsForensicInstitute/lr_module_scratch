from hashlib import blake2b
from pathlib import Path
from typing import Optional

from lir.lrsystems.specific_source import SpecificSourceSystem


def _get_model_dirname(settings: dict[str, str]) -> str:
    return blake2b(str(sorted(settings.items())).encode("utf8")).hexdigest()


def load_model(cache_dir: Path, settings: dict[str, str]) -> Optional[SpecificSourceSystem]:
    model_path = cache_dir / _get_model_dirname(settings) / "model.pkl"
    raise NotImplementedError


def save_model(cache_dir: Path, settings: dict[str, str], model: SpecificSourceSystem) -> None:
    model_dir = cache_dir / _get_model_dirname(settings)
    raise NotImplementedError
