# from hashlib import sha256
#
# from lrmodule.data_types import ModelSettings

# from lir.lrsystems.binary_lrsystem import BinaryLRSystem


# def _get_model_dirname(settings: ModelSettings, dataset_id: str) -> str:
#     h = sha256()
#     h.update(str(settings).encode("utf8"))
#     h.update(dataset_id.encode("utf8"))
#     return h.hexdigest()


# def load_model(settings: ModelSettings, dataset_id: str, cache_dir: Path) -> BinaryLRSystem | None:
#     """Load previously cached model."""
#     cache_dir / _get_model_dirname(settings, dataset_id) / "model.pkl"
#     raise NotImplementedError
#
#
# def save_model(model: BinaryLRSystem, cache_dir: Path) -> None:
#     """Save a model to disk."""
#     cache_dir / _get_model_dirname(model.settings, model.dataset_id)
#     raise NotImplementedError
