from hashlib import sha256

from lir.data.models import FeatureData


def get_dataset_id(data: FeatureData) -> str:
    """Obtain a unique identifier for a data set."""
    h = sha256()
    h.update(data.features)
    if data.has_labels:
        h.update(data.labels)
    return h.hexdigest()
