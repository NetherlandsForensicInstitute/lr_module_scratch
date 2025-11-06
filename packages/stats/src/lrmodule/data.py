from hashlib import sha1

from lir.data.models import FeatureData


def get_dataset_id(data: FeatureData) -> str:
    """obtain a unique identifier for a data set"""
    return sha1(str(data.get_instances()).encode("utf8")).hexdigest()
