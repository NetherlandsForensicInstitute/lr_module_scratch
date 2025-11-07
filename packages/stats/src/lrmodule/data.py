from lir.data.models import FeatureData


def get_dataset_id(data: FeatureData) -> str:
    """Obtain a unique identifier for a data set."""
    # TODO: `get_instances()` bestaat niet op `FeatureData`
    # return sha256(str(data.get_instances()).encode("utf8")).hexdigest()
    return ""
