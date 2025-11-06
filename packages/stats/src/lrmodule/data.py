# from hashlib import sha256
#
# from lir.data.models import FeatureData


# def get_dataset_id(data: FeatureData) -> str:
#     """Obtain a unique identifier for a data set."""
#     return sha256(str(data.get_instances()).encode("utf8")).hexdigest()
