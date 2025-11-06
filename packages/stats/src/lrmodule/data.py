from hashlib import sha1

from lir.data.models import DataSet


def get_dataset_id(data: DataSet) -> str:
    return sha1(str(data.get_instances()).encode("utf8")).hexdigest()
