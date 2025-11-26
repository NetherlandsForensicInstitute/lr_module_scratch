from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem


def test_run_lrsystem(trained_lr_system: LRSystem, sample_feature_data: FeatureData):
    llrs = trained_lr_system.apply(sample_feature_data)
    assert llrs.features.shape == (200, 3)
