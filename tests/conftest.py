import pytest
from lir.data.models import FeatureData
from lir.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalData
from lir.lrsystems.lrsystems import LRSystem

from lrmodule.data_types import MarkType, ModelSettings, ScoreType
from lrmodule.lrsystem import load_lrsystem


@pytest.fixture
def sample_feature_data() -> FeatureData:
    """Provide FeatureData collection of synthesized normal binary data."""
    data = SynthesizedNormalBinaryData(
        SynthesizedNormalData(mean=1, std=1, size=100),
        SynthesizedNormalData(mean=-1, std=1, size=100),
        seed=0,
    )
    data = data.get_instances()
    data = data.replace(features=data.features.flatten())

    return data


@pytest.fixture
def trained_lr_system(sample_feature_data: FeatureData) -> LRSystem:
    """Provide a basic trained LR system model based on specific settings and data."""
    lrsystem = load_lrsystem(ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF))
    lrsystem.fit(sample_feature_data)

    return lrsystem
