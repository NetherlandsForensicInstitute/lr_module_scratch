from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalDataClass
from lrmodule.data_types import MarkType, ModelSettings, ScoreType
from src.lrmodule.lrsystem import load_lrsystem


def test_load_lrsystem():
    load_lrsystem(ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF))


def test_run_lrsystem():
    lrsystem = load_lrsystem(ModelSettings(MarkType.FIRING_PIN_IMPRESSION, ScoreType.ACCF))
    data = SynthesizedNormalBinaryData(
        data_classes={
            0: SynthesizedNormalDataClass(mean=-1, std=1, size=100),
            1: SynthesizedNormalDataClass(mean=1, std=1, size=100),
        },
        seed=0,
    )
    data = data.get_instances()
    data = data.replace(features=data.features.flatten())
    llrs = lrsystem.fit(data).apply(data)
    assert llrs.features.shape == (200, 3)


if __name__ == "__main__":
    test_load_lrsystem()
    test_run_lrsystem()
