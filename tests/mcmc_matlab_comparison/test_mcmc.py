from pathlib import Path

import confidence
import numpy as np
import pandas as pd
import pytest
from lir.data.models import FeatureData

from lrmodule.mcmc import McmcLLRModel, McmcModel

# Tests against references from the Matlab implementation.

base_directory = Path(__file__).parent
cfg = confidence.loadf(base_directory / "config.yaml")


def test_fit_parameter_stats():
    # Conversion from observed score data to (statistics of) model parameter samples.
    # We expect some differences here due to mcmc-differences and randomness.

    csv_prefix = "firing_pin_impression-" + cfg.dataset.score
    df_scores = pd.read_csv(base_directory / (csv_prefix + "-all-input.csv"), header=None)

    # beta-binomial model
    scores_h1 = np.array(df_scores[df_scores[0] == "km"].iloc[:, 1:], dtype=np.int16)
    model_h1 = McmcModel(cfg.distribution_h1, cfg.parameters_h1, random_seed=cfg.random_seed).fit(scores_h1)
    stats_calc = np.array([
        [np.mean(model_h1.parameter_samples["alpha"]), np.std(model_h1.parameter_samples["alpha"])],
        [np.mean(model_h1.parameter_samples["beta"]), np.std(model_h1.parameter_samples["beta"])],
    ])
    stats_ref = np.loadtxt(base_directory / (csv_prefix + "-km_parameter_stats.csv"), delimiter=",")
    assert np.allclose(stats_calc, stats_ref, rtol=2e-2, atol=2e-2)

    # binomial model
    scores_h2 = np.array(df_scores[df_scores[0] == "knm"].iloc[:, 1:], dtype=np.int16)
    model_h2 = McmcModel(cfg.distribution_h2, cfg.parameters_h2, random_seed=cfg.random_seed).fit(scores_h2)
    stats_calc = np.array([np.mean(model_h2.parameter_samples["p"]), np.std(model_h2.parameter_samples["p"])])
    stats_ref = np.loadtxt(base_directory / (csv_prefix + "-knm_parameter_stats.csv"), delimiter=",")
    assert np.allclose(stats_calc, stats_ref, rtol=1e-4, atol=1e-4)


def test_transform_quantiles():
    # Conversion from model parameter samples to (percentiles of) log10-probabilities.

    csv_prefix = "firing_pin_impression-" + cfg.dataset.score
    df_scores = pd.read_csv(base_directory / (csv_prefix + "-all-input.csv"), header=None)
    scores_eval = np.array(df_scores[df_scores[0] == "eval"].iloc[:, 1:], dtype=np.int16)

    # betabinomial model
    alpha, beta = np.loadtxt(base_directory / (csv_prefix + "-km_parameter_samples.csv"), delimiter=",", unpack=True)
    model_h1 = McmcModel(cfg.distribution_h1, None)
    model_h1.parameter_samples = {"alpha": alpha, "beta": beta}
    logp_eval = model_h1.transform(scores_eval)
    logp_calc = np.quantile(logp_eval, [0.05, 0.5, 0.95], axis=1, method="midpoint")
    logp_ref = np.loadtxt(base_directory / (csv_prefix + "-km_logp.csv"), delimiter=",")
    assert np.allclose(logp_calc, logp_ref, rtol=1e-4, atol=1e-4)

    # binomial model
    model_h2 = McmcModel(cfg.distribution_h2, None)
    model_h2.parameter_samples = {
        "p": np.loadtxt(base_directory / (csv_prefix + "-knm_parameter_samples.csv"), delimiter=",")
    }
    logp_eval = model_h2.transform(scores_eval)
    logp_calc = np.quantile(logp_eval, [0.05, 0.5, 0.95], axis=1, method="midpoint")
    logp_ref = np.loadtxt(base_directory / (csv_prefix + "-knm_logp.csv"), delimiter=",")
    assert np.allclose(logp_calc, logp_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dataset_name",
    [
        "firing_pin_impression",
        "breech_face_impression",
    ],
)
def test_llr_dataset(dataset_name: str):
    # From pre-calculated scores to (percentiles of) log-10 lrs.
    # We expect some differences here due to mcmc-differences and randomness.

    csv_prefix = f"{dataset_name}-" + cfg.dataset.score
    df_scores = pd.read_csv(base_directory / (csv_prefix + "-all-input.csv"), header=None)

    scores_eval = np.array(df_scores[df_scores[0] == "eval"].iloc[:, 1:], dtype=np.int16)
    scores_km = np.array(df_scores[df_scores[0] == "km"].iloc[:, 1:], dtype=np.int16)
    scores_knm = np.array(df_scores[df_scores[0] == "knm"].iloc[:, 1:], dtype=np.int16)

    features = np.concatenate([scores_km, scores_knm])
    labels = np.concatenate([np.ones(scores_km.shape[0]), np.zeros(scores_knm.shape[0])])

    model = McmcLLRModel(cfg.distribution_h1, cfg.parameters_h1,
                         cfg.distribution_h2, cfg.parameters_h2,
                         random_seed=cfg.random_seed)
    model.fit(FeatureData(features=features, labels=labels))
    llrs = model.transform(FeatureData(features=scores_eval))
    llrs_ref = np.loadtxt(base_directory / (csv_prefix + "-llr_unbound.csv"), delimiter=",")
    assert np.allclose(llrs.llrs, llrs_ref[1], rtol=5e-2, atol=5e-2)
    assert np.allclose(llrs.llr_intervals[:, 0], llrs_ref[0], rtol=5e-2, atol=5e-2)  # type: ignore
    assert np.allclose(llrs.llr_intervals[:, 1], llrs_ref[2], rtol=5e-2, atol=5e-2)  # type: ignore
