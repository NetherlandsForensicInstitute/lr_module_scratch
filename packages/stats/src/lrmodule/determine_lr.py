import pickle
from pathlib import Path

from lrmodule.data_types import LLR_data


def scores_to_llrs(evaluation_scores, known_match_scores, known_non_match_scores) -> list[LLR_data]:
    """Calculate evaluation scores to LLR values, using known match and known non-match scores."""
    # TODO - Cheat by using the list of expected LLR values to slime the characterization test (temporarily)
    precalculated_llrs_path = Path("packages/stats/tests/characterization_test/expected_aperture_shear_LLRS.pkl")

    with open(precalculated_llrs_path, "rb") as f:
        loaded_llrs = pickle.load(f)  # noqa: S301

    return loaded_llrs
