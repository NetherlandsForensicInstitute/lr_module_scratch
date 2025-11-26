import numpy as np


def transform_marktype_ccf(original_score):
    """Transform the 'ccf' score using a log transformation.

    The input values are expected to be in the range [-1, 1]. They are first scaled to [0, 1], after which a logit
    transformation is applied.
    """
    transformed_score = (original_score + 1) / 2
    transformed_score = np.log10(transformed_score / (1 - transformed_score))
    return transformed_score.reshape(-1, 1)


def select_marktype_accf(all_features: np.ndarray) -> np.ndarray:
    """Select the 'accf' score from the input features."""
    return all_features[:, 0]


def transform_marktype_accf(original_score):
    """Transform the 'accf' score using a log transformation.

    The input values are expected to be in the range [-100, 100]. They are first scaled to [0, 1], after which a logit
    transformation is applied.
    """
    transformed_score = original_score / 100
    transformed_score = (transformed_score + 1) / 2
    transformed_score = np.log10(transformed_score / (1 - transformed_score))
    return transformed_score.reshape(-1, 1)


def transform_marktype_rel_cmc(original_score: np.ndarray) -> np.ndarray:
    """Transform the 'rel_cmc' score by calculating the ratio between the two columns.

    When used, select_marktype_cmc should be used first to select the relevant columns.
    Currently not used, but equivalent to the Matlab implementation.
    """
    cmc = original_score[:, 0]
    n = original_score[:, 1]
    return cmc / n


def select_marktype_cmc(all_features: np.ndarray) -> np.ndarray:
    """Select 'cmc' and 'n' from the input features."""
    # The 'cmc' is expected to be in column 1 and 'n' in column 2.
    relevant_features = all_features[:, 1:3]
    if np.any(relevant_features[:, 1] - relevant_features[:, 0] < 0):
        raise ValueError("n must be larger than or equal to cmc")
    return relevant_features
