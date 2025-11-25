import numpy as np
from lir.data.models import FeatureData


def select_marktype_ccf(p: np.ndarray) -> np.ndarray:
    """Select the 'ccf' marktype from the input features.

    As there is only one feature, this function just returns the input as is.
    """
    return p


def transform_marktype_ccf(original):
    """Transform the 'ccf' marktype using a log transformation.

    The input values are expected to be in the range [-1, 1]. They are first scaled to [0, 1], and then the logit'ed..
    """
    transformed = (original + 1) / 2
    transformed = np.log10(transformed / (1 - transformed))
    return transformed.reshape(-1, 1)


def select_marktype_accf(p: np.ndarray) -> np.ndarray:
    """Select the 'accf' marktype from the input features."""
    return p[:, 0]


def transform_marktype_accf(original):
    """Transform the 'accf' marktype using a log transformation.

    The input values are expected to be in the range [-100, 100]. They are first scaled to [0, 1], and then logit'ed.
    """
    transformed = original / 100
    transformed = (transformed + 1) / 2
    transformed = np.log10(transformed / (1 - transformed))
    return transformed.reshape(-1, 1)


def transform_marktype_rel_cmc(p: np.ndarray) -> np.ndarray:
    """Transform the 'rel_cmc' marktype by calculating the ratio between the two columns.

    Currently not used, but equivalent to the Matlab implementation.
    """
    print("Transforming marktype")
    cmc = p[:, 1]
    n = p[:, 2]
    return cmc / n


def select_marktype_cmc(p: np.ndarray) -> np.ndarray:
    """Select 'cmc' and 'n'  the input features."""
    r = p[:, 1:3]
    if np.any(r[:, 1] - r[:, 0] < 0):
        raise ValueError("n must be larger than or equal to cmc")
    return r
