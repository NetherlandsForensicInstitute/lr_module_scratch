import logging

import numpy as np
from lir.aggregation import Aggregation, AggregationData

LOG = logging.getLogger(__name__)


def transform_marktype_ccf(original_score):
    """Transform the 'ccf' score using a log transformation.

    The input values are expected to be in the range [-1, 1]. They are first scaled to [0, 1], after which a logit
    transformation is applied.
    """
    transformed_score = (original_score + 1) / 2
    transformed_score = np.log10(transformed_score / (1 - transformed_score))
    return transformed_score.reshape(-1, 1)


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

    Currently not used, but equivalent to the Matlab implementation.
    """
    cmc = original_score[:, 0]
    n = original_score[:, 1]
    return cmc / n


class FullFitLRSystem(Aggregation):
    def report(self, data: AggregationData) -> None:
        """Fit the LR-system on all available data."""
        if data.get_full_fit_lrsystem is not None:
            data.get_full_fit_lrsystem()
        else:
            LOG.warning(f"No full-data-fitted model factory available for run `{data.run_name}`.")
