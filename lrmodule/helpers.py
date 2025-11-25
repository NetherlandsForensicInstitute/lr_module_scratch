import numpy as np


def select_marktype_ccf(p):
    """Select the 'ccf' marktype from the input features.

    As there is only one feature, this function just returns the input as is.
    """
    return p.reshape(-1, 1)


def transform_marktype_ccf(original):
    """Transform the 'ccf' marktype using a log transformation.

    The input values are expected to be in the range [-1, 1]. They are first scaled to [0, 1], and then the logged.
    """
    transformed = (original + 1) / 2
    transformed = np.log10(transformed / (1 - transformed))
    return transformed.reshape(-1, 1)


# def select_marktype_accf(p):
#     print("Selecting marktype")
#     print(p)
#     return p


# def transform_marktype_accf(p):
#     print("Transforming marktype")
#     return p


# def select_marktype_cmc(p):
#     print("Selecting marktype")
#     print(p)
#     return p


# def transform_marktype_cmc(p):
#     print("Transforming marktype")
#     return p
