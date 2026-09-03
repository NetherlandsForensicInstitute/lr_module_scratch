from collections.abc import Callable
from typing import Any, Self

import numpy as np
from lir.aggregation import PlotEach
from scipy.stats import betabinom, binom, norm

from lir.algorithms.bayeserror import ELUBBounder
from lir.algorithms.mcmc import McmcModel
from lir.bounding import LLRBounder, check_type
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.transform import Transformer


elub_bounder_factory = ELUBBounder


class McmcLLRModel(Transformer):
    """
    Use Markov Chain Monte Carlo simulations to fit a statistical distribution for each of the two hypotheses.

    Using samples from the posterior distributions of the model parameters, a posterior distribution of the LR is
    obtained. The median of this distribution is used as best estimate for the LR; a credible interval is also
    determined.

    Parameters
    ----------
    distribution_h1 : str
        Statistical distribution used to model H1.
    parameters_h1 : dict[str, dict[str, float | int | str]] | None
        Parameter definitions and priors for the H1 distribution.
    distribution_h2 : str
        Statistical distribution used to model H2.
    parameters_h2 : dict[str, dict[str, float | int | str]] | None
        Parameter definitions and priors for the H2 distribution.
    bounding : Callable[[], LLRBounder] | None, optional
        Bounding method factory to prevent over-extrapolation.
    interval : tuple[float, float], optional
        Lower and upper bounds of the credible interval in range ``[0, 1]``.
    **mcmc_kwargs : Any
        Additional MCMC simulation settings passed to `McmcModel`.
    """

    def __init__(
        self,
        distribution_h1: str,
        parameters_h1: dict[str, dict[str, float | int | str]] | None,
        distribution_h2: str,
        parameters_h2: dict[str, dict[str, float | int | str]] | None,
        bounding: Callable[[], LLRBounder] | None = elub_bounder_factory,
        interval: tuple[float, float] = (0.05, 0.95),
        **mcmc_kwargs: Any,
    ):
        self.model_h1 = McmcModel(distribution_h1, parameters_h1, **mcmc_kwargs)
        self.model_h2 = McmcModel(distribution_h2, parameters_h2, **mcmc_kwargs)
        self.bounder_factory = bounding
        self.bounders: list[LLRBounder] | None = None
        self.interval = interval

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the defined model to the supplied instances.

        Parameters
        ----------
        instances : InstanceData
            Training instances.

        Returns
        -------
        Self
            Fitted model.
        """
        instances = check_type(FeatureData, instances)

        self.model_h1.fit(instances.features[instances.require_labels == 1])
        self.model_h2.fit(instances.features[instances.require_labels == 0])
        if self.bounder_factory is not None:
            # determine the bounds based on the LLRs of the training data, each sample results into an LR-system
            logp_h1 = self.model_h1.transform(instances.features)
            logp_h2 = self.model_h2.transform(instances.features)
            llrs = logp_h1 - logp_h2

            # determine the bounds for each LR-system individually
            self.bounders = [self.bounder_factory() for _ in range(llrs.shape[1])]
            for i_system in range(llrs.shape[1]):
                llr_data = LLRData(features=llrs[:, i_system], hypothesis=instances.require_labels)
                self.bounders[i_system] = self.bounders[i_system].fit(llr_data)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """
        Apply the fitted model to the supplied instances.

        Parameters
        ----------
        instances : InstanceData
            Instances to transform.

        Returns
        -------
        LLRData
            LLR estimates with median and credible interval columns.
        """
        instances = check_type(FeatureData, instances)
        logp_h1 = self.model_h1.transform(instances.features)
        logp_h2 = self.model_h2.transform(instances.features)
        llrs = logp_h1 - logp_h2
        if (self.bounder_factory is not None) and (self.bounders is not None):
            # apply the bounders one by one
            for i_system in range(llrs.shape[1]):
                llr_data = LLRData(features=llrs[:, i_system], hypothesis=instances.hypothesis)
                bound_llr_data = self.bounders[i_system].apply(llr_data)
                llrs[:, i_system] = bound_llr_data.llrs
        quantiles = np.quantile(llrs, [0.5] + list(self.interval), axis=1, method='midpoint')
        mcmc_details = {'mcmc_distribution_h1': self.model_h1.distribution,
                        'mcmc_parameters_h1': self.model_h1.parameter_samples,
                        'mcmc_distribution_h2': self.model_h2.distribution,
                        'mcmc_parameters_h2': self.model_h2.parameter_samples}
        # llr_data = instances.replace_as(LLRData, features=quantiles.transpose(1, 0), mcmc_details=mcmc_details)
        llr_data = instances.replace_as(LLRData, features=quantiles.transpose(1, 0))
        llr_data = llr_data.replace(**mcmc_details)
        return llr_data


def plot_mcmc(llrdata, ax):
    return


class MCMCParameterPlot(PlotEach):

    def __init__(self):
        super().__init__(plot_fn=plot_mcmc, plot_name='MCMC parameter plot')
