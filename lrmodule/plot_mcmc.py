import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import numpy as np
from KDEpy import FFTKDE
from lir.aggregation import Aggregation, AggregationData
from lir.algorithms.bayeserror import ELUBBounder
from lir.algorithms.mcmc import McmcModel
from lir.bounding import LLRBounder, check_type
from lir.config.base import ConfigValue, config_parser
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.transform import Transformer
from matplotlib import pyplot as plt

LOG = logging.getLogger(__name__)


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

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        distribution_h1: str,
        parameters_h1: dict[str, dict[str, float | int | str]] | None,
        distribution_h2: str,
        parameters_h2: dict[str, dict[str, float | int | str]] | None,
        bounding: Callable[[], LLRBounder] | None = elub_bounder_factory,
        interval: tuple[float, float] = (0.05, 0.95),
        include_parameter_plots: bool = False,
        plot_path: Path | None = None,
        **mcmc_kwargs: Any,
    ):
        self.model_h1 = McmcModel(distribution_h1, parameters_h1, **mcmc_kwargs)
        self.model_h2 = McmcModel(distribution_h2, parameters_h2, **mcmc_kwargs)
        self.bounder_factory = bounding
        self.bounders: list[LLRBounder] | None = None
        self.interval = interval
        self.include_parameter_plots = include_parameter_plots
        self.plot_path = plot_path
        self.plot_count = 0

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

        # optionally, plot distributions of the sampled distribution parameters
        if self.include_parameter_plots and self.plot_path is not None:
            self.plot_count += 1
            hypothesis_models = {"h1": self.model_h1, "h2": self.model_h2}
            for hypothesis, model in hypothesis_models.items():
                for parameter_name, parameter_values in model.parameter_samples.items():
                    plot_name = "distribution-" + hypothesis + "_" + model.distribution + "_" + parameter_name
                    fig, ax = plt.subplots()

                    try:
                        x, y = FFTKDE(bw="silverman").fit(parameter_values).evaluate(2**10)
                        ax.plot(x, y)
                        ax.set_xlabel(parameter_name)
                        ax.set_ylabel("probability density")
                    except ValueError as e:
                        LOG.warning(f"Could not generate plot {plot_name}: {e}")
                        continue

                    file_name = self.plot_path / f"{self.plot_count:02d}-{plot_name}.png"

                    LOG.info(f"Saving plot {plot_name} to {file_name}")
                    fig.savefig(file_name)

                    plt.close(fig)

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
        quantiles = np.quantile(llrs, [0.5] + list(self.interval), axis=1, method="midpoint")
        return instances.replace_as(LLRData, features=quantiles.transpose(1, 0))


@config_parser
def parse_mcmc_llr_model_config(config: ConfigValue, output_dir: Path) -> McmcLLRModel:
    """Add output folder if parameter plots are requested."""
    config_dict = config.as_dict()
    if "include_parameter_plots" in config_dict and config_dict["include_parameter_plots"]:
        folder_name = "mcmc_output"
        mcmc_folders = [f.name for f in os.scandir(output_dir) if f.is_dir() and f.name.startswith(folder_name)]
        plot_path = output_dir / f"{folder_name}-{len(mcmc_folders) + 1:02d}"
        plot_path.mkdir(parents=True, exist_ok=True)
    else:
        plot_path = None
    return McmcLLRModel(**config_dict, plot_path=plot_path)


class FullFitLRSystem(Aggregation):
    def report(self, data: AggregationData) -> None:
        """Fit the LR-system on all available data."""
        if data.get_full_fit_lrsystem is not None:
            data.get_full_fit_lrsystem()
        else:
            LOG.warning(f"No full-data-fitted model factory available for run `{data.run_name}`.")
