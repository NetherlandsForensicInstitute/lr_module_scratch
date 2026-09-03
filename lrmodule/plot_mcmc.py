import logging
from matplotlib import pyplot as plt

from lir.aggregation import Aggregation, AggregationData


LOG = logging.getLogger(__name__)


class MCMCParameterPlot(Aggregation):

    # Based on PlotEach and CaseLLRToCsv

    def report(self, data: AggregationData) -> None:
        """
        Plot the data when new results are available.

        Parameters
        ----------
        data : AggregationData
            The aggregated data to be plotted.
        """

        run_name = data.run_name
        lrsystem = data.get_full_fit_lrsystem()

        mcmc_system = [x[1] for x in lrsystem.lrsystem.pipeline.steps if x[0] == 'mcmc']
        if not mcmc_system:
            raise ValueError('expected an `mcmc` step in the lrsystem pipeline, but found none.')
        else:
            mcmc_system = mcmc_system[0]

        hypothesis_models = {'h1': mcmc_system.model_h1, 'h2': mcmc_system.model_h2}
        for hypothesis, model in hypothesis_models.items():
            for parameter_name, parameter_values in model.parameter_samples.items():

                plot_name = 'MCMC_parameter-' + hypothesis + '_' + model.distribution + '_' + parameter_name
                fig, ax = plt.subplots()

                try:
                    ax.hist(parameter_values, density=True)
                    ax.set_xlabel(parameter_name)
                    ax.set_ylabel('probability density')
                except ValueError as e:
                    LOG.warning(f'Could not generate plot {plot_name} for run `{run_name}`: {e}')
                    return

                file_name = data.resolve_path_for_run(f'{plot_name}.png')

                LOG.info(f'Saving plot {plot_name} for run `{run_name}` to {file_name}')
                fig.savefig(file_name)

                plt.close(fig)
