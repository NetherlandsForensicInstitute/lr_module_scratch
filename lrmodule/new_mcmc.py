import logging
from matplotlib import pyplot as plt

from lir.aggregation import Aggregation, AggregationData
from lir.util import check_not_none


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
        if data.get_full_fit_lrsystem is not None:
            lrsystem = data.get_full_fit_lrsystem()
        else:
            LOG.warning(
                f'No full-data-fitted model factory available for run `{run_name}`; '
                f'using split-trained model instead.'
            )
            lrsystem = check_not_none(data.lrsystem)
        mcmc_system = [x[1] for x in lrsystem.lrsystem.pipeline.steps if x[0] == 'mcmc'][0]

        for hypothesis, model in {'h1': mcmc_system.model_h1, 'h2': mcmc_system.model_h2}.items():
            for parameter_name, parameter_values in model.parameter_samples.items():

                plot_name = 'MCMC_parameter-' + hypothesis + '_' + model.distribution + '_' + parameter_name
                fig, ax = plt.subplots()

                try:
                    ax.hist(parameter_values)
                except ValueError as e:
                    LOG.warning(f'Could not generate plot {plot_name} for run `{run_name}`: {e}')
                    return

                file_name = data.resolve_path_for_run(f'{plot_name}.png')

                LOG.info(f'Saving plot {plot_name} for run `{run_name}` to {file_name}')
                fig.savefig(file_name)

                plt.close(fig)
