# LR module for Scratch

This work accompanies the corresponding [Scratch repository](https://github.com/NetherlandsForensicInstitute/scratch).
Within this specific repository, the LR module - responsible for calculating the appropriate statistical data - is provided.

From the `lrmodule` python module, several public API methods are exposed:
 - `get_lr_system`: load a trained LR system from disk from a given folder;
 - `get_reference_data`: load reference data from disk;
 - `get_validation_experiment`: return an `Experiment` that builds and validates a model.


## Local development

1. Install all dependencies using `pdm sync -G dev` (to install dev dependencies as well)
2. Run checks with `pdm run check` or `pdm check-quality` to automatically fix the things as well
3. Run tests with `pdm run test`
4. To run everything `pdm run all` (no auto fixes) or with fixes: `pdm run all-fix`

All typing, linting and formatting configuration was taken from the Scratch repository for seamless integration.


## Model validation

1. Update hyperparameters, experiment setup and data path as needed in `models/[NAME]/validation.yaml`;
2. Run validation experiments as `pdm run lir models/[NAME]/validation.yaml`;
3. Inspect the results in the output folder;
4. When satisfied, update the stored model (TODO).

## Debugging in PyCharm

For example, instead of `pdm run lir models/[NAME]/validation.yaml`:

- Create a new 'Run/Debug configuration', with the following settings:
- Select the 'Python' template/default to start from
- Instead of 'script' select 'module', and specify `lir`
- As 'Script parameters', specify `lrmodule/models/[NAME]/validation.yaml`
- Make that the path to this repo is used as 'Working directory'

See the [PyCharm documentation](https://www.jetbrains.com/help/pycharm/run-debug-configuration.html) for more info.
