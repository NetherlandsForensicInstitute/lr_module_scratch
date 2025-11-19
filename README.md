# LR module for Scratch

This work accompanies the corresponding Scratch repository: https://github.com/NetherlandsForensicInstitute/scratch.
Within this specific repository, the LR module - responsible for calculating the appropriate statistical data - is provided.

From the `lrmodule` python module, several public API methods are exposed: 
 - ... (TODO)


## Local development

1. Install all dependencies using `pdm sync -G dev` (to install dev dependencies as well)
2. Run checks with `pdm run check` or `pdm check-quality` to automatically fix the things as well
3. Run tests with `pdm run test`
4. To run everything `pdm run all` (no auto fixes) or with fixes: `pdm run fix-all`

All typing, linting and formatting configuration was taken from the Scratch repository for seamless integration.


## Model validation

1. Update hyperparameters, experiment setup and data path as needed in `validation.yaml`;
2. Run validation experiments as `pdm run lir validation.yaml`;
3. Inspect the results in the output folder;
4. When satisfied, update the stored model (TODO).
