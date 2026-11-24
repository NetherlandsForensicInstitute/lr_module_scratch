from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sample_size = 10**5
rng = np.random.default_rng()

bin_width_h1 = 0.01
bin_width_h2 = 0.002

#  .../lr_module_scratch/ has to be the working directory further on
# it is assumed that full_glock_mcmc.yaml has been run using lir

bf_scores_path = "data/2026-Q4/all_types/breech_face_impression.csv"
fp_scores_path = "data/2026-Q4/all_types/firing_pin_impression.csv"
as_scores_path = "data/2026-Q4/all_types/aperture_shear.csv"

bf_scores = pd.read_csv(bf_scores_path)
fp_scores = pd.read_csv(fp_scores_path)
as_scores = pd.read_csv(as_scores_path)

output_folder = Path("output/all_glocks_mcmc/model_fit_plots")
output_folder.mkdir(exist_ok=True)


def parameters_csv(path):
    pathfile_alphas = path + "batch_01-h1_betabinomial_alpha-values.csv"
    alphas = pd.read_csv(pathfile_alphas, header=None)

    pathfile_betas = path + "batch_01-h1_betabinomial_beta-values.csv"
    betas = pd.read_csv(pathfile_betas, header=None)

    pathfile_ps = path + "batch_01-h2_binomial_p-values.csv"
    ps = pd.read_csv(pathfile_ps, header=None)

    return alphas.to_numpy(), betas.to_numpy(), ps.to_numpy()


bf_alpha, bf_betas, bf_ps = parameters_csv(bf_path)
fp_alpha, fp_betas, fp_ps = parameters_csv(fp_path)


bf_total_cells_h1 = bf_scores[bf_scores["Hypothesis"] == 1]["Total Cells"].to_numpy()
bf_total_cells_h2 = bf_scores[bf_scores["Hypothesis"] == 0]["Total Cells"].to_numpy()

fp_total_cells_h1 = fp_scores[fp_scores["Hypothesis"] == 1]["Total Cells"].to_numpy()
fp_total_cells_h2 = fp_scores[fp_scores["Hypothesis"] == 0]["Total Cells"].to_numpy()

# comparison takes place of new observed relative cmc values to values corresponding to the probability model
# for breech face this means the following:
# the mcmc procedure rendered a trace of values for either alpha and beta (H1) or the success parameter p (H2)
# we cannot directly compare these to the observed relative cmc values
# therefore we take new samples of success parameters (under H1) or we have them already (under H2)
# and given these, we take samples of simulated cmc scores, which in turn we divide by the corresponding number
# of total cells
def sample_fraction_cmc_bf(total_cells, probs):
    sample = rng.binomial(total_cells, probs) / total_cells
    sample = sample.ravel()
    sample = rng.choice(sample, sample_size, replace=False)

    return sample

# we basically do the same here, but since firing pin yields no values of zero, we sample from the
# distribution of outcomes minus 1, and then after that again add 1
def sample_fraction_cmc_fp(total_cells, probs):
    sample = (rng.binomial(total_cells, probs) + 1) / (total_cells + 1)
    sample = sample.ravel()
    sample = rng.choice(sample, sample_size, replace=False)

    return sample


bf_sample_h1_probs = rng.beta(bf_alpha, bf_betas)
bf_sample_h1 = sample_fraction_cmc_bf(bf_total_cells_h1, bf_sample_h1_probs)
bf_sample_h1 = pd.DataFrame(bf_sample_h1, columns=["pct"])
bf_sample_h2 = sample_fraction_cmc_bf(bf_total_cells_h2, bf_ps)
bf_sample_h2 = pd.DataFrame(bf_sample_h2, columns=["pct"])


fp_sample_h1_probs = rng.beta(fp_alpha, fp_betas)
fp_sample_h1 = sample_fraction_cmc_fp(fp_total_cells_h1, fp_sample_h1_probs)
fp_sample_h1 = pd.DataFrame(fp_sample_h1, columns=["pct"])
fp_sample_h2 = sample_fraction_cmc_fp(fp_total_cells_h2, fp_ps)
fp_sample_h2 = pd.DataFrame(fp_sample_h2, columns=["pct"])


def plot_data(ax, model, data, hyp=None, name="", marktype=""):

    if hyp == 1:
        binwidth = bin_width_h1
    else:
        binwidth = bin_width_h2

    data["pct"] = data["Matching Cells"] / data["Total Cells"]
    sns.histplot(data=model, x="pct", binwidth=binwidth, kde=False, stat="density", color="red", ax=ax)
    sns.histplot(data=data, x="pct", binwidth=binwidth, kde=False, stat="density", color="green", ax=ax)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{marktype}: Model H{hyp} versus {name} ")
    plt.tight_layout()


def plot_hypothesis(df_hX, sample_hX, X, marktype):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4))

    df_nn = df_hX[(df_hX["Material1"] == "Nickel") & (df_hX["Material2"] == "Nickel")]
    plot_data(ax[0, 0], sample_hX, df_nn, X, "Nickel-Nickel", marktype)

    df_bb = df_hX[(df_hX["Material1"] == "Brass") & (df_hX["Material2"] == "Brass")]
    plot_data(ax[0, 1], sample_hX, df_bb, X, "Brass-Brass", marktype)

    df_nb = df_hX[df_hX["Material1"] != df_hX["Material2"]]
    plot_data(ax[1, 0], sample_hX, df_nb, X, "Nickel-Brass", marktype)

    plot_data(ax[1, 1], sample_hX, df_hX, X, "all", marktype)

    plt.savefig(f"{output_folder}/{marktype}_H{X}", dpi=300)


def plot_h1_and_h2(scores, samples_h1, samples_h2, marktype):
    df_h1 = scores[(scores["Hypothesis"] == 1)]
    df_h2 = scores[(scores["Hypothesis"] == 0)]

    plot_hypothesis(df_h1, samples_h1, 1, marktype)
    plot_hypothesis(df_h2, samples_h2, 2, marktype)


plot_h1_and_h2(bf_scores, bf_sample_h1, bf_sample_h2, "BF")
plot_h1_and_h2(fp_scores, fp_sample_h1, fp_sample_h2, "FP")

plt.show()
