from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#  .../lr_module_scratch/ has to be the working directory further on

statuses = [True, False]

for no_bounds in statuses:

    #  Read score results for all types of traces
    bf_scores_path = "data/2026-Q4/all_types/breech_face_impression.csv"
    fp_scores_path = "data/2026-Q4/all_types/firing_pin_impression.csv"
    as_scores_path = "data/2026-Q4/all_types/aperture_shear.csv"

    bf_scores = pd.read_csv(bf_scores_path)
    fp_scores = pd.read_csv(fp_scores_path)
    as_scores = pd.read_csv(as_scores_path)

    #  Read LR results for all types of traces
    bf_lrs_path = "output/all_types_independence_filtered/bf_cmc/modules.csv"
    fp_lrs_path = "output/all_types_independence_filtered/fp_cmc/modules.csv"
    as_lrs_path = "output/all_types_independence_filtered/as_ccf/modules.csv"

    bf_lrs = pd.read_csv(bf_lrs_path)
    fp_lrs = pd.read_csv(fp_lrs_path)
    as_lrs = pd.read_csv(as_lrs_path)

    if no_bounds:
        output_folder = Path("output/all_types_independence_filtered/no_bounds")
    else:
        output_folder = Path("output/all_types_independence_filtered/bounds")
    output_folder.mkdir(exist_ok=True)

    #  Merge score results for each pair of types of traces
    bf_fp_scores_merged = pd.merge(bf_scores, fp_scores, on=['Weapon1', 'Weapon2'], how='inner')
    bf_as_scores_merged = pd.merge(bf_scores, as_scores, on=['Weapon1', 'Weapon2'], how='inner')
    fp_as_scores_merged = pd.merge(fp_scores, as_scores, on=['Weapon1', 'Weapon2'], how='inner')
    # Add cmc/n column for BF and FP
    bf_fp_scores_merged['cmc/n bf'] = bf_fp_scores_merged['Matching Cells_x']/bf_fp_scores_merged['Total Cells_x']
    bf_fp_scores_merged['cmc/n fp'] = bf_fp_scores_merged['Matching Cells_y']/bf_fp_scores_merged['Total Cells_y']
    bf_as_scores_merged['cmc/n bf'] = bf_as_scores_merged['Matching Cells']/bf_as_scores_merged['Total Cells']
    fp_as_scores_merged['cmc/n fp'] = fp_as_scores_merged['Matching Cells']/fp_as_scores_merged['Total Cells']

    def plot_conditional_scores(merged_data, name_x, name_y, name_output):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(data=merged_data[merged_data['Hypothesis_x'] == 1], x=name_x, y=name_y,
                        ax=ax1, color="steelblue", s=10)
        ax1.set_title("Hypothesis 1")
        sns.scatterplot(data=merged_data[merged_data['Hypothesis_x'] == 0], x=name_x, y=name_y,
                        ax=ax2, color="coral", s=10)
        ax2.set_title("Hypothesis 2")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{name_output}", dpi=300)

    # plot BF versus FP scores under H1 and H2
    plot_conditional_scores(bf_fp_scores_merged, "cmc/n bf", "cmc/n fp", 'bf_fp_scores')
    # plot BF versus AS scores under H1 and H2
    plot_conditional_scores(bf_as_scores_merged, "cmc/n bf", "Score", 'bf_as_scores')
    # plot FP versus AS scores under H1 and H2
    plot_conditional_scores(fp_as_scores_merged, "cmc/n fp", "Score", 'fp_as_scores')

    # Merge LR results for comparison of all two types of traces
    bf_fp_lrs_merged = pd.merge(bf_lrs, fp_lrs, on=['source_ids.0', 'source_ids.1'], how='inner')
    bf_as_lrs_merged = pd.merge(bf_lrs, as_lrs, on=['source_ids.0', 'source_ids.1'], how='inner')
    fp_as_lrs_merged = pd.merge(fp_lrs, as_lrs, on=['source_ids.0', 'source_ids.1'], how='inner')

    def plot_conditional_lrs(merged_data, name_x, name_y, type_1_name, type_2_name, name_output):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(data=merged_data[merged_data['label_x'] == 1], x=name_x, y=name_y, ax=ax1,
                        color="steelblue", s=10)
        ax1.set_title("Hypothesis 1")
        sns.scatterplot(data=merged_data[merged_data['label_x'] == 0], x=name_x, y=name_y, ax=ax2,
                        color="coral", s=10)
        ax2.set_title("Hypothesis 2")
        plt.tight_layout()
        ax1.set_xlabel(type_1_name + " LR score")
        ax1.set_ylabel(type_2_name + " LR score")
        ax2.set_xlabel(type_1_name + " LR score")
        ax2.set_ylabel(type_2_name + " LR score")
        plt.savefig(f"{output_folder}/{name_output}", dpi=300)

    # Plot LLR results of two types of trace, under both hypotheses
    if no_bounds:
        plot_conditional_lrs(bf_fp_lrs_merged, "mcmc.0_x", "mcmc.0_y", "BF", "LR", "bf_fp_lrs")
        plot_conditional_lrs(bf_as_lrs_merged, "mcmc.0", "bootstrap.1", "BF", "AS", "bf_as_lrs")
        plot_conditional_lrs(fp_as_lrs_merged, "mcmc.0", "bootstrap.1", "FP", "AS", "fp_as_lrs")
    else:
        plot_conditional_lrs(bf_fp_lrs_merged, "bounder.0_x", "bounder.0_y", "BF", "LR", "bf_fp_lrs")
        plot_conditional_lrs(bf_as_lrs_merged, "bounder.0", "bootstrap.1", "BF", "AS", "bf_as_lrs")
        plot_conditional_lrs(fp_as_lrs_merged, "bounder.0", "bootstrap.1", "FP", "AS", "fp_as_lrs")

a = 1