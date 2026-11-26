from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#  .../lr_module_scratch/ has to be the working directory further on

bf_scores_path = "data/2026-Q4/all_types/breech_face_impression.csv"
fp_scores_path = "data/2026-Q4/all_types/firing_pin_impression.csv"
as_scores_path = "data/2026-Q4/all_types/aperture_shear.csv"

bf_scores = pd.read_csv(bf_scores_path)
fp_scores = pd.read_csv(fp_scores_path)
as_scores = pd.read_csv(as_scores_path)

bf_lrs_path = "output/all_types_independence_filtered/bf_cmc/modules.csv"
fp_lrs_path = "output/all_types_independence_filtered/fp_cmc/modules.csv"
as_lrs_path = "output/all_types_independence_filtered/as_ccf/modules.csv"

bf_lrs = pd.read_csv(bf_lrs_path)
fp_lrs = pd.read_csv(fp_lrs_path)
as_lrs = pd.read_csv(as_lrs_path)

output_folder = Path("output/all_types_independence")
output_folder.mkdir(exist_ok=True)

bf_fp_scores_merged = pd.merge(bf_scores, fp_scores, on=['Weapon1', 'Weapon2'], how='inner')
bf_as_scores_merged = pd.merge(bf_scores, as_scores, on=['Weapon1', 'Weapon2'], how='inner')
fp_as_scores_merged = pd.merge(fp_scores, as_scores, on=['Weapon1', 'Weapon2'], how='inner')

bf_fp_scores_merged['cmc/n bf'] = bf_fp_scores_merged['Matching Cells_x']/bf_fp_scores_merged['Total Cells_x']
bf_fp_scores_merged['cmc/n fp'] = bf_fp_scores_merged['Matching Cells_y']/bf_fp_scores_merged['Total Cells_y']

# BF versus FP
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=bf_fp_scores_merged[bf_fp_scores_merged['Hypothesis_x'] == 1], x="cmc/n bf", y="cmc/n fp", ax=ax1, color="steelblue", s=10)
ax1.set_title("Hypothesis 1")
sns.scatterplot(data=bf_fp_scores_merged[bf_fp_scores_merged['Hypothesis_x'] == 0], x="cmc/n bf", y="cmc/n fp", ax=ax2, color="coral", s=10)
ax2.set_title("Hypothesis 2")
plt.tight_layout()
plt.savefig(f"{output_folder}/{'bf_fp_scores'}", dpi=300)
# plt.show()

bf_as_scores_merged['cmc/n bf'] = bf_fp_scores_merged['Matching Cells_x']/bf_fp_scores_merged['Total Cells_x']

# # BF versus AS
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# sns.scatterplot(data=bf_as_scores_merged[bf_as_scores_merged['Hypothesis_x'] == 1], x="cmc/n bf", y="Score", ax=ax1, color="steelblue", s=10)
# ax1.set_title("Hypothesis 1")
# sns.scatterplot(data=bf_as_scores_merged[bf_as_scores_merged['Hypothesis_x'] == 0], x="cmc/n bf", y="Score", ax=ax2, color="coral", s=10)
# ax2.set_title("Hypothesis 2")
# plt.tight_layout()
# plt.savefig(f"{output_folder}/{'bf_as_scores'}", dpi=300)
# # plt.show()
#
# fp_as_scores_merged['cmc/n fp'] = fp_as_scores_merged['Matching Cells']/fp_as_scores_merged['Total Cells']
#
# # FP versus AS
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# sns.scatterplot(data=fp_as_scores_merged[fp_as_scores_merged['Hypothesis_x'] == 1], x="cmc/n fp", y="Score", ax=ax1, color="steelblue", s=10)
# ax1.set_title("Hypothesis 1")
# sns.scatterplot(data=fp_as_scores_merged[fp_as_scores_merged['Hypothesis_x'] == 0], x="cmc/n fp", y="Score", ax=ax2, color="coral", s=10)
# ax2.set_title("Hypothesis 2")
# plt.tight_layout()
# plt.savefig(f"{output_folder}/{'fp_as_scores'}", dpi=300)
# # plt.show()

bf_fp_lrs_merged = pd.merge(bf_lrs, fp_lrs, on=['source_ids.0', 'source_ids.1'], how='inner')
bf_as_lrs_merged = pd.merge(bf_lrs, as_lrs, on=['source_ids.0', 'source_ids.1'], how='inner')
fp_as_lrs_merged = pd.merge(fp_lrs, as_lrs, on=['source_ids.0', 'source_ids.1'], how='inner')

# BF versus FP
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=bf_fp_lrs_merged[bf_fp_lrs_merged['label_x'] == 1], x="mcmc.0_x", y="mcmc.0_y", ax=ax1, color="steelblue", s=10)
ax1.set_title("Hypothesis 1")
sns.scatterplot(data=bf_fp_lrs_merged[bf_fp_lrs_merged['label_x'] == 0], x="mcmc.0_x", y="mcmc.0_y", ax=ax2, color="coral", s=10)
ax2.set_title("Hypothesis 2")
plt.tight_layout()
ax1.set_xlabel("BF LR score")
ax1.set_ylabel("FP LR score")
ax2.set_xlabel("BF LR score")
ax2.set_ylabel("FP LR score")
plt.savefig(f"{output_folder}/{'bf_fp_lrs_no_bounds'}", dpi=300)
# plt.show()

# # BF versus AS
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# sns.scatterplot(data=bf_as_lrs_merged[bf_as_lrs_merged['label_x'] == 1], x="bounder.0", y="bootstrap.1", ax=ax1, color="steelblue", s=10)
# ax1.set_title("Hypothesis 1")
# sns.scatterplot(data=bf_as_lrs_merged[bf_as_lrs_merged['label_x'] == 0], x="bounder.0", y="bootstrap.1", ax=ax2, color="coral", s=10)
# ax2.set_title("Hypothesis 2")
# plt.tight_layout()
# ax1.set_xlabel("BF LR score")
# ax1.set_ylabel("AS LR score")
# ax2.set_xlabel("BF LR score")
# ax2.set_ylabel("AS LR score")
# plt.savefig(f"{output_folder}/{'bf_as_lrs'}", dpi=300)
# # plt.show()
#
# # FP versus AS
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# sns.scatterplot(data=fp_as_lrs_merged[fp_as_lrs_merged['label_x'] == 1], x="bounder.0", y="bootstrap.1", ax=ax1, color="steelblue", s=10)
# ax1.set_title("Hypothesis 1")
# sns.scatterplot(data=fp_as_lrs_merged[fp_as_lrs_merged['label_x'] == 0], x="bounder.0", y="bootstrap.1", ax=ax2, color="coral", s=10)
# ax2.set_title("Hypothesis 2")
# plt.tight_layout()
# ax1.set_xlabel("FP LR score")
# ax1.set_ylabel("AS LR score")
# ax2.set_xlabel("FP LR score")
# ax2.set_ylabel("AS LR score")
# plt.savefig(f"{output_folder}/{'fp_as_lrs'}", dpi=300)
# # plt.show()

a = 1