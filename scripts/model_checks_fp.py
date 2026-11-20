import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# specify paths and filenames
path = 'C:\\Code\\Github\\lr_module_scratch\\data\\2026-Q4\\all_glocks_mcmc_traces_fp\\'

filename_scores = 'firing_pin_impression.csv'
pathfile_scores = path + filename_scores
filename_alphas = '01-distribution-h1_betabinomial_alpha.csv'
pathfile_alphas = path + filename_alphas
filename_betas = '01-distribution-h1_betabinomial_beta.csv'
pathfile_betas = path + filename_betas
filename_ps = '01-distribution-h2_binomial_p.csv'
pathfile_ps = path + filename_ps

scores = pd.read_csv(pathfile_scores)
total_cells_h1 = scores[scores['Hypothesis']==1]['Total Cells']
total_cells_h1 = total_cells_h1.to_numpy()
total_cells_h2 = scores[scores['Hypothesis']==0]['Total Cells']
total_cells_h2 = total_cells_h2.to_numpy()

sample_size = 10**5
alphas = pd.read_csv(pathfile_alphas, header=None)
alphas =alphas.to_numpy()
betas = pd.read_csv(pathfile_betas, header=None)
betas = betas.to_numpy()
sample_h2_probs = pd.read_csv(pathfile_ps, header=None)
sample_h2_probs = sample_h2_probs.to_numpy()
rng = np.random.default_rng()
sample_h2 = rng.binomial(total_cells_h2, sample_h2_probs)
sample_h2 = sample_h2/total_cells_h2
sample_h2 = sample_h2.ravel()
sample_h2 = rng.choice(sample_h2, sample_size, replace=False)

sample_h1_probs = rng.beta(alphas, betas)
sample_h1 = rng.binomial(total_cells_h1, sample_h1_probs)
sample_h1 = sample_h1/total_cells_h1
sample_h1 = sample_h1.ravel()
sample_h1 = rng.choice(sample_h1, sample_size, replace=False)

# fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8, 4))
#
# sns.kdeplot(data=sample_h1, color="crimson", label="Dataset H1", ax=ax[0,0])
# df_nn_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Nickel') & (scores['Material2']=='Nickel')]
# df_nn_h1['pct'] = df_nn_h1['Matching Cells']/df_nn_h1['Total Cells']
# sns.histplot(data=df_nn_h1, x="pct", bins=20, kde=False, stat="density", ax=ax[0,0])
# ax[0,0].set_xlabel("Value")
# ax[0,0].set_ylabel("Density")
# ax[0,0].legend()
# ax[0,0].set_title('Model H1 versus Nickel/Nickel')
# plt.tight_layout()
#
# sns.kdeplot(data=sample_h2, color="crimson", label="Dataset H2", ax=ax[0,1], bw_adjust=0.01)
# df_nn_h2 = scores[(scores['Hypothesis']==0) & (scores['Material1']=='Nickel') & (scores['Material2']=='Nickel')]
# df_nn_h2['pct'] = df_nn_h2['Matching Cells']/df_nn_h2['Total Cells']
# sns.histplot(data=df_nn_h2, x="pct", bins=200, kde=False, stat="density", ax=ax[0,1])
# ax[0,1].set_xlabel("Value")
# ax[0,1].set_ylabel("Density")
# ax[0,1].legend()
# ax[0,1].set_title('Model H2 versus Nickel/Nickel')
# plt.tight_layout()
#
# sns.kdeplot(data=sample_h1, color="crimson", label="Dataset H1", ax=ax[1,0])
# df_bb_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Brass') & (scores['Material2']=='Brass')]
# df_bb_h1['pct'] = df_bb_h1['Matching Cells']/df_bb_h1['Total Cells']
# sns.histplot(data=df_bb_h1, x="pct", bins=20, kde=False, stat="density", ax=ax[1,0])
# ax[1,0].set_xlabel("Value")
# ax[1,0].set_ylabel("Density")
# ax[1,0].legend()
# ax[1,0].set_title('Model H1 versus Brass/Brass')
# plt.tight_layout()
#
# sns.kdeplot(data=sample_h2, color="crimson", label="Dataset H2", ax=ax[1,1], bw_adjust=0.01)
# df_bb_h2 = scores[(scores['Hypothesis']==0) & (scores['Material1']=='Brass') & (scores['Material2']=='Brass')]
# df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
# sns.histplot(data=df_bb_h2, x="pct", bins=200, kde=False, stat="density", ax=ax[1,1])
# ax[1,1].set_xlabel("Value")
# ax[1,1].set_ylabel("Density")
# ax[1,1].legend()
# ax[1,1].set_title('Model H2 versus Brass/Brass')
# plt.tight_layout()
#
# sns.kdeplot(data=sample_h1, color="crimson", label="Dataset H1", ax=ax[2,0])
# df_bb_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Brass') & (scores['Material2']=='Brass')|
#     ((scores['Hypothesis']==1) & (scores['Material2']=='Nickle') & (scores['Material1']=='Brass'))]
# df_bb_h1['pct'] = df_bb_h1['Matching Cells']/df_bb_h1['Total Cells']
# sns.histplot(data=df_bb_h1, x="pct", bins=20, kde=False, stat="density", ax=ax[2,0])
# ax[2,0].set_xlabel("Value")
# ax[2,0].set_ylabel("Density")
# ax[2,0].legend()
# ax[2,0].set_title('Model H1 versus Nickel/Brass')
# plt.tight_layout()
#
# sns.kdeplot(data=sample_h2, color="crimson", label="Dataset H2", ax=ax[2,1], bw_adjust=0.01)
# df_bb_h2 = scores[((scores['Hypothesis']==0) & (scores['Material1']=='Nickle') & (scores['Material2']=='Brass'))|
#     ((scores['Hypothesis']==0) & (scores['Material2']=='Nickle') & (scores['Material1']=='Brass'))]
# df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
# sns.histplot(data=df_bb_h2, x="pct", bins=200, kde=False, stat="density", ax=ax[2,1])
# ax[2,1].set_xlabel("Value")
# ax[2,1].set_ylabel("Density")
# ax[2,1].legend()
# ax[2,1].set_title('Model H2 versus Nickel/Brass')
# plt.tight_layout()
#
# sns.kdeplot(data=sample_h1, color="crimson", label="Dataset H1", ax=ax[3,0])
# df_bb_h1 = scores[(scores['Hypothesis']==1)]
# df_bb_h1['pct'] = df_bb_h1['Matching Cells']/df_bb_h1['Total Cells']
# sns.histplot(data=df_bb_h1, x="pct", bins=20, kde=False, stat="density", ax=ax[3,0])
# ax[3,0].set_xlabel("Value")
# ax[3,0].set_ylabel("Density")
# ax[3,0].legend()
# ax[3,0].set_title('Model H1 versus alles')
# plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 4))

# sns.kdeplot(data=sample_h2, color="crimson", label="Dataset H2", ax=ax[3,1], bw_adjust=0.01)
# df_bb_h2 = scores[(scores['Hypothesis']==0)]
# df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
# sns.histplot(data=df_bb_h2, x="pct", bins=200, kde=False, stat="density", ax=ax[3,1])
# ax[3,1].set_xlabel("Value")
# ax[3,1].set_ylabel("Density")
# ax[3,1].legend()
# ax[3,1].set_title('Model H2 versus alles')
# plt.tight_layout()sns.kdeplot(data=sample_h2, color="crimson", label="Dataset H2", ax=ax[3,1], bw_adjust=0.01)

df_bb_h2 = scores[(scores['Hypothesis']==0)]
df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
sample_h2 = pd.DataFrame(sample_h2, columns=['pct'])
sns.histplot(data=sample_h2, x="pct", binwidth=0.002, kde=False, stat="density", ax=ax, color='red')
sns.histplot(data=df_bb_h2, x="pct", binwidth=0.002, kde=False, stat="density", ax=ax, color='green')
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
ax.set_title('Model H2 versus alles')
plt.tight_layout()

plt.show()

a = 1