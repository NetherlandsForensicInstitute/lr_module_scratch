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

alphas = pd.read_csv(pathfile_alphas, header=None)
alphas =alphas.to_numpy()
betas = pd.read_csv(pathfile_betas, header=None)
betas = betas.to_numpy()
sample_h2 = pd.read_csv(pathfile_ps, header=None)
sample_h2 = sample_h2.to_numpy()
scores = pd.read_csv(pathfile_scores)

rng = np.random.default_rng()
sample_h1 = rng.beta(alphas, betas)

fig, ax = plt.subplots(figsize=(8, 4))

sns.kdeplot(data=sample_h1.ravel(), color="crimson", label="Dataset H1", ax=ax)
df_nn_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Nickel') & (scores['Material2']=='Nickel')]
df_nn_h1['pct'] = df_nn_h1['Matching Cells']/df_nn_h1['Total Cells']
sns.histplot(data=df_nn_h1, x="pct", bins=20, kde=False, stat="density")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=sample_h2.ravel(), color="crimson", label="Dataset H2", ax=ax)
df_nn_h2 = scores[(scores['Hypothesis']==0) & (scores['Material1']=='Nickel') & (scores['Material2']=='Nickel')]
df_nn_h2['pct'] = df_nn_h2['Matching Cells']/df_nn_h2['Total Cells']
sns.histplot(data=df_nn_h2, x="pct", bins=200, kde=False, stat="density")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=sample_h1.ravel(), color="crimson", label="Dataset H1", ax=ax)
df_bb_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Brass') & (scores['Material2']=='Brass')]
df_bb_h1['pct'] = df_bb_h1['Matching Cells']/df_bb_h1['Total Cells']
sns.histplot(data=df_bb_h1, x="pct", bins=20, kde=False, stat="density")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=sample_h2.ravel(), color="crimson", label="Dataset H2", ax=ax)
df_bb_h2 = scores[(scores['Hypothesis']==0) & (scores['Material1']=='Brass') & (scores['Material2']=='Brass')]
df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
sns.histplot(data=df_bb_h2, x="pct", bins=200, kde=False, stat="density")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()

a = 1