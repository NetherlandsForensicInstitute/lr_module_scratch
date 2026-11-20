import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# specify paths and filenames
path = 'C:\\Code\\Github\\lr_module_scratch\\output\\all_glocks_mcmc\\cmc\\mcmc_output\\'

filename_scores = 'breech_face_impression.csv'
pathfile_scores = path + filename_scores
filename_alphas = 'batch_01-h1_betabinomial_alpha-values.csv'
pathfile_alphas = path + filename_alphas
filename_betas = 'batch_01-h1_betabinomial_beta-values.csv'
pathfile_betas = path + filename_betas
filename_ps = 'batch_01-h2_binomial_p-values.csv'
pathfile_ps = path + filename_ps

scores = pd.read_csv(pathfile_scores)
total_cells_h1 = scores[scores['Hypothesis']==1]['Total Cells']
total_cells_h1 = total_cells_h1.to_numpy()
total_cells_h2 = scores[scores['Hypothesis']==0]['Total Cells']
total_cells_h2 = total_cells_h2.to_numpy()

sample_size = 10**5
alphas = pd.read_csv(pathfile_alphas, header=None)
alphas = alphas.to_numpy()
betas = pd.read_csv(pathfile_betas, header=None)
betas = betas.to_numpy()

rng = np.random.default_rng()

sample_h1_probs = rng.beta(alphas, betas)
sample_h1 = rng.binomial(total_cells_h1, sample_h1_probs)
sample_h1 = sample_h1/total_cells_h1
sample_h1 = sample_h1.ravel()
sample_h1 = rng.choice(sample_h1, sample_size, replace=False)

sample_h2_probs = pd.read_csv(pathfile_ps, header=None)
sample_h2_probs = sample_h2_probs.to_numpy()
rng = np.random.default_rng()
sample_h2 = rng.binomial(total_cells_h2, sample_h2_probs)
sample_h2 = sample_h2/total_cells_h2
sample_h2 = sample_h2.ravel()
sample_h2 = rng.choice(sample_h2, sample_size, replace=False)

sample_h1 = pd.DataFrame(sample_h1, columns=['pct'])
sample_h2 = pd.DataFrame(sample_h2, columns=['pct'])

bin_width_h1 = 0.01
bin_width_h2 = 0.002

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4))

df_nn_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Nickel') & (scores['Material2']=='Nickel')]
df_nn_h1['pct'] = df_nn_h1['Matching Cells']/df_nn_h1['Total Cells']
sns.histplot(data=sample_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='red', ax=ax[0,0])
sns.histplot(data=df_nn_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='green', ax=ax[0,0])
ax[0,0].set_xlabel("Value")
ax[0,0].set_ylabel("Density")
ax[0,0].set_title('BF: Model H1 versus Nickel/Nickel')
plt.tight_layout()

df_bb_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Brass') & (scores['Material2']=='Brass')]
df_bb_h1['pct'] = df_bb_h1['Matching Cells']/df_bb_h1['Total Cells']
sns.histplot(data=sample_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='red', ax=ax[0,1])
sns.histplot(data=df_bb_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='green', ax=ax[0,1])
ax[0,1].set_xlabel("Value")
ax[0,1].set_ylabel("Density")
ax[0,1].set_title('BF: Model H1 versus Brass/Brass')
plt.tight_layout()

df_bb_h1 = scores[(scores['Hypothesis']==1) & (scores['Material1']=='Nickel') & (scores['Material2']=='Brass')|
    ((scores['Hypothesis']==1) & (scores['Material2']=='Nickel') & (scores['Material1']=='Brass'))]
df_bb_h1['pct'] = df_bb_h1['Matching Cells']/df_bb_h1['Total Cells']
sns.histplot(data=sample_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='red', ax=ax[1,0])
sns.histplot(data=df_bb_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='green', ax=ax[1,0])
ax[1,0].set_xlabel("Value")
ax[1,0].set_ylabel("Density")
ax[1,0].set_title('BF: Model H1 versus Nickel/Brass')
plt.tight_layout()

df_bb_h1 = scores[(scores['Hypothesis']==1)]
df_bb_h1['pct'] = df_bb_h1['Matching Cells']/df_bb_h1['Total Cells']
sns.histplot(data=sample_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='red', ax=ax[1,1])
sns.histplot(data=df_bb_h1, x="pct", binwidth=bin_width_h1, kde=False, stat="density", color='green', ax=ax[1,1])
ax[1,1].set_xlabel("Value")
ax[1,1].set_ylabel("Density")
ax[1,1].set_title('BF: Model H1 versus alles')
plt.tight_layout()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4))

df_nn_h2 = scores[(scores['Hypothesis']==0) & (scores['Material1']=='Nickel') & (scores['Material2']=='Nickel')]
df_nn_h2['pct'] = df_nn_h2['Matching Cells']/df_nn_h2['Total Cells']
sns.histplot(data=sample_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='red', ax=ax[0,0])
sns.histplot(data=df_nn_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='green', ax=ax[0,0])
ax[0,0].set_xlabel("Value")
ax[0,0].set_ylabel("Density")
ax[0,0].set_title('BF: Model H2 versus Nickel/Nickel')
plt.tight_layout()

df_bb_h2 = scores[(scores['Hypothesis']==0) & (scores['Material1']=='Brass') & (scores['Material2']=='Brass')]
df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
sns.histplot(data=sample_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='red', ax=ax[0,1])
sns.histplot(data=df_bb_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='green', ax=ax[0,1])
ax[0,1].set_xlabel("Value")
ax[0,1].set_ylabel("Density")
ax[0,1].set_title('BF: Model H2 versus Brass/Brass')
plt.tight_layout()

df_bb_h2 = scores[((scores['Hypothesis']==0) & (scores['Material1']=='Nickel') & (scores['Material2']=='Brass'))|
    ((scores['Hypothesis']==0) & (scores['Material2']=='Nickel') & (scores['Material1']=='Brass'))]
df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
sns.histplot(data=sample_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='red', ax=ax[1,0])
sns.histplot(data=df_bb_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='green', ax=ax[1,0])
ax[1,0].set_xlabel("Value")
ax[1,0].set_ylabel("Density")
ax[1,0].set_title('BF: Model H2 versus Nickel/Brass')
plt.tight_layout()

df_bb_h2 = scores[(scores['Hypothesis']==0)]
df_bb_h2['pct'] = df_bb_h2['Matching Cells']/df_bb_h2['Total Cells']
sns.histplot(data=sample_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='red', ax=ax[1,1])
sns.histplot(data=df_bb_h2, x="pct", binwidth=bin_width_h2, kde=False, stat="density", color='green', ax=ax[1,1])
ax[1,1].set_xlabel("Value")
ax[1,1].set_ylabel("Density")
ax[1,1].set_title('BF: Model H2 versus alles')
plt.tight_layout()

plt.show()

a = 1