from __future__ import division

import sys
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="ticks")

from experiment import arrays

# Get the microphone array locations
array_str = 'pyramic'
R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)
mic_array = arrays['pyramic_tetrahedron'][:, R_flat_I].copy()

# This is the output from `figure_doa_experiment.py`
data_file = 'data/20160907-034306_doa_synthetic.npz'
data = np.load(data_file)

# extra variables
algo_names = data['algo_names'].tolist()
parameters = data['parameters']
args = data['args'].tolist()
sim_out = data['out']

# algorithms to take in the plot
algos = ['FRI','MUSIC','SRP','CSSM']

# build the data table line by line
err_header = ['n_sources','SNR','n_bands','Algo','Error']
table = []
for i,a in enumerate(args):
    for alg in sim_out[i][1].keys():

        entry = copy.copy(a)
        entry.append(alg)
        entry.append(np.mean(sim_out[i][1][alg]))

        table.append(entry)
   
# create a pandas frame
df = pd.DataFrame(table, columns=err_header)

# Group by SNR, n_sources, and n_bands before averaging
grp_src = np.logical_and(df['SNR'] == 5, df['n_bands'] == 4)
grp_bands = np.logical_and(df['SNR'] == 5, df['n_sources'] == 2)

# Draw box plots
plt.figure()
sns.boxplot(x="n_sources", y="Error", hue="Algo", data=df[grp_src], palette="PRGn")
sns.despine(offset=10, trim=True)

plt.figure()
sns.boxplot(x="n_bands", y="Error", hue="Algo", data=df[grp_bands], palette="PRGn")
sns.despine(offset=10, trim=True)

plt.show()
