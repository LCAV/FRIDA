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
data_file = 'data/20160907-183030_doa_separation.npz'
data = np.load(data_file)

# extra variables
algo_names = data['algo_names'].tolist()
parameters = data['parameters']
args = data['args'].tolist()
sim_out = data['out']

# algorithms to take in the plot
algos = ['FRI','MUSIC','SRP','CSSM','WAVES','TOPS']

# build the data table line by line
columns = ['SNR','angle','algo','error']
table = []
for i,a in enumerate(args):
    for alg in algos:

        entry = copy.copy(a)
        entry.append(alg)
        entry.append(np.mean(sim_out[i][1][alg]))

        table.append(entry)
   
# create a pandas frame
df = pd.DataFrame(table, columns=columns)

# Draw box plots
plt.figure()
sns.boxplot(x="angle", y="error", hue="algo", data=df[df['SNR'] == 5], palette="PRGn")
sns.despine(offset=10, trim=True)

plt.show()
