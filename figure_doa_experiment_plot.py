from __future__ import division

import sys
import numpy as np

from tools import polar_distance
from experiment import arrays

# Get the speakers and microphones grounndtruth locations
exp_folder = './recordings/20160831/'
sys.path.append(exp_folder)
from edm_to_positions import twitters

# Get the microphone array locations
array_str = 'pyramic'
twitters.center(array_str)
R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)
mic_array = arrays['pyramic_tetrahedron'][:, R_flat_I].copy()
mic_array += twitters[[array_str]]

# set the reference point to center of pyramic array
v = {array_str: np.mean(mic_array, axis=1)}
twitters.correct(v)

# This is the output from `figure_doa_experiment.py`
data_file = 'data/20160906-205811_doa_experiment.npz'
#data_file = 'data/20160905-212909_doa_experiment.npz'
#data_file = 'data/20160906-091115_doa_experiment.npz'
data = np.load(data_file)

# build some container arrays
SNR = []
algo_names = data['algo_names'].tolist()
errors = [dict(zip(algo_names, [[] for alg in algo_names])) for i in range(3)]
rmse = [dict(zip(algo_names, [0]*len(algo_names))) for i in range(3)]

# Now loop and process the results
for pt in data['out']:

  SNR.append(pt[0])

  speakers = [s.replace("'","") for s in pt[1]]
  K = len(speakers)

  phi_gt = np.array([twitters.doa(array_str, s) for s in speakers])

  for alg in pt[2].keys():

    phi_recon = pt[2][alg]
    recon_err, sort_idx = polar_distance(phi_recon, phi_gt)

    errors[K-1][alg].append(recon_err * len(speakers))


for K in range(3):
  for alg in algo_names:
    errors[K][alg] = np.array(errors[K][alg])
    rmse[K][alg] = np.degrees(errors[K][alg].mean())

