from __future__ import division

import sys
import numpy as np
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tools import polar_distance
from experiment import arrays

if __name__ == "__main__":
    # parse arguments
    argv = sys.argv[1:]

    # This is the output from `figure_doa_experiment.py`
    data_file = 'data/20160906-205811_doa_experiment.npz'
    #data_file = 'data/20160905-212909_doa_experiment.npz'
    #data_file = 'data/20160906-091115_doa_experiment.npz'

    try:
        opts, args = getopt.getopt(argv, "hf:", ["file=",])
    except getopt.GetoptError:
        print('test_doa_recorded.py -f <data_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test_doa_recorded.py -a <algo> -f <file> -b <n_bands>')
            sys.exit()
        elif opt in ("-f", "--file"):
            data_file = arg

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

    data = np.load(data_file)

    # build some container arrays
    algo_names = data['algo_names'].tolist()

    # Now loop and process the results
    columns = ['sources','SNR','Algo','Error']
    table = []
    for pt in data['out']:

        SNR = pt[0]
        speakers = [s.replace("'","") for s in pt[1]]
        K = len(speakers)

        # Get groundtruth for speaker
        phi_gt = np.array([twitters.doa(array_str, s) for s in speakers])

        for alg in pt[2].keys():
            phi_recon = pt[2][alg]
            recon_err, sort_idx = polar_distance(phi_recon, phi_gt)
            table.append([K, SNR, alg, np.degrees(recon_err)])

    # Create pandas data frame
    df = pd.DataFrame(table, columns=columns)

    sns.boxplot(x="sources", y="Error", hue="Algo", hue_order=['FRI','MUSIC','SRP'], data=df, palette="PRGn")
    sns.despine(offset=10, trim=True)
    plt.yticks(np.arange(0,6))
    plt.ylim([0.0, 3.])

    plt.show()

