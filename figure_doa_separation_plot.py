from __future__ import division

import sys
import getopt
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="ticks")

from tools import polar_error, polar_distance

from experiment import arrays

if __name__ == "__main__":

    # parse arguments
    argv = sys.argv[1:]

    data_file = 'data/20160907-183030_doa_separation.npz'

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

    # Get the microphone array locations
    array_str = 'pyramic'
    R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)
    mic_array = arrays['pyramic_tetrahedron'][:, R_flat_I].copy()

    # This is the output from `figure_doa_experiment.py`
    data = np.load(data_file)

    # extra variables
    algo_names = data['algo_names'].tolist()
    parameters = data['parameters']
    args = data['args'].tolist()
    sim_out = data['out']

    # algorithms to take in the plot
    algos = ['FRI','MUSIC','SRP','CSSM','WAVES','TOPS']

    # build the data table line by line
    columns = ['SNR','angle','phi_r1','phi_r2','success','algo','error']
    table = []
    for i,a in enumerate(args):
        for alg in algos:

            snr = a[0]
            phi = a[1]
            phi_gt = sim_out[i][0]['groundtruth']
            phi_recon = sim_out[i][0][alg]

            # sort the angles
            recon_err, sort_idx = polar_distance(phi_gt, phi_recon)
            phi_gt = phi_gt[sort_idx[:,0]]
            phi_recon = phi_recon[sort_idx[:,1]]

            success = 0
            for p1,p2 in zip(phi_gt, phi_recon):
                if polar_error(p1,p2) < phi / 2.:
                    success += 1

            entry = []
            entry.append(snr)
            entry.append(np.degrees(phi))
            entry.append(np.degrees(polar_error(0.,phi_recon[0])))
            entry.append(np.degrees(polar_error(phi, phi_recon[1])))
            entry.append(success)
            entry.append(alg)

            error = np.mean([np.minimum(polar_error(phi_g, phi_r), phi) for phi_g, phi_r in zip(phi_gt,phi_recon)])
            entry.append(error / phi)

            table.append(entry)
       
    # create a pandas frame
    df = pd.DataFrame(table, columns=columns)

    # Draw box plots
    plt.figure()
    sns.boxplot(x="angle", y="error", hue="algo", data=df[df['SNR'] == 0], palette="PRGn")
    sns.despine(offset=10, trim=True)

    plt.figure()
    sns.boxplot(x="angle", y="phi_r1", hue="algo", data=df[df['SNR'] == 0], palette="PRGn")
    sns.despine(offset=10, trim=True)

    plt.figure()
    sns.boxplot(x="angle", y="phi_r2", hue="algo", data=df[df['SNR'] == 0], palette="PRGn")
    sns.despine(offset=10, trim=True)
    plt.show()
