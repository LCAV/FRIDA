from __future__ import division

import sys
import numpy as np
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tools import polar_distance, polar_error
from experiment import arrays

if __name__ == "__main__":
    # parse arguments
    argv = sys.argv[1:]

    # This is the output from `figure_doa_experiment.py`
    data_file = 'data/20160909-203344_doa_experiment.npz'

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
    exp_folder = './recordings/20160908/'
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
    columns = ['sources','SNR','Algorithm','Error']
    table = []
    close_sources = []
    for pt in data['out']:

        SNR = pt[0]
        speakers = [s.replace("'","") for s in pt[1]]
        K = len(speakers)

        # Get groundtruth for speaker
        phi_gt = np.array([twitters.doa(array_str, s) for s in speakers])[:,0]

        for alg in pt[2].keys():
            phi_recon = pt[2][alg]
            recon_err, sort_idx = polar_distance(phi_gt, phi_recon)
            table.append([K, SNR, alg, np.degrees(recon_err)])

            # we single out the reconstruction of the two closely located sources
            if '7' in speakers and '16' in speakers:
                # by construction '7' is always first and '16' second
                success = 0
                for p1,p2 in zip(phi_gt[:2], phi_recon[sort_idx[:2,1]]):
                    if polar_error(p1,p2) < polar_error(phi_gt[0], phi_gt[1]) / 2:
                        success += 1
                close_sources.append([alg, 
                    success, 
                    phi_recon[sort_idx[0,1]], 
                    phi_recon[sort_idx[1,1]],
                    ])

    # Create pandas data frame
    df = pd.DataFrame(table, columns=columns)

    # Compute statistics for the reconstructed angles
    df_close_sources = pd.DataFrame(close_sources, columns=['Algorithm','Success','7','16'])
    mu = {'7':{},'16':{}}
    std = {'7':{},'16':{}}
    for alg in ['FRI','MUSIC','SRP']:
        phi_r = df_close_sources[['Algorithm','7','16']][df_close_sources['Algorithm'] == alg]
        for spkr in ['7','16']:
            mu[spkr][alg] = np.angle(np.mean(np.exp(1j*phi_r[spkr])))
            std[spkr][alg] = np.mean([polar_error(p, mu[spkr][alg]) for p in phi_r[spkr]])

    for spkr in ['7','16']:
        for alg in ['FRI','MUSIC','SRP']:
            print spkr,alg,'mu=',np.degrees(mu[spkr][alg]),'std=',np.degrees(std[spkr][alg])
    for spkr in ['7','16']:
        for alg in ['FRI','MUSIC','SRP']:
            print np.degrees(mu[spkr][alg]),np.degrees(std[spkr][alg]),
        print ''



    # Create the super plot comparing all algorithms
    algo_plot = ['FRI','MUSIC','SRP', 'CSSM', 'TOPS', 'WAVES']

    plt.figure(figsize=(6,4))

    sns.set(style='whitegrid',context='paper', font_scale=1.2)
    pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)
    sns.set_palette(pal)

    plt.figure(figsize=(4.7,3.15), dpi=90)

    sns.boxplot(x="sources", y="Error", hue="Algorithm", 
            hue_order=algo_plot, data=df, 
            palette=pal,
            fliersize=0.)
            
            #palette="PRGn")
    sns.despine(offset=10, trim=True, left=True)

    plt.xlabel("Number of sources")
    plt.ylabel("Error $[^\circ]$")
    plt.yticks(np.arange(0,80))
    plt.ylim([0.0, 3.3])
    plt.tight_layout(pad=0.1)

    plt.savefig('figures/experiment_error_box.pdf')

    plt.show()

