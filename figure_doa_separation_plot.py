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
        print('figure_doa_separation_plot.py -f <data_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('figure_doa_separation_plot.py -f <data_file>')
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
    parameters = data['parameters'][()]
    args = data['args'].tolist()
    sim_out = data['out']

    # algorithms to take in the plot
    algos = ['FRI','MUSIC','SRP','CSSM','WAVES','TOPS']
    algo_lut = {
            'FRI': 'FRIDA', 'MUSIC': 'MUSIC', 'SRP': 'SRP-PHAT', 
            'CSSM':'CSSM', 'WAVES':'WAVES','TOPS':'TOPS'
            }

    # find min angle of separation
    angles = set()
    for a in args:
        angles.add(a[1])
    phi_min = min(angles)
    phi_max = max(angles)

    # build the data table line by line
    print 'Building table'
    columns = ['SNR','Algorithm','angle','err1','err2','erravg','success']
    table = []
    for i,a in enumerate(args):
        for alg in algos:

            snr = a[0]
            phi = a[1]
            look = a[2]
            phi_gt = sim_out[i]['groundtruth']
            phi_recon = sim_out[i][alg]

            # sort the angles
            recon_err, sort_idx = polar_distance(phi_gt, phi_recon)

            thresh = phi / 2.

            if len(phi_recon) == 2:

                phi_gt = phi_gt[sort_idx[:,0]]
                phi_recon = phi_recon[sort_idx[:,1]]

                # compute individual and average error
                err = [polar_error(phi_gt[j],phi_recon[j]) for j in range(2)]
                err_avg = np.mean(err)

                # number of sources resolved
                success = 0
                for p1,p2 in zip(phi_gt, phi_recon):
                    if polar_error(p1,p2) < thresh:
                        success += 1

            elif len(phi_recon) == 1:

                phi_gt = phi_gt[sort_idx[0]]
                phi_recon = phi_recon
                err = [np.nan, np.nan]
                err[sort_idx[0]] = polar_error(phi_gt, phi_recon)
                    
                err_avg = err[sort_idx[1]]

                if err < phi/2:
                    success = 1
                else:
                    success = 0

            entry = []
            entry.append(snr)
            entry.append(algo_lut[alg])
            entry.append(np.round(np.degrees(phi), decimals=1))
            entry.append(np.degrees(err[0]))
            entry.append(np.degrees(err[1]))
            entry.append(np.degrees(err_avg))
            entry.append(success)

            table.append(entry)
       
    # create a pandas frame
    print 'Creating dataframe'
    df = pd.DataFrame(table, columns=columns)

    print 'Plot...'

    sns.set(style='whitegrid',context='paper', font_scale=1.2)
    pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    g = sns.factorplot(x='angle',y='success',hue='Algorithm',
            data=df[['angle','success','Algorithm']],
            hue_order=['FRIDA','MUSIC','SRP-PHAT','CSSM','TOPS','WAVES'],
            aspect=1, palette=pal,
            legend=False,
            markers=['^','o','x','s','d','v'],
            ci=None, figsize=(3.15,3.15))

    plt.legend(loc='lower right',title='Algorithm')
    g.set_ylabels("Algorithm")

    plt.xlabel('Separation angle [$^\circ$]')
    plt.ylabel('\# sources resolved')

    plt.ylim([0.45,2.1])
    plt.yticks(np.arange(0.5,2.5,0.5))

    sns.despine(offset=10, trim=False, left=True, bottom=True)

    plt.savefig('figures/experiment_minimum_separation.pdf')

    plt.show()
