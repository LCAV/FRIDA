from __future__ import division

import sys
import copy
import numpy as np
import pandas as pd
import getopt

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="ticks")

from experiment import arrays
from tools import polar_distance
if __name__ == "__main__":

    argv = sys.argv[1:]
    files = '20160911-035215_doa_synthetic.npz'

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
            files = arg


    # Get the microphone array locations
    array_str = 'pyramic'
    R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)
    mic_array = arrays['pyramic_tetrahedron'][:, R_flat_I].copy()

    # algorithms to take in the plot
    algo_names = ['FRI','MUSIC','SRP','CSSM','TOPS','WAVES']
    algo_lut = {
            'FRI': 'FRIDA', 'MUSIC': 'MUSIC', 'SRP': 'SRP-PHAT', 
            'CSSM':'CSSM', 'WAVES':'WAVES','TOPS':'TOPS'
            }

    # build the data table line by line
    print 'Building table...'
    err_header = ['SNR','Algorithm','Error','Loop index']
    table = []

    # For easy plotting in seaborn, seems we need a loop count
    loop_index = {}
    Sources = [1,2,3]
    SNRs = np.arange(-35,21)
    for s in SNRs:
        loop_index[s] = {}
        for src in Sources:
            loop_index[s][src] = {}
            for alg in algo_names:
                loop_index[s][src][alg] = 0

    # This is the output from `figure_doa_experiment.py`
    data_files = files.split(',')
    for data_file in data_files:

        data = np.load(data_file)

        # extra variables
        algo_names = data['algo_names'].tolist()
        parameters = data['parameters']
        args = data['args'].tolist()
        sim_out = data['out']

        for i,a in enumerate(args):
            K = int(a[0])

            # only retain values for 1 source
            if K != 1:
                continue

            snr = int(a[1])
            phi_gt = sim_out[i][0]['groundtruth']
            for alg in algo_names:


                recon_err, sort_idx = polar_distance(phi_gt, sim_out[i][0][alg])

                entry = [snr]
                entry.append(algo_lut[alg])
                entry.append(np.degrees(recon_err))
                entry.append(loop_index[snr][K][alg])
                table.append(entry)

                loop_index[snr][K][alg] += 1

    # create a pandas frame
    print 'Making PANDAS frame...'
    df = pd.DataFrame(table, columns=err_header)

    sns.set(style='whitegrid')
    sns.plotting_context(context='poster', font_scale=2.)
    pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    # Draw the figure
    print 'Plotting...'

    '''
    SNR_order = np.sort(np.unique(df['SNR']))

    sns.factorplot(data=df[df['Sources'] == 1], x='SNR',y='Error',hue='Algorithm', col='Sources',
            hue_order=algo_names, order=SNR_order,
            palette=pal, aspect=1.5, markers=['o','v','^','s','d','+'])

    sns.despine(offset=10, trim=True)

    sns.factorplot(data=df, x='SNR',y='Error',hue='Algorithm', col='Sources', kind='box',
            palette=pal, aspect=1.5)
    '''

    #sns.tsplot(time="SNR", value="Error", condition="Algorithm", unit="Loop index", data=df, color=pal)

    sns.set(style='whitegrid',context='paper', font_scale=1.2)
    pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)
    sns.set_palette(pal)

    plt.figure(figsize=(3.15,3.15))

    algo_order = ['FRIDA','MUSIC','SRP-PHAT','CSSM','TOPS','WAVES']
    markers=['^','o','*','s','d','v']
    perf = pd.pivot_table(df, values='Error', index=['SNR'], columns=['Algorithm'], aggfunc=np.mean)

    for alg,mkr in zip(algo_order, markers):
        plt.plot(perf.index, perf[alg], marker=mkr)
    plt.legend(algo_order, title='Algorithm', frameon=False)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Average Error [$^\circ$]')
    plt.xlim([-35,15])
    plt.ylim([-0.5, 95])
    plt.xticks([-30, -20, -10, 0, 10])
    plt.yticks([0, 20, 40, 60, 80])

    ax = plt.gca()
    ax.xaxis.grid(False)

    sns.despine(offset=10, trim=False, left=True, bottom=True)

    plt.tight_layout(pad=0.1)

    plt.savefig('figures/experiment_snr_synthetic.pdf')

    plt.show()
