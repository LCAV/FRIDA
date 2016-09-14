from __future__ import division

import sys
import copy
import numpy as np
import pandas as pd
import getopt
import os

from tools import polar_distance

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    argv = sys.argv[1:]
    data_files = '20160911-035215_doa_synthetic.npz'
    data_files = [
             'data/20160911-161112_doa_synthetic.npz',
             'data/20160911-225325_doa_synthetic.npz',
             'data/20160911-175127_doa_synthetic.npz',
             'data/20160911-035215_doa_synthetic.npz',
             'data/20160911-192530_doa_synthetic.npz',
             ]

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
            data_files = arg.split(',')

    # algorithms to take in the plot
    algo_names = ['FRI','MUSIC','SRP','CSSM','TOPS','WAVES']
    algo_lut = {
            'FRI': 'FRIDA', 'MUSIC': 'MUSIC', 'SRP': 'SRP-PHAT', 
            'CSSM':'CSSM', 'WAVES':'WAVES','TOPS':'TOPS'
            }

    # check if a pickle file exists for these files
    pickle_file = os.path.splitext(data_files[0])[0] + '_{}'.format(len(data_files)) + '.pickle'

    if os.path.isfile(pickle_file):
        # read the pickle file
        perf = pd.read_pickle(pickle_file)

    else:
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

        #if os.

        # This is the output from `figure_doa_experiment.py`
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

        # turns out all we need is the follow pivoted table
        perf = pd.pivot_table(df, values='Error', index=['SNR'], columns=['Algorithm'], aggfunc=np.mean)

        perf.to_pickle(pickle_file)

    sns.set(style='whitegrid')
    sns.plotting_context(context='poster', font_scale=2.)
    pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    # Draw the figure
    print 'Plotting...'

    #sns.tsplot(time="SNR", value="Error", condition="Algorithm", unit="Loop index", data=df, color=pal)

    sns.set(style='whitegrid', context='paper', font_scale=1.2,
            rc={
                'figure.figsize':(3.5,3.15), 
                'lines.linewidth':2.,
                'font.family': 'sans-serif',
                'font.sans-serif': [u'Helvetica'],
                'text.usetex': False,
                })
    #pal = sns.cubehelix_palette(6, start=0.5, rot=-0.75, dark=0.25, light=.75, reverse=True, hue=0.9)
    pal = sns.cubehelix_palette(6, start=0.5, rot=-0.5,dark=0.3, light=.75, reverse=True, hue=1.)
    sns.set_palette(pal)
    #sns.set_palette('viridis')

    plt.figure()

    algo_order = ['FRIDA','MUSIC','SRP-PHAT','CSSM','TOPS','WAVES']
    markers=['^','o','*','s','d','v']

    for alg,mkr in zip(algo_order, markers):
        plt.plot(perf.index, perf[alg], marker=mkr, clip_on=False)

    ax = plt.gca()

    # remove the x-grid
    ax.xaxis.grid(False)

    ax.text(-45,87.5, 'A', fontsize=27, fontweight='bold')

    # nice legend box
    leg = plt.legend(algo_order, title='Algorithm', frameon=True, framealpha=0.6)
    leg.get_frame().set_linewidth(0.0)

    # set all the labels
    plt.xlabel('SNR [dB]')
    plt.ylabel('Average Error [$^\circ$]')
    plt.xlim([-35,15])
    plt.ylim([-0.5, 95])
    plt.xticks([-30, -20, -10, 0, 10])
    plt.yticks([0, 20, 40, 60, 80])

    sns.despine(offset=10, trim=False, left=True, bottom=True)

    plt.tight_layout(pad=0.5)

    plt.savefig('figures/experiment_snr_synthetic.pdf')

    plt.show()
