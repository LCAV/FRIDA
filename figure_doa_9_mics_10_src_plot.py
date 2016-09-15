from __future__ import division

import sys
import copy
import numpy as np
import pandas as pd
import getopt

import matplotlib.pyplot as plt

import seaborn as sns

from experiment import arrays
from tools import polar_distance

if __name__ == "__main__":

    argv = sys.argv[1:]
    files = 'data/20160913-011415_doa_9_mics_10_src.npz'

    try:
        opts, args = getopt.getopt(argv, "hf:", ["file=",])
    except getopt.GetoptError:
        print('figure_doa_9_mics_10_src.py -f <data_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('figure_doa_9_mics_10_src.py -f <data_file>')
            sys.exit()
        elif opt in ("-f", "--file"):
            files = arg

    data = np.load(files)

    phi_ref = data['phi_ks']
    phi_recon = data['phi_recon']
    phi_plt = data['phi_grid']
    dirty_img = data['dirty_img']

    # call seaborn and set the style
    sns.set(style='whitegrid',context='paper',font_scale=1.2,
            rc={
                'figure.figsize':(3.5,3.15), 
                'lines.linewidth':0.75,
                'font.family': 'sans-serif',
                'font.sans-serif': [u'Helvetica'],
                'text.usetex': False,
                })

    # plot
    fig = plt.figure(figsize=(3.15, 3.15), dpi=90)
    ax = fig.add_subplot(111, projection='polar')
    base = 1.
    height = 10.

    #blue = [0, 0.447, 0.741]
    #red = [0.850, 0.325, 0.098]

    pal = sns.cubehelix_palette(6, start=0.5, rot=-0.5,dark=0.3, light=.75, reverse=True, hue=1.)
    #col_recon = pal[0]
    col_gt = pal[3]
    col_spectrum = pal[5]

    pal = sns.color_palette("RdBu_r", 7)
    pal = sns.color_palette("coolwarm", 7)
    col_recon = pal[6]
    #col_gt = pal[1]
    #col_spectrum = pal[2]

    sns.set_palette(pal)

    # We are not interested in amplitude for this plot
    alpha_ref = 2./3 * np.ones(phi_ref.shape)
    alpha_recon = 2./3 * np.ones(phi_recon.shape)

    # match detected with truth
    recon_err, sort_idx = polar_distance(phi_recon, phi_ref)
    phi_recon = phi_recon[sort_idx[:, 0]]
    phi_ref = phi_ref[sort_idx[:, 1]]

    K_est = phi_recon.size
    K = len(phi_ref)

    if phi_ref.shape[0] < 10:
        raise ValueError('WE NEED 10 SOURCES!')

    # plot the 'dirty' image
    dirty_img = np.abs(dirty_img)
    min_val = dirty_img.min()
    max_val = dirty_img.max()
    dirty_img = (dirty_img - min_val) / (max_val - min_val)

    # we need to make a complete loop, copy first value to last
    c_phi_plt = np.r_[phi_plt, phi_plt[0]]
    c_dirty_img = np.r_[dirty_img, dirty_img[0]]
    ax.plot(c_phi_plt, base + 0.95*height*c_dirty_img, linewidth=1, 
        alpha=0.7,linestyle='-', color=col_spectrum, 
        label='spatial spectrum', zorder=0)


    # stem for original doa
    for k in range(K):
        ax.plot([phi_ref[k], phi_ref[k]], [base, base + 
            height*alpha_ref[k]], linewidth=0.5, linestyle='-', 
            color=col_gt, alpha=1., zorder=1)

    # markers for original doa
    ax.scatter(phi_ref, base + height*alpha_ref, 
            c=np.tile(col_gt, (K, 1)), 
            s=70, alpha=1.00, marker='^', linewidths=0, 
        label='groundtruth', zorder=1)

    # stem for reconstructed doa
    for k in range(K_est):
        ax.plot([phi_recon[k], phi_recon[k]], [base, base + 
            height*alpha_recon[k]], linewidth=0.5, linestyle='-', 
            color=col_recon, alpha=1.,zorder=2)

    # markers for reconstructed doa
    ax.scatter(phi_recon, base + height*alpha_recon, c=np.tile(col_recon, 
        (K_est, 1)), s=100, alpha=1., marker='*', linewidths=0, 
        label='reconstruction', zorder=2)



    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:3], framealpha=1.,
              scatterpoints=1, loc=8, 
              ncol=1, bbox_to_anchor=(0.85, -0.22),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)

    ax.set_xlabel(r'DOA') #, fontsize=11)
    ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.set_yticks([1])
    ax.set_yticklabels([])
    #ax.set_yticks(np.linspace(0, 1, 2))
    ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':', linewidth=0.7)
    ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--', linewidth=0.7)
    ax.set_ylim([0, base + height])

    plt.tight_layout(pad=0.5)

    filename = 'figures/experiment_9_mics_10_src'
    plt.savefig(filename + '.pdf', format='pdf') #, transparent=True)
    plt.savefig(filename + '.png', format='png') #, transparent=True)

