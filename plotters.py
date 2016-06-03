
from __future__ import division

import numpy as np
import os

if os.environ.get('DISPLAY') is None:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm

from utils import polar_distance

def plt_planewave(y_mic_noiseless, y_mic_noisy, mic=0, save_fig=False, **kwargs):
    """
    plot received planewaves at microhpnes
    :param y_mic_noiseless: the noiseless planewave
    :param y_mic_noisy: the noisy planewave
    :param mic: planewave at which microphone to plot
    :return:
    """
    if 'SNR' in kwargs:
        SNR = kwargs['SNR']
    else:
        SNR = 20 * np.log10(linalg.norm(y_mic_noiseless[mic, :].flatten('F')) /
                            linalg.norm((y_mic_noisy[mic, :] - y_mic_noiseless[mic, :]).flatten('F')))
    plt.figure(figsize=(6, 3), dpi=90)
    ax1 = plt.axes([0.1, 0.53, 0.85, 0.32])
    plt.plot(np.real(y_mic_noiseless[mic,:]), color=[0, 0.447, 0.741],
             linestyle='--', linewidth=1.5, label='original')
    plt.plot(np.real(y_mic_noisy[mic,:]), color=[0.850, 0.325, 0.098],
             linestyle='-', linewidth=1, label='reconstruction')

    plt.xlim([0, y_mic_noisy.shape[1] - 1])
    # plt.xlabel(r'time snapshot', fontsize=11)
    plt.ylabel(r'$\Re\{y(\omega, t)\}$', fontsize=11)

    ax1.yaxis.major.locator.set_params(nbins=5)
    plt.legend(framealpha=0.5, scatterpoints=1, loc=0,
               fontsize=9, ncol=2, handletextpad=.2,
               columnspacing=1.7, labelspacing=0.1)

    plt.title(r'received planewaves at microphe {0} ($\mbox{{SNR}} = {1:.1f}$dB)'.format(repr(mic), SNR),
              fontsize=11)

    ax2 = plt.axes([0.1, 0.14, 0.85, 0.32])
    plt.plot(np.imag(y_mic_noiseless[mic,:]), color=[0, 0.447, 0.741],
             linestyle='--', linewidth=1.5, label='original')
    plt.plot(np.imag(y_mic_noisy[mic,:]), color=[0.850, 0.325, 0.098],
             linestyle='-', linewidth=1, label='reconstruction')

    plt.xlim([0, y_mic_noisy.shape[1] - 1])
    plt.xlabel(r'time snapshot', fontsize=11)
    plt.ylabel(r'$\Im\{y(\omega, t)\}$', fontsize=11)

    ax2.yaxis.major.locator.set_params(nbins=5)

    if save_fig:
        if 'file_name' in kwargs:
            file_name = kwargs['file_name']
        else:
            file_name = 'planewave_mic{0}.pdf'.format(repr(mic))
        plt.savefig(file_name, format='pdf', dpi=300, transparent=True)

def polar_plt_diracs(phi_ref, phi_recon, alpha_ref, alpha_recon, num_mic, P, save_fig=False, **kwargs):
    """
    plot Diracs in the polar coordinate
    :param phi_ref: ground truth Dirac locations (azimuths)
    :param phi_recon: reconstructed Dirac locations (azimuths)
    :param alpha_ref: ground truth Dirac amplitudes
    :param alpha_recon: reconstructed Dirac amplitudes
    :param num_mic: number of microphones
    :param P: PSNR in the visibility measurements
    :param save_fig: whether save the figure or not
    :param kwargs: optional input argument(s), include:
            file_name: file name used to save figure
    :return:
    """
    dist_recon = polar_distance(phi_ref, phi_recon)[0]
    if 'dirty_img' in kwargs and 'phi_plt' in kwargs:
        plt_dirty_img = True
        dirty_img = kwargs['dirty_img']
        phi_plt = kwargs['phi_plt']
    else:
        plt_dirty_img = False
    fig = plt.figure(figsize=(5, 4), dpi=90)
    ax = fig.add_subplot(111, projection='polar')
    K = phi_ref.size
    K_est = phi_recon.size

    ax.scatter(phi_ref, 1 + alpha_ref, c=np.tile([0, 0.447, 0.741], (K, 1)), s=70,
               alpha=0.75, marker='^', linewidths=0, label='original')
    ax.scatter(phi_recon, 1 + alpha_recon, c=np.tile([0.850, 0.325, 0.098], (K_est, 1)), s=100,
               alpha=0.75, marker='*', linewidths=0, label='reconstruction')
    for k in xrange(K):
        ax.plot([phi_ref[k], phi_ref[k]], [1, 1 + alpha_ref[k]],
                linewidth=1.5, linestyle='-', color=[0, 0.447, 0.741])

    for k in xrange(K_est):
        ax.plot([phi_recon[k], phi_recon[k]], [1, 1 + alpha_recon[k]],
                linewidth=1.5, linestyle='-', color=[0.850, 0.325, 0.098])

    if plt_dirty_img:
        dirty_img = dirty_img.real
        min_val = dirty_img.min()
        max_val = dirty_img.max()
        # color_lines = cm.spectral_r((dirty_img - min_val) / (max_val - min_val))
        # ax.scatter(phi_plt, 1 + dirty_img, edgecolor='none', linewidths=0,
        #         c=color_lines, label='dirty image')  # 1 is for the offset
        ax.plot(phi_plt, 1 + dirty_img, linewidth=1.5,
                linestyle='-', color=[0.466, 0.674, 0.188], label='dirty image')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:3], framealpha=0.5,
              scatterpoints=1, loc=8, fontsize=9,
              ncol=1, bbox_to_anchor=(0.9, -0.17),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)
    title_str = r'$K={0}$, $\mbox{{\# of mic.}}={1}$, $\mbox{{SNR}}={2:.1f}$dB, average error={3:.1e}'
    ax.set_title(title_str.format(repr(K), repr(num_mic), P, dist_recon),
                 fontsize=11)
    ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
    ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
    ax.xaxis.set_label_coords(0.5, -0.11)
    ax.set_yticks(np.linspace(0, 1, 2))
    ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
    ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
    if plt_dirty_img:
        ax.set_ylim([0, 1.05 + np.max(np.append(np.concatenate((alpha_ref, alpha_recon)), max_val))])
    else:
        ax.set_ylim([0, 1.05 + np.max(np.concatenate((alpha_ref, alpha_recon)))])
    if save_fig:
        if 'file_name' in kwargs:
            file_name = kwargs['file_name']
        else:
            file_name = 'polar_recon_dirac.pdf'
        plt.savefig(file_name, format='pdf', dpi=300, transparent=True)
        # plt.show()

