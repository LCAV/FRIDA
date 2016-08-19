
from __future__ import division
import numpy as np
from scipy import linalg
import os
import time
import matplotlib.pyplot as plt

import pyroomacoustics as pra

import doa
from tools import *
from utils import *

if __name__ == '__main__':
    '''
    In this file we will simulate K sources all emitting white noise
    The source are placed circularly in the far field with respect to the microphone array
    '''

    save_fig = False
    save_param = False
    fig_dir = './result/'

    # Check if the directory exists
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Simulation parameters
    SNR = np.arange(-10., 20., 5.)
    sim_loops = 1
    thresh_detect = 1e-2

    # Fixed parameters
    fs = 8000  # sampling frequency in Hz
    speed_sound = pra.constants.get('c')

    K = 5  # Real number of sources
    K_est = 5  # Number of sources to estimate
    num_mic = 10  # number of microphones

    # algorithm parameters
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'
    max_ini = 50  # maximum number of random initialisation
    num_snapshot = 256  # number of snapshots used to estimate the covariance matrix
    fft_size = 1024  # number of FFT bins
    fc = np.array([500.])      # frequency in Hertz
    fft_bins = [int(f / fs * fft_size) for f in fc]
    M = 15  # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2

    # generate microphone array layout
    radius_array = 2.5 * speed_sound / fc[0]  # radius of antenna arrays

    # This is the microphone array
    mic_array_coordinate = pra.spiral_2D_array([0, 0], num_mic,
                                               radius=radius_array,
                                               divi=7, angle=0)

    # The grid for the search
    phi_plt = np.linspace(0, 2*np.pi, num=300, dtype=float)

    # Create the doa object
    algo = doa.algos['SRP']

    d = algo(L=mic_array_coordinate, 
            fs=fs, 
            nfft=fft_size, 
            num_sources=K_est, 
            theta=phi_plt, 
            num_bands=4, 
            max_four=14)

    # Receptacle arrays
    avg_error = np.zeros((SNR.shape[0], sim_loops))
    num_recov_sources = np.zeros((SNR.shape[0], sim_loops))

    # ----------------------------
    # Generate all necessary signals

    for i in range(SNR.shape[0]):
        for loop in range(sim_loops):

            # Generate Diracs at random
            alpha_ks, phi_ks, time_stamp = gen_diracs_param(
                    K, positive_amp=True, log_normal_amp=False,
                    semicircle=False, save_param=save_param
                    )

            # generate complex base-band signal received at microphones
            y_mic_stft, y_mic_stft_noiseless = \
                gen_sig_at_mic_stft(phi_ks, alpha_ks, mic_array_coordinate, SNR[i],
                                    fs, fft_size=fft_size, Ns=num_snapshot)

            # Run the reconstruction algorithm
            d.locate_sources(y_mic_stft, freq=fc)

            # Sort the sources
            recon_err, sort_idx = polar_distance(d.phi_recon, phi_ks)

            # reorder the arrays for convenience
            #alphak_recon_s = d.alpha_recon[sort_idx[:,0]]
            phik_recon_s = d.phi_recon[sort_idx[:,0]]
            phi_ks_s = phi_ks[sort_idx[:,1]]

            # find large sources
            #I_large = np.where(alphak_recon_s > thresh_detect)

            # Average error of reconstructed sources
            avg_error[i,loop] = np.mean(polar_error(phik_recon_s, phi_ks_s))
            #avg_error[i,loop] = np.mean(polar_error(phik_recon_s[I_large], phi_ks_s[I_large]))

            # Count the number of recovered sources
            #num_recov_sources[i,loop] = np.count_nonzero(alphak_recon > 1e-2)

