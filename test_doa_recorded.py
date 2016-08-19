from __future__ import division
import numpy as np
from scipy import linalg
from scipy.io import wavfile
import os, sys, getopt
import time
import matplotlib.pyplot as plt

import pyroomacoustics as pra
import doa
from tools import *
from experiment import *

if __name__ == '__main__':

    # parse arguments
    argv = sys.argv[1:]
    algo = None
    rec_file = None
    n_bands = None
    try:
        opts, args = getopt.getopt(argv,"ha:f:b:",["algo=","file=","n_bands"])
    except getopt.GetoptError:
        print 'test_doa_recorded.py -a <algo> -f <file>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test_doa_recorded.py -a <algo> -f <file>'
            sys.exit()
        elif opt in ("-a", "--algo"):
            algo = int(arg)
        elif opt in ("-f", "--file"):
            rec_file = arg
        elif opt in ("-b", "--n_bands"):
            n_bands = int(arg)

    # pick microphone array, TODO: ADD SHIFT OF ARRAY
    mic_array = arrays.R_pyramic
    num_mic = mic_array.shape[1]  # number of microphones
    K = rec_file.count('-')+1  # Real number of sources
    K_est = K  # Number of sources to estimate

    # save parameters
    save_fig = False
    save_param = True
    fig_dir = './result/'

    # Check if the directory exists
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # parameters setup
    speed_sound = pra.constants.get('c')

    # algorithm parameters
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'
    fft_size = 1024  # number of FFT bins
    f_array_tuning = 600  # hertz
    M = 5  # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2

    # Import all speech signal
    # -------------------------
    rec_folder = './recordings/jul26-fpga/Sliced_Data/'
    if K==1:
        filename = rec_folder + 'one-speaker/' + rec_file + '.wav'
    elif K==2:
        filename = rec_folder + 'two-speakers/' + rec_file + '.wav'
    fs, speech_signals = wavfile.read(filename)

    # Compute DFT of snapshots
    # -------------------------
    frame_shift_step = np.int(fft_size / 1.)
    y_mic_stft = []
    for k in range(num_mic):
        y_stft = pra.stft(speech_signals[:,k], fft_size, frame_shift_step, transform=rfft).T / np.sqrt(fft_size)
        y_mic_stft.append(y_stft)
    y_mic_stft = np.array(y_mic_stft)

    # True direction of arrival
    # -------------------------
    sources = rec_file.split('-')
    phi_ks = np.array([twitters.doa('FPGA',sources[k])[0] for k in range(K)])

    #----------------------------
    # Perform direction of arrival
    phi_plt = np.linspace(0, 2*np.pi, num=720, dtype=float)
    freq_range = [100., 2000.]
    freq_bins = [int(np.round(f/fs*fft_size)) for f in freq_range]
    freq_bins = np.arange(freq_bins[0],freq_bins[1])

    # Subband selection (may need to avoid search in low and high frequencies if there is something like DC bias or unwanted noise)
    bands_pwr = np.mean(np.mean(np.abs(y_mic_stft[:,freq_bins,:]) ** 2, axis=0), axis=1)
    freq_bins = np.argsort(bands_pwr)[-n_bands:]
    freq_hz = freq_bins*float(fs)/float(fft_size)
    # print('Selected frequency bins: {0}'.format(freq_bins))
    print('Selected frequencies: {0} Hertz'.format(freq_hz))

    # create DOA object
    if algo == 1:
        algo_name = 'SRP-PHAT'
        d = doa.SRP(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, theta=phi_plt)
    if algo == 2:
        algo_name = 'MUSIC'
        d = doa.MUSIC(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est,theta=phi_plt)
    elif algo == 3:
        algo_name = 'CSSM'
        d = doa.CSSM(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, theta=phi_plt, num_iter=10)
    elif algo == 4:
        algo_name = 'WAVES'
        d = doa.WAVES(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, theta=phi_plt, num_iter=10)
    elif algo == 5:
        algo_name = 'TOPS'
        d = doa.TOPS(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, theta=phi_plt)
    elif algo == 6:
        algo_name = 'FRI'
        d = doa.FRI(L=mic_array, fs=fs, nfft=fft_size, num_src=K_est, theta=phi_plt, max_four=M)

    # perform localization
    print 'Applying ' + algo_name + '...'
    # d.locate_sources(y_mic_stft, freq_bins=freq_bins)
    if isinstance(d, doa.TOPS):
        d.locate_sources(y_mic_stft, freq_range=freq_range)
    else:
        print 'using bins'
        d.locate_sources(y_mic_stft, freq_bins=freq_bins)

    # # plot received planewaves
    # mic_count = 0  # signals at which microphone to plot
    # file_name = fig_dir + 'planewave_mic{0}_SNR_{1:.0f}dB.pdf'.format(repr(mic_count), SNR)
    # plt_planewave(y_mic_stft_noiseless[:, fft_bins[0], :],
    #               y_mic_stft[:, fft_bins[0], :], mic=mic_count,
    #               save_fig=save_fig, file_name=file_name)


    # print reconstruction results
    recon_err, sort_idx = polar_distance(d.phi_recon, phi_ks)
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    print('Reconstructed spherical coordinates (in degrees) and amplitudes:')
    if K_est > 1:
        print('Original azimuths        : {0}'.format(np.degrees(phi_ks[sort_idx[:, 1]])))
        print('Reconstructed azimuths   : {0}\n'.format(np.degrees(d.phi_recon[sort_idx[:, 0]])))
    else:
        print('Original azimuths        : {0}'.format(np.degrees(phi_ks)))
        print('Reconstructed azimuths   : {0}\n'.format(np.degrees(d.phi_recon)))
    # print('Original amplitudes      : \n{0}'.format(alpha_ks[sort_idx[:, 1]].squeeze()))
    # print('Reconstructed amplitudes : \n{0}\n'.format(np.real(d.alpha_recon[sort_idx[:, 0]].squeeze())))
    print('Reconstruction error     : {0:.3e}'.format(recon_err))
    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)

    # plot results
    file_name = (fig_dir + 'polar_sources_{0}_numMic_{1}_' +
                 '_locations'+'.pdf').format(repr(rec_file), repr(num_mic))

    # plot response (for FRI one subband)
    d.polar_plt_dirac(phi_ks, file_name=file_name)

    plt.show()
