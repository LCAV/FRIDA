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
from experiment import arrays

if __name__ == '__main__':

    # parse arguments
    argv = sys.argv[1:]
    algo = None
    num_src = None
    try:
        opts, args = getopt.getopt(argv,"ha:n:b:",["algo=","num_src=", "n_bands"])
    except getopt.GetoptError:
        print 'test_doa_simulated.py -a <algo> -n <num_src> -b <n_bands>' 
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test_doa_simulated.py -a <algo> -n <num_src> -b <n_bands>' 
            sys.exit()
        elif opt in ("-a", "--algo"):
            algo = int(arg)
        elif opt in ("-n", "--num_src"):
            num_src = int(arg)
        elif opt in ("-b", "--n_bands"):
            n_bands = int(arg)

    # file parameters
    save_fig = False
    save_param = True
    fig_dir = './result/'
    exp_dir = './experiment/'
    available_files = ['samples/fq_sample1.wav', 'samples/fq_sample2.wav']
    speech_files = available_files[:num_src]

    # Check if the directory exists
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # parameters setup
    fs = 16000  # sampling frequency in Hz
    SNR = 5  # SNR for the received signal at microphones in [dB]
    speed_sound = pra.constants.get('c')

    R_flat_I = range(8, 16) + range(24, 32) + range(40, 48)
    mic_array_coordinate = arrays['pyramic_tetrahedron'][:2,R_flat_I]
    num_mic = mic_array_coordinate.shape[1]

    K = len(speech_files)  # Real number of sources
    K_est = K  # Number of sources to estimate

    # algorithm parameters
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'
    fft_size = 256  # number of FFT bins
    M = 13  # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2

    # Import all speech signals
    # -------------------------
    speech_signals = []
    for f in speech_files:
        filename = exp_dir + f
        r, s = wavfile.read(filename)
        if r != fs:
            raise ValueError('All speech samples should have correct sampling frequency %d' % (fs))
        speech_signals.append(s)

    # Pad signals so that they all have the same number of samples
    signal_length = np.max([s.shape[0] for s in speech_signals])
    for i, s in enumerate(speech_signals):
        speech_signals[i] = np.concatenate((s, np.zeros(signal_length - s.shape[0])))
    speech_signals = np.array(speech_signals)

    # Adjust the SNR
    for s in speech_signals:
        # We normalize each signal so that \sum E[x_k**2] / E[ n**2 ] = SNR
        s /= np.std(s[np.abs(s) > 1e-2]) * K
    noise_power = 10 ** (-SNR * 0.1)

    # ----------------------------
    # Generate all necessary signals

    # Generate Diracs at random
    alpha_ks, phi_ks, time_stamp = gen_diracs_param(K, positive_amp=True, log_normal_amp=False, semicircle=False, save_param=save_param)

    # load saved Dirac parameters
    '''
    dirac_file_name = './data/polar_Dirac_' + '31-08_09_14' + '.npz'
    alpha_ks, phi_ks, time_stamp = load_dirac_param(dirac_file_name)
    phi_ks[1] = phi_ks[0] + 10. / 180. * np.pi
    '''

    print('Dirac parameter tag: ' + time_stamp)

    # generate complex base-band signal received at microphones
    y_mic_stft, y_mic_stft_noiseless, speech_stft = \
        gen_speech_at_mic_stft(phi_ks, speech_signals, mic_array_coordinate, noise_power, fs, fft_size=fft_size)

    # ----------------------------
    # Perform direction of arrival
    phi_plt = np.linspace(0, 2*np.pi, num=360, dtype=float)
    freq_range = [100., 2000.]
    freq_bnd = [int(np.round(f/fs*fft_size)) for f in freq_range]
    freq_bins = np.arange(freq_bnd[0],freq_bnd[1])
    fmin = min(freq_bins)

    # Subband selection (may need to avoid search in low and high frequencies if there is something like DC bias or unwanted noise)
    # bands_pwr = np.mean(np.mean(np.abs(y_mic_stft[:,fft_bins,:]) ** 2, axis=0), axis=1)
    bands_pwr = np.mean(np.mean(np.abs(y_mic_stft[:,freq_bins,:]) ** 2, axis=0), axis=1)
    freq_bins = np.argsort(bands_pwr)[-n_bands:] + fmin
    freq_hz = freq_bins*float(fs)/float(fft_size)

    freq_hz = np.linspace(freq_range[0], freq_range[1], n_bands)
    freq_bins = np.array([int(np.round(f / fs * fft_size)) for f in freq_hz])

    print('Selected frequency bins: {0}'.format(freq_bins))
    print('Selected frequencies: {0} Hertz'.format(freq_hz))

    # create DOA object
    if algo == 1:
        algo_name = 'SRP-PHAT'
        d = doa.SRP(L=mic_array_coordinate, fs=fs, nfft=fft_size, 
            num_src=K_est, theta=phi_plt)
    if algo == 2:
        algo_name = 'MUSIC'
        d = doa.MUSIC(L=mic_array_coordinate, fs=fs, nfft=fft_size, 
            num_src=K_est,theta=phi_plt)
    elif algo == 3:
        algo_name = 'CSSM'
        d = doa.CSSM(L=mic_array_coordinate, fs=fs, nfft=fft_size, 
            num_src=K_est, theta=phi_plt, num_iter=10)
    elif algo == 4:
        algo_name = 'WAVES'
        d = doa.WAVES(L=mic_array_coordinate, fs=fs, nfft=fft_size, 
            num_src=K_est, theta=phi_plt, num_iter=10)
    elif algo == 5:
        algo_name = 'TOPS'
        d = doa.TOPS(L=mic_array_coordinate, fs=fs, nfft=fft_size, 
            num_src=K_est, theta=phi_plt)
    elif algo == 6:
        algo_name = 'FRI'
        d = doa.FRI(L=mic_array_coordinate, fs=fs, nfft=fft_size, 
            num_src=K_est, theta=phi_plt, max_four=M)

    # perform localization
    print 'Applying ' + algo_name + '...'
    # d.locate_sources(y_mic_stft, fft_bins=fft_bins)
    if isinstance(d, doa.TOPS):
        d.locate_sources(y_mic_stft, freq_range=freq_range)
    else:
        print 'using bins'
        d.locate_sources(y_mic_stft, freq_bins=freq_bins)
        

    print('SNR for microphone signals: {0}dB\n'.format(SNR))

    # print reconstruction results
    recon_err, sort_idx = polar_distance(d.phi_recon, phi_ks)
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    print('Reconstructed spherical coordinates (in degrees) and amplitudes:')
    if d.num_src > 1:
        print('Original azimuths        : {0}'.format(np.degrees(phi_ks[sort_idx[:, 1]])))
        print('Reconstructed azimuths   : {0}\n'.format(np.degrees(d.phi_recon[sort_idx[:, 0]])))
    else:
        print('Original azimuths        : {0}'.format(np.degrees(phi_ks)))
        print('Reconstructed azimuths   : {0}\n'.format(np.degrees(d.phi_recon)))
    # print('Original amplitudes      : \n{0}'.format(alpha_ks[sort_idx[:, 1]].squeeze()))
    # print('Reconstructed amplitudes : \n{0}\n'.format(np.real(d.alpha_recon[sort_idx[:, 0]].squeeze())))
    print('Reconstruction error     : {0:.3e}'.format(np.degrees(recon_err)))
    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)

    # plot results
    file_name = (fig_dir + 'polar_K_{0}_numMic_{1}_' +
                 'noise_{2:.0f}dB_locations' +
                 time_stamp + '.pdf').format(repr(K), repr(num_mic), SNR)

    # plot response (for FRI just one subband)
    if isinstance(d, doa.FRI):
        alpha_ks = np.array([np.mean(np.abs(s_loop) ** 2, axis=1) for s_loop in speech_stft])[:, d.freq_bins]
        d.polar_plt_dirac(phi_ks, np.mean(alpha_ks, axis=1), file_name=file_name)
    else:
        d.polar_plt_dirac(phi_ks, file_name=file_name)

    plt.show()
