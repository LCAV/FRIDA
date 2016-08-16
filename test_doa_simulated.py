from __future__ import division
import numpy as np
from scipy import linalg
from scipy.io import wavfile
import os
import time
import matplotlib.pyplot as plt

import pyroomacoustics as pra
import doa

from utils import polar_distance, load_mic_array_param, load_dirac_param
from generators import gen_diracs_param, gen_dirty_img, gen_speech_at_mic_stft
from plotters import polar_plt_diracs, plt_planewave

from tools_fri_doa_plane import pt_src_recon_multiband, extract_off_diag, cov_mtx_est

if __name__ == '__main__':
    '''
    In this file we will simulate K sources all emitting white noise
    The source are placed circularly in the far field with respect to the microphone array
    '''

    save_fig = False
    save_param = True
    fig_dir = './result/'
    exp_dir = './Experiment/'
    # speech_files = ['fq_sample1.wav', 'fq_sample2.wav']
    speech_files = ['fq_sample1.wav']

    # Check if the directory exists
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # parameters setup
    fs = 16000  # sampling frequency in Hz
    SNR = 0  # SNR for the received signal at microphones in [dB]
    speed_sound = pra.constants.get('c')

    num_mic = 6  # number of microphones
    K = len(speech_files)  # Real number of sources
    K_est = K  # Number of sources to estimate

    # algorithm parameters
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'
    fft_size = 1024  # number of FFT bins
    n_bands = 4
    f_array_tuning = 600  # hertz
    M = 14  # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2

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
    # dirac_file_name = './data/polar_Dirac_' + '18-06_21_43' + '.npz'
    # alpha_ks, phi_ks, time_stamp = load_dirac_param(dirac_file_name)

    print('Dirac parameter tag: ' + time_stamp)

    # generate microphone array layout
    radius_array = 2.5 * speed_sound / f_array_tuning  # radiaus of antenna arrays

    # we would like gradually to switch to our "standardized" functions
    mic_array_coordinate = pra.spiral_2D_array([0, 0], num_mic,
                                               radius=radius_array,
                                               divi=7, angle=0)
    # R = pra.linear2DArray([0,0], num_mic, 0, radius_array)
    # array = pra.Beamformer(R, fs)

    # generate complex base-band signal received at microphones
    y_mic_stft, y_mic_stft_noiseless, speech_stft = \
        gen_speech_at_mic_stft(phi_ks, speech_signals, mic_array_coordinate, noise_power, fs, fft_size=fft_size)

    # ----------------------------
    # Perform direction of arrival
    algo = 6
    phi_plt = np.linspace(0, 2*np.pi, num=300, dtype=float)
    freq_range = [100., 900.]

    # create DOA object
    if algo == 1:
        d = doa.SRP(L=mic_array_coordinate, fs=fs, nfft=fft_size, num_sources=K_est, theta=phi_plt)
    if algo == 2:
        d = doa.MUSIC(L=mic_array_coordinate, fs=fs, nfft=fft_size, num_sources=K_est,theta=phi_plt)
    elif algo == 3:
        d = doa.CSSM(L=mic_array_coordinate, fs=fs, nfft=fft_size, num_sources=K_est, theta=phi_plt, num_iter=10)
    elif algo == 4:
        d = doa.WAVES(L=mic_array_coordinate, fs=fs, nfft=fft_size, num_sources=K_est, theta=phi_plt, num_iter=10)
    elif algo == 5:
        d = doa.TOPS(L=mic_array_coordinate, fs=fs, nfft=fft_size, num_sources=K_est, theta=phi_plt)
    elif algo == 6:
        d = doa.FRI(L=mic_array_coordinate, fs=fs, nfft=fft_size, num_sources=K_est, theta=phi_plt, num_bands=4, max_four=14)

    # perform localization
    d.locate_sources(y_mic_stft, freq_range=freq_range)    

    # visi_noiseless_all = []
    # for band_count in xrange(fft_bins.size):
    #     # Estimate the covariance matrix and extract off-diagonal entries
    #     visi_noiseless = extract_off_diag(cov_mtx_est(y_mic_stft_noiseless[:, d.fft_bins[band_count], :]))  # TODO: bins as fri class member
    #     visi_noiseless_all.append(visi_noiseless)
    # visi_noiseless_all = np.column_stack(visi_noiseless_all)

    # # plot received planewaves
    # mic_count = 0  # signals at which microphone to plot
    # file_name = fig_dir + 'planewave_mic{0}_SNR_{1:.0f}dB.pdf'.format(repr(mic_count), SNR)
    # plt_planewave(y_mic_stft_noiseless[:, fft_bins[0], :],
    #               y_mic_stft[:, fft_bins[0], :], mic=mic_count,
    #               save_fig=save_fig, file_name=file_name)

    # # stack them as columns
    # visi_noiseless_all = np.column_stack(visi_noiseless_all)
    # visi_noisy_all = np.column_stack(visi_noisy_all)

    # noise_visi_all = visi_noisy_all - visi_noiseless_all

    print('SNR for microphone signals: {0}dB\n'.format(SNR))

    # plot dirty image based on the measured visibilities
    # use one subband to generate the dirty image
    # could also generate an average dirty image over all subbands considered
    # dirty_img = gen_dirty_img(d.visi_noisy_all[:, 0], mic_array_coordinate[0, :], mic_array_coordinate[1, :], 2*np.pi*d.fc[0],sound_speed=343., phi_plt=phi_plt)    # like self.P

    recon_err, sort_idx = polar_distance(d.phi_recon, phi_ks)
    # alpha_ks = np.array([np.mean(np.abs(s_loop) ** 2, axis=1) for s_loop in speech_stft])[:, d.fft_bins]

    # print reconstruction results
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

    # # plot results
    # file_name = (fig_dir + 'polar_K_{0}_numMic_{1}_' +
    #              'noise_{2:.0f}dB_locations' +
    #              time_stamp + '.pdf').format(repr(K), repr(num_mic), SNR)
    # # here we use the amplitudes in ONE subband for plotting
    # polar_plt_diracs(phi_ks, phik_recon, np.mean(alpha_ks, axis=1).squeeze(),
    #                  np.mean(alphak_recon, axis=1), num_mic, SNR, save_fig,
    #                  file_name=file_name, phi_plt=phi_plt, dirty_img=dirty_img)
    # plt.show()
