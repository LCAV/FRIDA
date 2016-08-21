from __future__ import division
import numpy as np
from scipy import linalg
from scipy.io import wavfile
import os
import time
import matplotlib.pyplot as plt

import pyroomacoustics as pra

from utils import polar_distance, load_mic_array_param, load_dirac_param
from generators import gen_dirty_img
from plotters import polar_plt_diracs
from mkl_fft import rfft
# from experiment import arrays, twitters
# import experiment as experi
from experiment.arrays import R_pyramic
from experiment import twitters

from tools_fri_doa_plane import pt_src_recon_multiband, extract_off_diag, cov_mtx_est


def calculate_speed_of_sound(t, h, p):
    '''
    Compute the speed of sound as a function of
    temperature, humidity and pressure

    Arguments
    ---------

    t: temperature [Celsius]
    h: relative humidity [%]
    p: atmospheric pressure [kpa]

    Return
    ------

    Speed of sound in [m/s]
    '''

    # using crude approximation for now
    return 331.4 + 0.6 * t + 0.0124 * h


if __name__ == '__main__':
    save_fig = False
    save_param = True
    fig_dir = './result/'
    # exp_dir = './experiment/pyramic_recordings/jul26/'
    exp_dir = './experiment/pyramic_recordings/jul26-fpga/Sliced_Data/'
    speech_files = '5-3'

    # Experiment related parameters
    temp = 25.4
    hum = 57.4
    pressure = 1000.
    speed_sound = calculate_speed_of_sound(temp, hum, pressure)

    K = speech_files.count('-') + 1  # number of sources
    K_est = K

    # Import speech signal
    # -------------------------
    if K == 1:
        filename = exp_dir + 'one-speaker/' + speech_files + '.wav'
    elif K == 2:
        filename = exp_dir + 'two-speakers/' + speech_files + '.wav'

    # parameters setup
    fs, speech_signals = wavfile.read(filename)

    speech_signals = speech_signals.astype(float)  # convert to float type
    fs = float(fs)

    # reference DOA for comparison
    sources = speech_files.split('-')
    phi_ks = np.mod(np.array([twitters.doa('FPGA', sources[k])[0]
                              for k in range(K)]),
                    2 * np.pi)

    # algorithm parameters
    stop_cri = 'max_iter'
    fft_size = 64  # number of FFT bins
    n_bands = 6
    M = 15  # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2

    # index of array where the microphones are on the same plane
    array_flat_idx = range(8, 16) + range(24, 32) + range(40, 48)
    mic_array_coordinate = R_pyramic[:, array_flat_idx]
    num_mic = mic_array_coordinate.shape[1]  # number of microphones

    frame_shift_step = np.int(fft_size / 1.)
    y_mic_stft = []
    for mic_count in array_flat_idx:
        y_mic_stft_loop = pra.stft(speech_signals[:, mic_count], fft_size,
                                   frame_shift_step, transform=rfft).T / np.sqrt(fft_size)
        y_mic_stft.append(y_mic_stft_loop)

    y_mic_stft = np.array(y_mic_stft)

    # Subband selection
    # TODO: check why the selected subbands frequencies are so 'quantised'
    # bands_pwr = np.mean(np.mean(np.abs(y_mic_stft) ** 2, axis=0), axis=1)
    # fft_bins = np.argsort(bands_pwr)[-n_bands - 1:-1]
    # ======================================
    max_baseline = 0.25  # <== calculated based on array layout
    f_array_tuning = (2.5 * speed_sound) / (0.5 * max_baseline)
    # print f_array_tuning
    freq_hz = np.linspace(800., 6500., n_bands)  #np.array([800, 1100, 1550, 2100, 2300, 2700])  #
    # freq_hz = np.logspace(np.log10(700.), np.log10(8000.), n_bands)
    # fft_bins = np.array([int(np.round(f / fs * fft_size)) for f in freq_hz])
    fft_bins = np.unique(np.round(freq_hz / fs * fft_size).astype(int))  # <= incase duplicate entries exist
    n_bands = fft_bins.size
    # ======================================
    # # Robin's new version
    # freq_range = [[100, 1000], [8000., 15000.]]
    # fft_bins = []
    # for fb in freq_range:
    #     freq_bnd = [int(np.round(f / fs * fft_size)) for f in fb]
    #
    #     # Subband selection (may need to avoid search in low and high
    #     # frequencies if there is something like DC bias or unwanted noise)
    #     bands_pwr = np.mean(np.mean(
    #         np.abs(y_mic_stft[:, freq_bnd[0]:freq_bnd[1] + 1, :]) ** 2
    #         , axis=0), axis=1)
    #     fft_bins.append(np.argsort(bands_pwr)[-int(n_bands / 2):] + freq_bnd[0])
    #
    # fft_bins = np.concatenate(fft_bins)
    # freq_hz = fft_bins * float(fs) / float(fft_size)
    #
    # freq_hz = np.linspace(500., 8000., n_bands)
    # fft_bins = np.array([int(np.round(f / fs * fft_size)) for f in freq_hz])
    # ======================================

    print('Selected bins: {0} Hertz'.format(fft_bins / fft_size * fs))
    fc = fft_bins / fft_size * fs  # The bands in hertz

    # loop over all subbands
    visi_noisy_all = []
    for band_count in range(fft_bins.size):
        # Estimate the covariance matrix and extract off-diagonal entries
        visi_noisy = extract_off_diag(cov_mtx_est(y_mic_stft[:, fft_bins[band_count], :]))
        visi_noisy_all.append(visi_noisy)

    # stack them as columns
    visi_noisy_all = np.column_stack(visi_noisy_all)

    # plot dirty image based on the measured visibilities
    phi_plt = np.linspace(0, 2 * np.pi, num=300, dtype=float)
    # use one subband to generate the dirty image
    # could also generate an average dirty image over all subbands considered
    dirty_img = np.zeros(phi_plt.shape, dtype=complex)
    for band_count in range(n_bands):
        dirty_img += gen_dirty_img(visi_noisy_all[:, 0], mic_array_coordinate[0, :],
                                  mic_array_coordinate[1, :], 2 * np.pi * fc[band_count],
                                  speed_sound, phi_plt)

    # reconstruct point sources with FRI
    max_ini = 100  # maximum number of random initialisation
    noise_level = 0
    phik_recon, alphak_recon = \
        pt_src_recon_multiband(visi_noisy_all,
                               mic_array_coordinate[0, :],
                               mic_array_coordinate[1, :],
                               2 * np.pi * fc, speed_sound,
                               K_est, M, noise_level,
                               max_ini, update_G=True,
                               G_iter=4, verbose=True)

    recon_err, sort_idx = polar_distance(phik_recon, phi_ks)
    if K_est > 1:
        phi_ks = phi_ks[sort_idx[:, 1]]
        phik_recon = phik_recon[sort_idx[:, 0]]
        alphak_recon = alphak_recon[sort_idx[:, 0]]

    # print reconstruction results
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    print('Reconstructed spherical coordinates (in degrees) and amplitudes:')
    print('Original azimuths        : {0}'.format(np.degrees(phi_ks)))
    print('Reconstructed azimuths   : {0}\n'.format(np.degrees(phik_recon)))
    print('Reconstructed amplitudes : \n{0}\n'.format(np.real(alphak_recon.squeeze())))
    # TODO: <= needs to decide use distance or degree
    print('Reconstruction error     : {0:.3e}'.format(recon_err))
    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)

    # plot results
    file_name = (fig_dir + 'polar_K_{0}_numMic_{1}_locations.pdf').format(repr(K),
                                                                          repr(num_mic))
    # here we use the amplitudes in ONE subband for plotting
    polar_plt_diracs(phi_ks, phik_recon,
                     np.concatenate((np.mean(alphak_recon, axis=1), np.zeros(K - K_est))),
                     np.mean(alphak_recon, axis=1), num_mic, 0, save_fig,
                     file_name=file_name, phi_plt=phi_plt, dirty_img=dirty_img)
    plt.show()
