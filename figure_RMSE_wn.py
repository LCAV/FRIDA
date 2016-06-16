from __future__ import division
import numpy as np
from scipy import linalg
from scipy.signal import fftconvolve
import os
import time
import matplotlib.pyplot as plt

import pyroomacoustics as pra

from utils import polar_distance, load_mic_array_param, load_dirac_param
from generators import gen_diracs_param, gen_mic_array_2d, \
        gen_visibility, gen_dirty_img, gen_sig_at_mic, gen_far_field_ir
from plotters import polar_plt_diracs, plt_planewave

from tools_fri_doa_plane import pt_src_recon, \
    cov_mtx_est, extract_off_diag


if __name__ == '__main__':
    '''
    In this file we will simulate K sources all emitting white noise
    The source are placed circularly in the far field with respect to the microphone array
    '''

    save_fig = True
    save_param = True
    fig_dir = './result/'

    # parameters setup
    fs = 8000  # sampling frequency in Hz
    SNR = 0  # SNR for the received signal at microphones in [dB]

    K = 5           # Real number of sources
    K_est = 5       # Number of sources to estimate
    num_mic = 10    # number of microphones

    # algorithm parameters
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'
    num_snapshot = 256  # number of snapshots used to estimate the covariance matrix
    fft_size = 128      # number of FFT bins
    fc = 2000.
    fft_bins = [int(fc/fs*fft_size)]
    M = 15  # Maximum Fourier coefficient index (-M to M), K_est <= M <= num_mic*(num_mic - 1) / 2

    # Check the directory exists
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    #----------------------------
    # Generate all necessary signals

    # generate source parameters at random
    '''
    alpha_ks, phi_ks, time_stamp = \
        gen_diracs_param(K, positive_amp=True, log_normal_amp=False,
                         semicircle=False, save_param=save_param)
    '''

    # Generate Diracs at random
    phi_ks = 2 * np.pi * np.random.uniform(size=(K))
    #phi_ks = [np.pi/4.]
    power_ks = 10**(SNR/10)*np.ones(K)
    #power_ks = [1.]
    alpha_ks = power_ks #* np.random.normal(size=(K))**2

    # generate microphone array layout
    fraction =0.1
    radius_array =  2000 * 343. / (2*np.pi*fc)  # (2pi * radius_array) compared with the wavelength

    # we would like gradually to switch to our "standardized" functions
    R = pra.spiral_2D_array([0,0], num_mic, radius=radius_array, divi=7, angle=0)
    #R = pra.linear2DArray([0,0], num_mic, 0, radius_array)
    array = pra.Beamformer(R, fs)
    print R

    # Generate the impulse responses for the array and source directions
    ir = gen_far_field_ir([phi_ks], [alpha_ks], R, fs)

    '''
    plt.figure()
    for k in range(K):
        for m in range(num_mic):
            plt.subplot(K, num_mic, k*num_mic + m + 1)
            plt.plot(ir[k,m])
    plt.show()
    '''

    # Now generate some noise
    x = (np.random.normal(size=(num_snapshot*fft_size, K)) * np.sqrt(alpha_ks)).T

    # Now generate all the microphone signals
    y = np.zeros((num_mic,x.shape[1] + ir.shape[2] - 1))
    for src in range(K):
        for mic in range(num_mic):
            y[mic] += fftconvolve(ir[src,mic], x[src])

    # Add noise to the signals
    #y_noisy = y + np.random.normal(size=y.shape)
    y_noisy = y

    # Now do the short time Fourier transform
    # The resulting signal is M x fft_size/2+1 x number of frames
    Y = np.array(
            [pra.stft(signal, fft_size, fft_size, transform=np.fft.rfft).T for signal in y_noisy]
            )

    # Estimate the covariance matrix
    print Y.shape
    cov_mat = np.dot(Y[:,fft_bins[0],:], np.conj(Y[:,fft_bins[0],:].T))/Y.shape[2]/fft_size
    print np.angle(cov_mat)/(2.*np.pi*fraction*np.cos(phi_ks[0]))
    '''
    plt.matshow(np.angle(cov_mat))
    plt.show()
    '''

    # plot received planewaves
    '''
    mic_count = 0  # signals at which microphone to plot
    file_name = fig_dir + 'planewave_mic{0}_SNR_{1:.0f}dB.pdf'.format(repr(mic_count), SNR)
    plt_planewave(y_mic_noiseless, y_mic_noisy, mic=mic_count,
                  save_fig=save_fig, file_name=file_name, SNR=SNR)
    '''

    # extract off-diagonal entries
    visi_noisy = extract_off_diag(cov_mat)

    # simulate the corresponding visibility measurements
    # sigma2_noise = 0.5
    # visi_noisy, SNR, noise_visi, visi_noiseless = \
    #     add_noise(gen_visibility(alpha_ks, phi_ks, p_mic_x, p_mic_y),
    #               sigma2_noise, num_mic, Ns=num_snapshot)
    print('SNR for microphone signals: {0}dB\n'.format(SNR))

    # plot dirty image based on the measured visibilities
    phi_plt = np.linspace(0, 2 * np.pi, num=300, dtype=float)
    dirty_img = gen_dirty_img(visi_noisy, R[0,:], R[1,:], phi_plt)

    # reconstruct point sources with FRI
    max_ini = 50  # maximum number of random initialisation
    #noise_level = np.max([1e-10, linalg.norm(noise_visi.flatten('F'))])
    noise_level = 1
    # tic = time.time()
    phik_recon, alphak_recon = \
            pt_src_recon(visi_noisy, R[0,:], R[1,:], K_est, M, noise_level,
                     max_ini, stop_cri, update_G=True, G_iter=5, verbose=False)
    # toc = time.time()
    # print(toc - tic)

    recon_err, sort_idx = polar_distance(phik_recon, phi_ks)
    print alpha_ks
    print alphak_recon
    # print reconstruction results
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    print('Reconstructed spherical coordinates (in degrees) and amplitudes:')
    print('Original azimuths        : {0}'.format(np.degrees(phi_ks[sort_idx[:, 1]])))
    print('Reconstructed azimuths   : {0}\n'.format(np.degrees(phik_recon[sort_idx[:, 0]])))
    print('Original amplitudes      : {0}'.format(alpha_ks[sort_idx[:, 1]]))
    print('Reconstructed amplitudes : {0}\n'.format(np.real(alphak_recon[sort_idx[:, 0]])))
    print('Reconstruction error     : {0:.3e}'.format(recon_err))
    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)

    # plot results
    polar_plt_diracs(phi_ks, phik_recon, alpha_ks, alphak_recon, num_mic, SNR, save_fig,
                     phi_plt=phi_plt, dirty_img=dirty_img)
    plt.show()
