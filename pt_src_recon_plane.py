from __future__ import division
import numpy as np
from scipy import linalg
import os
import matplotlib.pyplot as plt
from tools_fri_doa_plane import polar_distance, gen_diracs_param, gen_mic_array_2d, \
    gen_visibility, pt_src_recon, add_noise, polar_plt_diracs, load_dirac_param, \
    gen_dirty_img, gen_sig_at_mic, cov_mtx_est, extract_off_diag


if __name__ == '__main__':
    save_fig = False
    save_param = True
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'

    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # number of point sources
    K = 5
    K_est = 5  # estimated number of Diracs

    num_mic = 10  # number of microphones
    num_snapshot = 256  # number of snapshots used to estimate the covariance matrix

    # the Fourier series expansion is between -M to M (at least K_est)
    # at most num_mic * (num_mic - 1 ) / 2
    M = 12

    # generate source parameters at random
    alpha_ks, phi_ks, time_stamp = \
        gen_diracs_param(K, positive_amp=True, log_normal_amp=False,
                         semicircle=False, save_param=save_param)

    # load saved Dirac parameters
    # dirac_file_name = './data/polar_Dirac_' + '27-05_14_21' + '.npz'
    # alpha_ks, phi_ks, time_stamp = load_dirac_param(dirac_file_name)

    print('Dirac parameter tag: ' + time_stamp)

    # generate microphone array layout
    radius_array = 10  # (2pi * radius_array) compared with the wavelength
    p_mic_x, p_mic_y, layout_time_stamp = \
        gen_mic_array_2d(radius_array, num_mic, save_layout=save_param, divi=5,
                         plt_layout=True, save_fig=save_fig, fig_dir=fig_dir)
    print('Array layout tag: ' + layout_time_stamp)

    SNR = 5  # SNR for the received signal at microphones in [dB]
    # received signal at microphnes
    y_mic_noisy, y_mic_noiseless = \
        gen_sig_at_mic(alpha_ks, phi_ks, p_mic_x, p_mic_y, SNR, Ns=num_snapshot)

    # extract off-diagonal entries
    visi_noisy = extract_off_diag(cov_mtx_est(y_mic_noisy))
    visi_noiseless = extract_off_diag(cov_mtx_est(y_mic_noiseless))

    noise_visi = visi_noisy - visi_noiseless

    # simulate the corresponding visibility measurements
    # sigma2_noise = 0.5
    # visi_noisy, SNR, noise_visi, visi_noiseless = \
    #     add_noise(gen_visibility(alpha_ks, phi_ks, p_mic_x, p_mic_y),
    #               sigma2_noise, num_mic, Ns=num_snapshot)
    print('SNR for microphone signals: {0}\n'.format(SNR))

    # plot dirty image based on the measured visibilities
    phi_plt = np.linspace(0, 2 * np.pi, num=300, dtype=float)
    dirty_img = gen_dirty_img(visi_noisy, p_mic_x, p_mic_y, phi_plt)

    # reconstruct point sources with FRI
    max_ini = 50  # maximum number of random initialisation
    noise_level = np.max([1e-10, linalg.norm(noise_visi.flatten('F'))])
    phik_recon, alphak_recon = \
        pt_src_recon(visi_noisy, p_mic_x, p_mic_y, K_est, M, noise_level,
                     max_ini, stop_cri, update_G=True, G_iter=5, verbose=False)

    recon_err, sort_idx = polar_distance(phik_recon, phi_ks)
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
    file_name = (fig_dir + 'polar_K_{0}_numMic_{1}_' +
                 'noise_{2:.0f}dB_locations' +
                 time_stamp + '.pdf').format(repr(K), repr(num_mic), SNR)
    polar_plt_diracs(phi_ks, phik_recon, num_mic, SNR, save_fig,
                     file_name=file_name, phi_plt=phi_plt, dirty_img=dirty_img)
    plt.show()
