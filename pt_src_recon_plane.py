from __future__ import division
import numpy as np
from scipy import linalg
import os
from tools_fri_doa_plane import polar_distance, gen_diracs_param, gen_mic_array_2d, \
    gen_visibility, pt_src_recon, add_noise, polar_plt_diracs, load_dirac_param


if __name__ == '__main__':
    save_fig = False
    save_param = True
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'

    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # number of point sources
    K = 4
    K_est = 4  # estimated number of Diracs

    num_mic = 8  # number of microphones
    num_snapshot = 256  # number of snapshots used to estimate the covariance matrix

    M = 12  # the Fourier series expansion is between -M to M (at least K_est)

    # generate source parameters at random
    alpha_ks, phi_ks, time_stamp = \
        gen_diracs_param(K, positive_amp=True, log_normal_amp=False,
                         semicircle=False, save_param=save_param)

    # load saved Dirac parameters
    # dirac_file_name = './data/polar_Dirac_' + '26-05_00_20' + '.npz'
    # alpha_ks, phi_ks, time_stamp = load_dirac_param(dirac_file_name)

    print('Dirac parameter tag: ' + time_stamp)

    # generate microphone array layout
    radius_array = 10  # (2pi * radius_array) compared with the wavelength
    p_mic_x, p_mic_y, layout_time_stamp = \
        gen_mic_array_2d(radius_array, num_mic, save_layout=save_param, divi=5,
                         plt_layout=True, save_fig=save_fig, fig_dir=fig_dir)
    print('Array layout tag: ' + layout_time_stamp)

    # simulate the corresponding visibility measurements
    sigma2_noise = 0.5
    visi_noisy, P, noise, visi_noiseless = \
        add_noise(gen_visibility(alpha_ks, phi_ks, p_mic_x, p_mic_y),
                  sigma2_noise, num_mic, Ns=num_snapshot)
    print('PSNR in visibilities: {0}\n'.format(P))

    # reconstruct point sources with FRI
    max_ini = 50  # maximum number of random initialisation
    noise_level = np.max([1e-10, linalg.norm(noise.flatten('F'))])
    phik_recon, alphak_recon = \
        pt_src_recon(visi_noisy, p_mic_x, p_mic_y, K_est, M, noise_level,
                     max_ini, stop_cri, update_G=True, G_iter=5, verbose=True)

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
                 time_stamp + '.pdf').format(repr(K), repr(num_mic), P)
    polar_plt_diracs(phi_ks, phik_recon, num_mic, P, save_fig, file_name=file_name)
