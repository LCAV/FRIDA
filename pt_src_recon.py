from __future__ import division
import numpy as np
from scipy import linalg
import os
from tools_fri_doa import sph_distance, sph_gen_diracs_param, load_dirac_param, \
    sph_gen_mic_array, sph_gen_visibility, sph_recon_2d_dirac, sph_plot_diracs, \
    add_noise, sph_gen_sig_at_mic, sph_extract_off_diag, sph_cov_mtx_est, sph_plt_planewave


if __name__ == '__main__':
    save_fig = True
    save_param = True
    stop_cri = 'max_iter'  # can be 'mse' or 'max_iter'

    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # number of point sources
    K = 5
    K_est = 5  # estimated number of Diracs

    num_bands = 3  # number of sub-bands considered
    num_mic = 16  # number of microphones

    # generate source parameters at random
    alpha_ks, theta_ks, phi_ks, time_stamp = \
        sph_gen_diracs_param(K, num_bands=num_bands,
                             semisphere=False,
                             log_normal_amp=False,
                             save_param=save_param)

    # load saved Dirac parameters
    # dirac_file_name = './data/sph_Dirac_' + '04-06_22_32' + '.npz'
    # alpha_ks, theta_ks, phi_ks, time_stamp = load_dirac_param(dirac_file_name)
    #
    # # alpha_ks = alpha_ks[:, :num_bands]
    # alpha_ks = np.hstack((alpha_ks, np.tile(alpha_ks[:, -1][:, np.newaxis],
    #                                         (1, num_bands - alpha_ks.shape[1]))
    #                       ))

    print('Dirac parameter tag: ' + time_stamp)

    # generate microphone array layout
    radius_array = 10  # maximum baseline in the microphone array is twice this value
    # r_mic_x, r_mic_y, r_mic_z, mid_band_freq, layout_time_stamp = \
    #     sph_gen_mic_array(radius_array, num_mic, num_bands=num_bands,
    #                       max_ratio_omega=2, save_layout=save_param, divi=7,
    #                       plt_layout=True, save_fig=save_fig, fig_dir=fig_dir)
    r_mic_x, r_mic_y, r_mic_z, mid_band_freq, layout_time_stamp = \
        sph_gen_mic_array(radius_array, num_mic, num_bands=num_bands,
                          max_ratio_omega=2, save_layout=save_param,
                          plt_layout=True, save_fig=save_fig, fig_dir=fig_dir)
    print('Array layout tag: ' + layout_time_stamp)

    # # simulate the corresponding visibility measurements
    # visi_noiseless = sph_gen_visibility(alpha_ks, theta_ks, phi_ks,
    #                                     r_mic_x, r_mic_y, r_mic_z)
    #
    # # add noise
    # var_noise = np.tile(.1, num_bands)  # noise amplitude
    # visi_noisy, P_bands, noise_visi, visi_noiseless_off_diag = \
    #     add_noise(visi_noiseless, var_noise, num_mic, num_bands, Ns=256)
    # P = 20 * np.log10(linalg.norm(visi_noiseless_off_diag, 'fro') / linalg.norm(noise, 'fro'))
    # print(P)

    SNR = 0  # SNR for the received signal at microphones in [dB]
    # SNR = float('inf')
    # received signal at microphnes
    y_mic_noisy, y_mic_noiseless = \
        sph_gen_sig_at_mic(alpha_ks, theta_ks, phi_ks,
                           r_mic_x, r_mic_y, r_mic_z, mid_band_freq,
                           SNR, Ns=256)

    # plot received planewaves
    mic_count = 0  # signals at which microphone to plot
    band_count = 0  # which sub-band to plot
    file_name = fig_dir + 'sph_planewave_band{0}_mic{1}_SNR_{2:.0f}dB.pdf'.format(repr(band_count),
                                                                              repr(mic_count), SNR)
    sph_plt_planewave(y_mic_noiseless, y_mic_noisy, mic_count,
                      band_count, save_fig, file_name=file_name, SNR=SNR)

    # extract off-diagonal entries
    visi_noisy = sph_extract_off_diag(sph_cov_mtx_est(y_mic_noisy))
    visi_noiseless = sph_extract_off_diag(sph_cov_mtx_est(y_mic_noiseless))
    noise_visi = visi_noisy - visi_noiseless

    # reconstruct point sources with FRI
    L = 10  # maximum degree of spherical harmonics
    max_ini = 25  # maximum number of random initialisation
    noise_level = np.max([1e-10, linalg.norm(noise_visi, 'fro')])
    thetak_recon, phik_recon, alphak_recon = \
        sph_recon_2d_dirac(visi_noisy, r_mic_x, r_mic_y, r_mic_z, K_est, L,
                           noise_level, max_ini, stop_cri,
                           num_rotation=5, verbose=True,
                           update_G=True, G_iter=2)

    dist_recon, idx_sort = sph_distance(1, theta_ks, phi_ks, thetak_recon, phik_recon)

    # print reconstruction results
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    print('Reconstructed spherical coordinates (in degrees) and amplitudes:')
    print('Original colatitudes     : {0}'.format(np.degrees(theta_ks[idx_sort[:, 0]])))
    print('Reconstructed colatitudes: {0}\n'.format(np.degrees(thetak_recon[idx_sort[:, 1]])))
    print('Original azimuths        : {0}'.format(np.degrees(phi_ks[idx_sort[:, 0]])))
    print('Reconstructed azimuths   : {0}\n'.format(np.degrees(phik_recon[idx_sort[:, 1]])))
    print('Original amplitudes      : \n{0}'.format(alpha_ks[idx_sort[:, 0], :]))
    print('Reconstructed amplitudes : \n{0}\n'.format(np.real(alphak_recon[idx_sort[:, 1], :])))
    print('Reconstruction error (great-circle distance) : {0:.3e}'.format(dist_recon))
    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)

    # plot results
    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    file_name = (fig_dir + 'sph_K_{0}_numSta_{1}_' +
                 'noise_{2:.0f}dB_locations' +
                 time_stamp + '.pdf').format(repr(K), repr(num_mic), SNR)
    sph_plot_diracs(theta_ks, phi_ks, thetak_recon, phik_recon,
                    num_mic, SNR, save_fig, file_name)
