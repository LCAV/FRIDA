from __future__ import division
import datetime
import time
import numpy as np
import scipy as sp
from scipy import linalg
import scipy.special
import scipy.misc
import scipy.optimize
from functools import partial
from joblib import Parallel, delayed
import os

if os.environ.get('DISPLAY') is None:
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import rcParams

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{bm}"]


def great_circ_dist(r, theta1, phi1, theta2, phi2):
    """
    calculate great circle distance for points located on a sphere
    :param r: radius of the sphere
    :param theta1: colatitude of point 1
    :param phi1: azimuth of point 1
    :param theta2: colatitude of point 2
    :param phi2: azimuth of point 2
    :return: great-circle distance
    """
    dphi = np.abs(phi1 - phi2)
    dist = r * np.arctan2(np.sqrt((np.sin(theta2) * np.sin(dphi)) ** 2 +
                                  (np.sin(theta1) * np.cos(theta2) -
                                   np.cos(theta1) * np.sin(theta2) * np.cos(dphi)) ** 2),
                          np.cos(theta1) * np.cos(theta2) +
                          np.sin(theta1) * np.sin(theta2) * np.cos(dphi))
    return dist


def sph_distance(r, theta1, phi1, theta2, phi2):
    """
    Given two arrays of spherical points p1 = (r, theta1, phi1) and p2 = (r, theta2, phi2),
    pair the cells that are the cloest and provides the pairing matrix index:
    p1[index[0, :]] should be as close as possible to p2[index[1, :]].
    The function outputs the average of the absolute value of the differences
    abs(p1[index[0, :]] - p2[index[1, :]]).
    :param r: radius of the sphere
    :param theta1: colatitudes of points 1 (an array)
    :param phi1: azimuths of points 1 (an array)
    :param theta2: colatitudes of points 2 (an array)
    :param phi2: azimuths of points 2 (an array)
    :return: d: minimum distance
             index: the permutation matrix
    """
    theta1 = np.reshape(theta1, (1, -1), order='F')
    phi1 = np.reshape(phi1, (1, -1), order='F')
    theta2 = np.reshape(theta2, (1, -1), order='F')
    phi2 = np.reshape(phi2, (1, -1), order='F')
    N1 = theta1.size
    N2 = theta2.size
    diffmat = great_circ_dist(r, theta1, phi1,
                              np.reshape(theta2, (-1, 1), order='F'),
                              np.reshape(phi2, (-1, 1), order='F'))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in xrange(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        d = np.mean(np.abs(great_circ_dist(r, theta1[:, index[:, 0]],
                                           phi1[:, index[:, 0]],
                                           theta2[:, index[:, 1]],
                                           phi2[:, index[:, 1]])))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index


def sph_gen_diracs_param(K, num_bands=1, positive_amp=True,
                         log_normal_amp=False, semisphere=True,
                         save_param=True):
    """
    Randomly generate Diracs' parameters.
    :param K: number of Diracs
    :param num_bands: number of sub-bands
    :param positive_amp: whether the Diracs have positive amplitudes or not.
    :param log_normal_amp: whether the Diracs amplitudes follow log-normal distribution.
    :param save_param: whether to save the Diracs' parameter or not.
    :return:
    """
    # amplitudes
    if log_normal_amp:
        positive_amp = True
    if not positive_amp:
        alpha_ks = np.sign(np.random.randn(K, num_bands)) * (0.7 + 0.6 * (np.random.rand(K, num_bands) - 0.5) / 1.)
    elif log_normal_amp:
        alpha_ks = np.random.lognormal(mean=np.log(2), sigma=0.7, size=(K, num_bands))
    else:
        alpha_ks = 0.7 + 0.6 * (np.random.rand(K, num_bands) - 0.5) / 1.

    # locations on the sphere (S^2)
    if K == 1:
        if semisphere:
            theta_ks = 0.5 * np.pi * np.random.rand()  # in [0, pi/2)
        else:
            theta_ks = np.pi * np.random.rand()  # in [0, pi)
        phi_ks = 2. * np.pi * np.random.rand()  # in [0, 2*pi)
    else:
        # for stability reason, we allow theta to be between 0.2*pi to 0.8*pi
        # we may always take a raonodm rotation first to avoid cases where Diracs
        # are located exactly at the zenith.
        # theta_ks = (0.2 + 0.6 * np.random.rand(K)) * np.pi
        # theta_ks = np.random.rand(K) * np.pi
        if semisphere:
            theta_ks = (0.05 + 0.4 * np.random.rand(K)) * np.pi
        else:
            theta_ks = np.sign(np.random.randn(K)) * (0.05 + 0.4 * np.random.rand(K)) * np.pi
        # azimuth
        phi_ks = np.random.rand(K) * 2. * np.pi

    theta_ks[theta_ks < 0] += np.pi
    phi_ks = np.mod(phi_ks, 2 * np.pi)

    time_stamp = datetime.datetime.now().strftime('%d-%m_%H_%M')
    if save_param:
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        file_name = './data/sph_Dirac_' + time_stamp + '.npz'
        np.savez(file_name, alpha_ks=alpha_ks, theta_ks=theta_ks,
                 phi_ks=phi_ks, K=K, num_bands=num_bands, time_stamp=time_stamp)
    return alpha_ks, theta_ks, phi_ks, time_stamp


def load_dirac_param(file_name):
    """
    load stored Diracs' parameters
    :param file_name: the file name that the parameters are stored
    :return:
    """
    stored_param = np.load(file_name)
    alpha_ks = stored_param['alpha_ks']
    theta_ks = stored_param['theta_ks']
    phi_ks = stored_param['phi_ks']
    time_stamp = stored_param['time_stamp'].tostring()
    return alpha_ks, theta_ks, phi_ks, time_stamp


def sph_gen_mic_array(radius_array, num_mic_per_band=1, num_bands=1,
                      max_ratio_omega=1, save_layout=True):
    """
    generate microphone array layout.
    :param radius_array: radius of the microphone array
    :param num_mic_per_band: number of microphones
    :param num_bands: number of narrow bands considered
    :param max_ratio_omega: we normalise the mid-band frequencies w.r.t.
            the narraow band that has the lowest frequencies. The ratio here
            specifies the highest mid-band frequency.
    :param save_layout: whether to save the microphone array layout or not
    :return:
    """
    # TODO: adapt to the realistic setting later
    # assume the array is located on a sphere with radius that is
    # approximated 20 times the radius of stations (almost a flat surface)
    radius_sph = 10. * radius_array
    pos_array_norm = np.linspace(0, radius_array, num=num_mic_per_band, dtype=float)
    pos_array_angle = np.linspace(0, 12. * np.pi, num=num_mic_per_band, dtype=float)

    pos_mic_x = pos_array_norm * np.cos(pos_array_angle)
    pos_mic_y = pos_array_norm * np.sin(pos_array_angle)

    phi_mic = np.arctan2(pos_mic_y, pos_mic_x)
    theta_mic = np.arcsin(np.sqrt(pos_mic_x ** 2 + pos_mic_y ** 2) / radius_sph)

    p_mic0_x, p_mic0_y, p_mic0_z = sph2cart(radius_sph / max_ratio_omega, theta_mic, phi_mic)
    # reshape to use broadcasting
    p_mic0_x = np.reshape(p_mic0_x, (-1, 1), order='F')
    p_mic0_y = np.reshape(p_mic0_y, (-1, 1), order='F')
    p_mic0_z = np.reshape(p_mic0_z, (-1, 1), order='F')

    # mid-band frequencies of different sub-bands
    # mid_band_freq = np.random.rand(1, num_bands) * (max_ratio_omega - 1) + 1
    mid_band_freq = np.linspace(1, max_ratio_omega, num=num_bands, dtype=float)
    # mid_band_freq = np.logspace(0, np.log10(max_ratio_omega), num=num_bands, dtype=float)

    p_mic_x = p_mic0_x * mid_band_freq
    p_mic_y = p_mic0_y * mid_band_freq
    p_mic_z = p_mic0_z * mid_band_freq

    layout_time_stamp = datetime.datetime.now().strftime('%d-%m')
    if save_layout:
        directory = './data/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = directory + 'mic_layout_' + layout_time_stamp + '.npz'
        np.savez(file_name, p_mic_x=p_mic_x, p_mic_y=p_mic_y, p_mic_z=p_mic_z,
                 layout_time_stamp=layout_time_stamp)
    return p_mic_x, p_mic_y, p_mic_z, layout_time_stamp


def load_mic_array_param(file_name):
    """
    load stored microphone array parameters
    :param file_name: file that stored these parameters
    :return:
    """
    stored_param = np.load(file_name)
    p_mic_x = stored_param['p_mic_x']
    p_mic_y = stored_param['p_mic_y']
    p_mic_z = stored_param['p_mic_z']
    layout_time_stamp = stored_param['layout_time_stamp'].tostring()
    return p_mic_x, p_mic_y, p_mic_z, layout_time_stamp


def sph_gen_visibility(alphak, theta_k, phi_k, p_mic_x, p_mic_y, p_mic_z):
    """
    generate visibility from the Dirac parameter and microphone array layout
    :param alphak: Diracs' amplitudes
    :param theta_k: Diracs' co-latitudes
    :param phi_k: azimuths
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :return:
    """
    num_bands = p_mic_x.shape[1]
    xk, yk, zk = sph2cart(1, theta_k, phi_k)  # on S^2, -> r = 1
    # reshape xk, yk, zk to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')
    partial_visi_inner = partial(sph_gen_visi_inner, xk=xk, yk=yk, zk=zk)
    visibility_lst = \
        Parallel(n_jobs=-1, backend='threading')(
            delayed(partial_visi_inner)(alphak[:, band_loop], p_mic_x[:, band_loop],
                                        p_mic_y[:, band_loop], p_mic_z[:, band_loop])
            for band_loop in xrange(num_bands))
    return np.hstack(visibility_lst)


def sph_gen_visi_inner(alphak_loop, p_mic_x_loop, p_mic_y_loop,
                       p_mic_z_loop, xk, yk, zk):
    num_mic_per_band = p_mic_x_loop.size
    # visibility_qqp = np.zeros(((num_mic_per_band - 1) * num_mic_per_band, 1),
    #                           dtype=complex)
    visibility_qqp = np.zeros((num_mic_per_band ** 2, 1), dtype=complex)
    count_inner = 0
    for q in xrange(num_mic_per_band):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        p_z_outer = p_mic_z_loop[q]
        for qp in xrange(num_mic_per_band):
            # if not q == qp:
            p_x_qqp = p_x_outer - p_mic_x_loop[qp]
            p_y_qqp = p_y_outer - p_mic_y_loop[qp]
            p_z_qqp = p_z_outer - p_mic_z_loop[qp]
            visibility_qqp[count_inner] = \
                np.dot(np.exp(-1j * (xk * p_x_qqp + yk * p_y_qqp + zk * p_z_qqp)),
                       alphak_loop).squeeze()
            count_inner += 1
    return visibility_qqp


def add_noise(visi_noiseless, var_noise, num_mic, num_bands, Ns=1000):
    partial_add_noise_inner = partial(add_noise_inner, num_mic=num_mic, Ns=Ns)
    res_all_lst = Parallel(n_jobs=-1, backend='threading')(
        delayed(partial_add_noise_inner)(np.reshape(visi_noiseless[:, band_loop],
                                                    (num_mic, num_mic), order='F'),
                                         var_noise[band_loop])
        for band_loop in xrange(num_bands))
    visi_noisy_lst, P_lst, noise_lst, visi_noiseless_off_diag_lst = zip(*res_all_lst)
    return np.hstack(visi_noisy_lst), np.hstack(P_lst), \
           np.hstack(noise_lst), np.hstack(visi_noiseless_off_diag_lst)


def sph_recon_2d_dirac(a, p_mic_x, p_mic_y, p_mic_z, K, L, noise_level,
                       max_ini=50, stop_cri='mse', num_rotation=5,
                       verbose=False, update_G=False, **kwargs):
    """
    reconstruct point sources on the sphere from the visibility measurements
    :param a: the measured visibilities
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :param K: number of point sources
    :param L: maximum degree of spherical harmonics
    :param noise_level: noise level in the measured visiblities
    :param max_ini: maximum number of random initialisations used
    :param stop_cri: either 'mse' or 'max_iter'
    :param num_rotation: number of random initialisations
    :param verbose: print intermediate results to console or not
    :param update_G: update the linear mapping matrix in the objective function or not
    :return:
    """
    assert L >= K + np.sqrt(K) - 1
    num_mic = p_mic_x.shape[0]
    assert num_mic * (num_mic - 1) >= (L + 1) ** 2
    num_bands = p_mic_x.shape[1]
    # data used when the annihilation is enforced along each column
    a_ri = np.vstack((np.real(a), np.imag(a)))
    a_col_ri = a_ri
    # data used when the annihilation is enforced along each row
    a_row = a_ri

    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    # initialisation
    min_error_rotate = float('inf')
    thetak_opt = np.zeros(K)
    phik_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # SO(3) is defined by three angles
    rotate_angle_all_1 = np.random.rand(num_rotation) * np.pi * 2.
    rotate_angle_all_2 = np.random.rand(num_rotation) * np.pi
    rotate_angle_all_3 = np.random.rand(num_rotation) * np.pi * 2.
    # build rotation matrices for all angles
    for rand_rotate in xrange(num_rotation):
        rotate_angle_1 = rotate_angle_all_1[rand_rotate]
        rotate_angle_2 = rotate_angle_all_2[rand_rotate]
        rotate_angle_3 = rotate_angle_all_3[rand_rotate]
        # build rotation matrices
        rotate_mtx1 = np.array([[np.cos(rotate_angle_1), -np.sin(rotate_angle_1), 0],
                                [np.sin(rotate_angle_1), np.cos(rotate_angle_1), 0],
                                [0, 0, 1]])
        rotate_mtx2 = np.array([[np.cos(rotate_angle_2), 0, np.sin(rotate_angle_2)],
                                [0, 1, 0],
                                [-np.sin(rotate_angle_2), 0, np.cos(rotate_angle_2)]])
        rotate_mtx3 = np.array([[np.cos(rotate_angle_3), -np.sin(rotate_angle_3), 0],
                                [np.sin(rotate_angle_3), np.cos(rotate_angle_3), 0],
                                [0, 0, 1]])
        rotate_mtx = np.dot(rotate_mtx1, np.dot(rotate_mtx2, rotate_mtx3))

        # rotate microphone steering vector (due to change of coordinate w.r.t. the random rotation)
        p_mic_xyz_rotated = np.dot(rotate_mtx,
                                   np.vstack((np.reshape(p_mic_x, (1, -1), order='F'),
                                              np.reshape(p_mic_y, (1, -1), order='F'),
                                              np.reshape(p_mic_z, (1, -1), order='F')
                                              ))
                                   )
        p_mic_x_rotated = np.reshape(p_mic_xyz_rotated[0, :], p_mic_x.shape, order='F')
        p_mic_y_rotated = np.reshape(p_mic_xyz_rotated[1, :], p_mic_x.shape, order='F')
        p_mic_z_rotated = np.reshape(p_mic_xyz_rotated[2, :], p_mic_x.shape, order='F')
        mtx_freq2visibility = \
            sph_mtx_freq2visibility(L, p_mic_x_rotated, p_mic_y_rotated, p_mic_z_rotated)

        # linear transformation matrix that maps uniform samples of sinusoids to visibilities
        G_row = sph_mtx_fri2visibility_row_major(L, mtx_freq2visibility)
        G_col_ri = sph_mtx_fri2visibility_col_major_ri(L, mtx_freq2visibility)

        # ext_r, ext_i = sph_Hsym_ext(L)
        # mtx_hermi = linalg.block_diag(ext_r, ext_i)
        # mtx_repe = np.eye(num_bands, dtype=float)
        # G_col_ri_symb = np.dot(G_col_ri, np.kron(mtx_repe, mtx_hermi))

        for loop_G in xrange(max_loop_G):
            # parallel implentation
            partial_sph_dirac_recon = partial(sph_2d_dirac_v_and_h,
                                              G_row=G_row, a_row=a_row,
                                              G_col=G_col_ri, a_col=a_col_ri,
                                              K=K, L=L, noise_level=noise_level,
                                              max_ini=max_ini, stop_cri=stop_cri)
            coef_error_b_all = Parallel(n_jobs=-1)(delayed(partial_sph_dirac_recon)(ins)
                                                   for ins in xrange(2))
            c_row, error_row, b_opt_row = coef_error_b_all[0]
            c_col, error_col, b_opt_col = coef_error_b_all[1]

            # recover the signal parameters from the reconstructed annihilating filters
            tk = np.roots(np.squeeze(c_col))  # tk: cos(theta_k)
            tk[tk > 1] = 1
            tk[tk < -1] = -1
            theta_ks = np.real(np.arccos(tk))
            uk = np.roots(np.squeeze(c_row))
            phi_ks = np.mod(-np.angle(uk), 2. * np.pi)
            if verbose:
                print(c_col)
                print('noise level: {0:.3e}'.format(noise_level))
                print('objective function value (column): {0:.3e}'.format(error_col))
                print('objective function value (row)   : {0:.3e}'.format(error_row))

            # find the correct associations of theta_ks and phi_ks
            # assume we have K^2 Diracs and find the amplitudes
            theta_k_grid, phi_k_grid = np.meshgrid(theta_ks, phi_ks)
            theta_k_grid = theta_k_grid.flatten('F')
            phi_k_grid = phi_k_grid.flatten('F')

            mtx_amp = sph_mtx_amp(theta_k_grid, phi_k_grid, p_mic_x_rotated,
                                  p_mic_y_rotated, p_mic_z_rotated)
            # the amplitudes are positive (sigma^2)
            alphak_recon = np.reshape(sp.optimize.nnls(np.vstack((mtx_amp.real, mtx_amp.imag)),
                                                       np.concatenate((a.real.flatten('F'),
                                                                       a.imag.flatten('F')))
                                                       )[0],
                                      (-1, p_mic_x.shape[1]), order='F')
            # extract the indices of the largest amplitudes for each sub-band
            alphak_sort_idx = np.argsort(np.abs(alphak_recon), axis=0)[:-K - 1:-1, :]
            # now use majority vote across all sub-bands
            idx_all = np.zeros(alphak_recon.shape, dtype=int)
            for loop in xrange(num_bands):
                idx_all[alphak_sort_idx[:, loop], loop] = 1
            idx_sel = np.argsort(np.sum(idx_all, axis=1))[:-K - 1:-1]
            thetak_recon_rotated = theta_k_grid[idx_sel]
            phik_recon_rotated = phi_k_grid[idx_sel]
            # rotate back
            # transform to cartesian coordinate
            xk_recon, yk_recon, zk_recon = sph2cart(1, thetak_recon_rotated, phik_recon_rotated)
            xyz_rotate_back = linalg.solve(rotate_mtx,
                                           np.vstack((xk_recon.flatten('F'),
                                                      yk_recon.flatten('F'),
                                                      zk_recon.flatten('F')
                                                      ))
                                           )
            xk_recon = np.reshape(xyz_rotate_back[0, :], xk_recon.shape, 'F')
            yk_recon = np.reshape(xyz_rotate_back[1, :], yk_recon.shape, 'F')
            zk_recon = np.reshape(xyz_rotate_back[2, :], zk_recon.shape, 'F')

            # transform back to spherical coordinate
            thetak_recon = np.arccos(zk_recon)
            phik_recon = np.mod(np.arctan2(yk_recon, xk_recon), 2. * np.pi)
            # now use the correctly identified theta and phi to reconstruct the correct amplitudes
            mtx_amp_sorted = sph_mtx_amp(thetak_recon, phik_recon, p_mic_x, p_mic_y, p_mic_z)
            alphak_recon = sp.optimize.nnls(np.vstack((mtx_amp_sorted.real, mtx_amp_sorted.imag)),
                                            np.concatenate((a.real.flatten('F'),
                                                            a.imag.flatten('F')))
                                            )[0]

            error_loop = linalg.norm(a.flatten('F') - np.dot(mtx_amp_sorted, alphak_recon))

            if verbose:
                # for debugging only
                print('theta_ks  : {0}'.format(np.degrees(thetak_recon)))
                print('phi_ks    : {0}'.format(np.degrees(np.mod(phik_recon, 2 * np.pi))))
                print('error_loop: {0}\n'.format(error_loop))

            if error_loop < min_error_rotate:
                min_error_rotate = error_loop
                thetak_opt, phik_opt, alphak_opt = thetak_recon, phik_recon, alphak_recon

            # update the linear transformation matrix
            if update_G:
                G_row = sph_mtx_updated_G_row_major(thetak_recon_rotated, phik_recon_rotated,
                                                    L, p_mic_x_rotated, p_mic_y_rotated,
                                                    p_mic_z_rotated, mtx_freq2visibility)
                G_col_ri = sph_mtx_updated_G_col_major_ri(thetak_recon_rotated,
                                                          phik_recon_rotated, L,
                                                          p_mic_x_rotated,
                                                          p_mic_y_rotated,
                                                          p_mic_z_rotated,
                                                          mtx_freq2visibility)
                # G_col_ri_symb = np.dot(G_col_ri, np.kron(mtx_repe, mtx_hermi))
    return thetak_opt, phik_opt, np.reshape(alphak_opt, (-1, num_bands), order='F')


# ======== functions for the reconstruction of colaitudes and azimuth ========
def add_noise_inner(visi_noiseless, var_noise, num_mic, Ns=1000):
    """
    add noise to the Fourier measurements
    :param visi_noiseless: the noiseless Fourier data
    :param var_noise: variance of the noise
    :param num_mic: number of stations
    :param Ns: number of observation samples used to estimate the covariance matrix
    :return:
    """
    SigmaMtx = visi_noiseless + var_noise * np.eye(*visi_noiseless.shape)
    wischart_mtx = np.kron(SigmaMtx.conj(), SigmaMtx) / Ns
    # the noise vairance matrix is given by the Cholesky decomposition
    noise_conv_mtx_sqrt = np.linalg.cholesky(wischart_mtx)
    # the noiseless visibility
    visi_noiseless_vec = np.reshape(visi_noiseless, (-1, 1), order='F')
    # TODO: find a way to add Hermitian symmetric noise here.
    noise = np.dot(noise_conv_mtx_sqrt,
                   np.random.randn(*visi_noiseless_vec.shape) +
                   1j * np.random.randn(*visi_noiseless_vec.shape)) / np.sqrt(2)
    visi_noisy = np.reshape(visi_noiseless_vec + noise, visi_noiseless.shape, order='F')
    # extract the off-diagonal entries
    extract_cond = np.reshape((1 - np.eye(num_mic)).T.astype(bool), (-1, 1), order='F')
    visi_noisy = np.reshape(np.extract(extract_cond, visi_noisy.T), (-1, 1), order='F')
    visi_noiseless_off_diag = \
        np.reshape(np.extract(extract_cond, visi_noiseless.T), (-1, 1), order='F')
    # calculate the equivalent SNR
    noise = visi_noisy - visi_noiseless_off_diag
    P = 20 * np.log10(linalg.norm(visi_noiseless_off_diag) / linalg.norm(noise))
    return visi_noisy, P, noise, visi_noiseless_off_diag


def sph_2d_dirac_v_and_h(direction, G_row, a_row, G_col, a_col, K, L, noise_level, max_ini, stop_cri):
    """
    used to run the reconstructions along horizontal and vertical directions in parallel.
    """
    if direction == 0:
        c_recon, error_recon, b_opt = \
            sph_2d_dirac_horizontal_ri(G_row, a_row, K, L, noise_level, max_ini, stop_cri)[:3]
    else:
        c_recon, error_recon, b_opt = \
            sph_2d_dirac_vertical_real_coef(G_col, a_col, K, L, noise_level, max_ini, stop_cri)[:3]
        # c_recon, error_recon, b_opt = \
        #     sph_2d_dirac_vertical_symb(G_col, a_col, K, L, noise_level, max_ini, stop_cri)[:3]
    return c_recon, error_recon, b_opt


def sph_2d_dirac_horizontal_ri(G_row, a_ri, K, L, noise_level, max_ini=100, stop_cri='mse'):
    """
    Dirac reconstruction on the sphere along the horizontal direction.
    :param G_row: the linear transformation matrix that links the given spherical harmonics
                    to the annihilable sequence.
    :param a_ri: the SPATIAL domain samples.
            CAUTION: here we assume that the spherical harmonics is rearranged (based on the
            sign of the order 'm'): the spherical harmonics with negative m-s are complex-conjuaged
            and flipped around m = 0, i.e., m varies from 0 to -L. The column that corresponds to
            m = 0 is stored twice.
            The total number of elements in 'a' is: (L + 1) * (L + 2)
    :param K: number of Diracs
    :param L: maximum degree of the spherical harmonics
    :param M: maximum order of the spherical harmonics (M <= L)
    :param noise_level: noise level present in the spherical harmonics
    :param max_ini: maximum number of initialisations allowed for the algorithm
    :param stop_cri: either 'mse' or 'max_iter'
    :return:
    """
    num_bands = a_ri.shape[1]
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G_row: (L + 1)(L + 2) x (L + 1)^2
    GtG = np.dot(G_row.T, G_row)  # size: 2(L + 1)^2 x 2(L + 1)^2
    Gt_a = np.dot(G_row.T, a_ri)

    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    # the least square solution incase G is rank deficient
    # size: 2(L + 1)^2 x num_bands
    beta = np.reshape(linalg.lstsq(G_row, a_ri)[0], (-1, num_bands), order='F')
    # print linalg.norm(np.dot(G_row, beta) - a_ri)
    # size of Tbeta: (L + 1 - K)^2 x (K + 1)
    Tbeta = sph_Tmtx_row_major_ri(beta, K, L)

    # repetition matrix for Rmtx (the same annihilating filter for all sub-bands)
    mtx_repe = np.eye(num_bands, dtype=float)

    # size of various matrices / vectors
    sz_G1 = 2 * (L + 1) ** 2 * num_bands

    sz_Tb0 = 2 * (L + 1 - K) * (L - K + 2) * num_bands
    sz_Tb1 = 2 * (K + 1)

    sz_Rc0 = 2 * (L + 1 - K) * (L - K + 2) * num_bands
    sz_Rc1 = 2 * (L + 1) ** 2 * num_bands

    sz_coef = 2 * (K + 1)
    sz_b = 2 * (L + 1) ** 2 * num_bands  # here the data that corresponds to m = 0 is only stored once.

    rhs = np.concatenate((np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float),
                          np.array(1, dtype=float)[np.newaxis]))
    rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
                                   np.zeros((sz_Rc0, 1), dtype=Gt_a.dtype)
                                   ))
                        )

    # the main iteration with different random initialisations
    for ini in xrange(max_ini):
        c = np.random.randn(sz_coef, 1)  # the real and imaginary parts of the annihilating filter coefficients
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = np.kron(mtx_repe, sph_Rmtx_row_major_ri(c, K, L))
        for inner in xrange(max_iter):
            Mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta.T,
                                             np.zeros((sz_coef, sz_Rc1)), c0
                                             )),
                                  np.hstack((Tbeta, np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop, np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)), -R_loop.T,
                                             GtG, np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop = (Mtx_loop + Mtx_loop.T) / 2.
            c = linalg.solve(Mtx_loop, rhs)[:sz_coef]

            R_loop = np.kron(mtx_repe, sph_Rmtx_row_major_ri(c, K, L))
            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T
                                               )),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))
                                               ))
                                    ))
            b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:sz_b]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G_row, b_recon))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt_row = b_recon
                c_opt_row = c[:K + 1] + 1j * c[K + 1:]  # real and imaginary parts

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt_row, min_error, b_opt_row, ini


def sph_2d_dirac_vertical_real_coef(G_col_ri, a_ri, K, L, noise_level, max_ini=100, stop_cri='mse'):
    """
    Dirac reconstruction on the sphere along the vertical direction.
    Here we use explicitly the fact that the annihilating filter is real-valuded.
    :param G_col: the linear transformation matrix that links the given spherical harmonics
                    to the annihilable sequence.
    :param a: the given spherical harmonics
    :param K: number of Diracs
    :param L: maximum degree of the spherical harmonics
    :param M: maximum order of the spherical harmonics (M <= L)
    :param noise_level: noise level present in the spherical harmonics
    :param max_ini: maximum number of initialisations allowed for the algorithm
    :param stop_cri: either 'mse' or 'maxiter'
    :return:
    """
    num_visi_ri, num_bands = a_ri.shape
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G_col: (L + 1 + (2 * L + 1 - M) * M) x (L + 1 + (2 * L + 1 - M) * M)
    GtG = np.dot(G_col_ri.T, G_col_ri)
    Gt_a = np.dot(G_col_ri.T, a_ri)

    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    beta_ri = np.reshape(linalg.lstsq(G_col_ri, a_ri)[0], (-1, num_bands), order='F')

    # size of Tbeta: num_bands * 2(L + 1 - K)^2 x (K + 1)
    Tbeta = sph_Tmtx_col_major_real_coef(beta_ri, K, L)

    # repetition matrix for Rmtx (the same annihilating filter for all sub-bands)
    mtx_repe = np.eye(num_bands, dtype=float)

    # size of various matrices / vectors
    sz_G1 = 2 * (L + 1) ** 2 * num_bands

    sz_Tb0 = 2 * (L + 1 - K) ** 2 * num_bands
    sz_Tb1 = K + 1

    sz_Rc0 = 2 * (L + 1 - K) ** 2 * num_bands
    sz_Rc1 = 2 * (L + 1) ** 2 * num_bands

    sz_coef = K + 1
    sz_b = 2 * (L + 1) ** 2 * num_bands

    rhs = np.concatenate((np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float),
                          np.array(1, dtype=float)[np.newaxis]))
    rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
                                   np.zeros((sz_Rc0, 1), dtype=Gt_a.dtype)
                                   ))
                        )

    # the main iteration with different random initialisations
    for ini in xrange(max_ini):
        # c = np.random.randn(sz_coef, 1)  # we know that the coefficients are real-valuded
        # favor the polynomial to have roots inside unit circle
        c = np.sort(np.abs(np.random.randn(sz_coef, 1)))
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = np.kron(mtx_repe, sph_Rmtx_col_major_real_coef(c, K, L))
        for inner in xrange(max_iter):
            Mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta.T,
                                             np.zeros((sz_coef, sz_Rc1)), c0
                                             )),
                                  np.hstack((Tbeta, np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop, np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)), -R_loop.T,
                                             GtG, np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop = (Mtx_loop + Mtx_loop.T) / 2.
            c = linalg.solve(Mtx_loop, rhs)[:sz_coef]

            R_loop = np.kron(mtx_repe, sph_Rmtx_col_major_real_coef(c, K, L))
            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                    ))
            b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:sz_b]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G_col_ri, b_recon))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt_col = b_recon
                c_opt_col = c

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt_col, min_error, b_opt_col, ini


# def sph_Hsym_ext(L):
#     """
#     Expansion matrix to impose the Hermitian symmetry in the uniformly
#     sampled sinusoids \tilde{b}_nm.
#     We assume the input to the extension matrix is the tilde{b}_nm with m <= 0.
#     The first half of the data is the real-part of tilde{b}_nm for m <= 0;
#     while the second half of the data is the imaginary-part of tilde{b}_nm for m < 0.
#     The reason is that tilde{b}_n0 is real-valuded.
#     :param L: maximum degree of the spherical harmonics
#     :return:
#     """
#     # mtx = np.zeros((2 * (L + 1) ** 2, (L + 1) ** 2), dtype=float)
#     sz_non_pos = np.int((L + 1) * (L + 2) / 2.)
#     sz_pos = np.int((L + 1) * L / 2.)
#     mtx_real_neg = np.eye(sz_non_pos)
#     mtx_real_pos = np.zeros((sz_pos, sz_pos), dtype=float)
#     v_idx0 = 0
#     h_idx0 = sz_pos - L
#     for m in xrange(1, L + 1, 1):
#         vec_len = L + 1 - m
#         mtx_real_pos[v_idx0:v_idx0 + vec_len, h_idx0:h_idx0+vec_len] = np.eye(vec_len)
#         v_idx0 += vec_len
#         h_idx0 -= (vec_len - 1)
#     mtx_real = np.vstack((mtx_real_neg, np.hstack((mtx_real_pos, np.zeros((sz_pos, L + 1))))))
#     mtx_imag = np.vstack((mtx_real_neg[:, :sz_pos], -mtx_real_pos))
#     return mtx_real, mtx_imag


# def sph_2d_dirac_vertical_symb(G_col_ri, a_ri, K, L, noise_level,
#                                max_ini=100, stop_cri='mse'):
#     """
#     Dirac reconstruction on the sphere along the vertical direction.
#     Here we exploit the fact that the Diracs have real-valuded amplitudes. Hence b_nm is
#     Hermitian symmetric w.r.t. m.
#     Also we enforce the constraint that tk = cos theta_k is real-valued. Hence, the
#     annihilating filter coefficients (for co-latitudes) are real-valued.
#     Here we use explicitly the fact that the annihilating filter is real-valuded.
#     :param G_col: the linear transformation matrix that links the given spherical harmonics
#                     to the annihilable sequence.
#     :param a: the given spherical harmonics
#     :param K: number of Diracs
#     :param L: maximum degree of the spherical harmonics
#     :param M: maximum order of the spherical harmonics (M <= L)
#     :param noise_level: noise level present in the spherical harmonics
#     :param max_ini: maximum number of initialisations allowed for the algorithm
#     :param stop_cri: either 'mse' or 'maxiter'
#     :return:
#     """
#     num_visi_ri, num_bands = a_ri.shape
#     a_ri = a_ri.flatten('F')
#     compute_mse = (stop_cri == 'mse')
#
#     # size of G_col: (L + 1 + (2 * L + 1 - M) * M) x (L + 1 + (2 * L + 1 - M) * M)
#     GtG = np.dot(G_col_ri.T, G_col_ri)
#     Gt_a = np.dot(G_col_ri.T, a_ri)
#
#     # maximum number of iterations with each initialisation
#     max_iter = 50
#     min_error = float('inf')
#     beta_ri = np.reshape(linalg.lstsq(G_col_ri, a_ri)[0], (-1, num_bands), order='F')
#
#     # size of Tbeta: num_bands * (L + 1 - K)^2 x (K + 1)
#     Tbeta = sph_Tmtx_col_real_coef_symb(beta_ri, K, L)
#
#     # repetition matrix for Rmtx (the same annihilating filter for all sub-bands)
#     mtx_repe = np.eye(num_bands, dtype=float)
#
#     # size of various matrices / vectors
#     sz_G1 = (L + 1) ** 2 * num_bands
#
#     sz_Tb0 = (L + 1 - K) ** 2 * num_bands
#     sz_Tb1 = K + 1
#
#     sz_Rc0 = (L + 1 - K) ** 2 * num_bands
#     sz_Rc1 = (L + 1) ** 2 * num_bands
#
#     sz_coef = K + 1
#     sz_b = (L + 1) ** 2 * num_bands
#
#     rhs = np.concatenate((np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float),
#                           np.array(1, dtype=float)[np.newaxis]))
#     rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
#                                    np.zeros((sz_Rc0, 1), dtype=Gt_a.dtype)
#                                    ))
#                         )
#
#     # the main iteration with different random initialisations
#     for ini in xrange(max_ini):
#         c = np.random.randn(sz_coef, 1)  # we know that the coefficients are real-valuded
#         c0 = c.copy()
#         error_seq = np.zeros(max_iter)
#         R_loop = np.kron(mtx_repe, sph_Rmtx_col_real_coef_symb(c, K, L))
#         for inner in xrange(max_iter):
#             Mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta.T,
#                                              np.zeros((sz_coef, sz_Rc1)), c0
#                                              )),
#                                   np.hstack((Tbeta, np.zeros((sz_Tb0, sz_Tb0)),
#                                              -R_loop, np.zeros((sz_Rc0, 1))
#                                              )),
#                                   np.hstack((np.zeros((sz_Rc1, sz_Tb1)), -R_loop.T,
#                                              GtG, np.zeros((sz_Rc1, 1))
#                                              )),
#                                   np.hstack((c0.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
#                                              ))
#                                   ))
#             # matrix should be Hermitian symmetric
#             Mtx_loop = (Mtx_loop + Mtx_loop.T) / 2.
#             # c = linalg.solve(Mtx_loop, rhs)[:sz_coef]
#             c = linalg.lstsq(Mtx_loop, rhs)[0][:sz_coef]
#
#             R_loop = np.kron(mtx_repe, sph_Rmtx_col_real_coef_symb(c, K, L))
#             Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
#                                     np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
#                                     ))
#             # b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:sz_b]
#             b_recon = linalg.lstsq(Mtx_brecon, rhs_bl)[0][:sz_b]
#
#             error_seq[inner] = linalg.norm(a_ri - np.dot(G_col_ri, b_recon))
#             if error_seq[inner] < min_error:
#                 min_error = error_seq[inner]
#                 b_opt_col = b_recon
#                 c_opt_col = c
#
#             if min_error < noise_level and compute_mse:
#                 break
#
#         if min_error < noise_level and compute_mse:
#             break
#     return c_opt_col, min_error, b_opt_col, ini


# -----------------------------------------------
def sph_mtx_amp_inner(p_mic_x_loop, p_mic_y_loop, p_mic_z_loop, xk, yk, zk):
    """
    inner loop (for each sub-band) used for building the matrix for the least
    square estimation of Diracs' amplitudes
    :param p_mic_x_loop: a vector that contains the microphone x-coordinates (multiplied by mid-band frequency)
    :param p_mic_y_loop: a vector that contains the microphone y-coordinates (multiplied by mid-band frequency)
    :param p_mic_z_loop: a vector that contains the microphone z-coordinates (multiplied by mid-band frequency)
    :param xk: source location (x-coordinate)
    :param yk: source location (y-coordinate)
    :param zk: source location (z-coordinate)
    :return:
    """
    num_mic_per_band = p_mic_x_loop.size
    K = xk.size

    mtx_amp_blk = np.zeros(((num_mic_per_band - 1) * num_mic_per_band, K),
                           dtype=complex, order='C')
    count_inner = 0
    for q in xrange(num_mic_per_band):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        p_z_outer = p_mic_z_loop[q]
        for qp in xrange(num_mic_per_band):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x_loop[qp]
                p_y_qqp = p_y_outer - p_mic_y_loop[qp]
                p_z_qqp = p_z_outer - p_mic_z_loop[qp]
                mtx_amp_blk[count_inner, :] = np.exp(-1j * (xk * p_x_qqp +
                                                            yk * p_y_qqp +
                                                            zk * p_z_qqp))
                count_inner += 1
    return mtx_amp_blk


def sph_mtx_amp_inner_ri(p_mic_x_loop, p_mic_y_loop, p_mic_z_loop, xk, yk, zk):
    """
    inner loop (for each sub-band) used for building the matrix for the least
    square estimation of Diracs' amplitudes
    :param p_mic_x_loop: a vector that contains the microphone x-coordinates (multiplied by mid-band frequency)
    :param p_mic_y_loop: a vector that contains the microphone y-coordinates (multiplied by mid-band frequency)
    :param p_mic_z_loop: a vector that contains the microphone z-coordinates (multiplied by mid-band frequency)
    :param xk: source location (x-coordinate)
    :param yk: source location (y-coordinate)
    :param zk: source location (z-coordinate)
    :return:
    """
    num_mic_per_band = p_mic_x_loop.size
    K = xk.size

    mtx_amp_blk = np.zeros(((num_mic_per_band - 1) * num_mic_per_band, K),
                           dtype=complex, order='C')
    count_inner = 0
    for q in xrange(num_mic_per_band):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        p_z_outer = p_mic_z_loop[q]
        for qp in xrange(num_mic_per_band):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x_loop[qp]
                p_y_qqp = p_y_outer - p_mic_y_loop[qp]
                p_z_qqp = p_z_outer - p_mic_z_loop[qp]
                mtx_amp_blk[count_inner, :] = np.exp(-1j * (xk * p_x_qqp +
                                                            yk * p_y_qqp +
                                                            zk * p_z_qqp))
                count_inner += 1
    return cpx_mtx2real(mtx_amp_blk)


def sph_mtx_amp(theta_k, phi_k, p_mic_x, p_mic_y, p_mic_z):
    num_bands = p_mic_x.shape[1]

    xk, yk, zk = sph2cart(1, theta_k, phi_k)  # on S^2, -> r = 1
    # reshape xk, yk, zk to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')

    # parallel implementation
    partial_mtx_amp_mtx_inner = partial(sph_mtx_amp_inner, xk=xk, yk=yk, zk=zk)
    mtx_amp_lst = \
        Parallel(n_jobs=-1,
                 backend='threading')(delayed(partial_mtx_amp_mtx_inner)(p_mic_x[:, band_loop],
                                                                         p_mic_y[:, band_loop],
                                                                         p_mic_z[:, band_loop])
                                      for band_loop in xrange(num_bands))
    return linalg.block_diag(*mtx_amp_lst)


def sph_mtx_amp_ri(theta_k, phi_k, p_mic_x, p_mic_y, p_mic_z):
    num_bands = p_mic_x.shape[1]

    xk, yk, zk = sph2cart(1, theta_k, phi_k)  # on S^2, -> r = 1
    # reshape xk, yk, zk to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    zk = np.reshape(zk, (1, -1), order='F')

    # parallel implementation
    partial_mtx_amp_mtx_inner = partial(sph_mtx_amp_inner_ri, xk=xk, yk=yk, zk=zk)
    mtx_amp_lst = \
        Parallel(n_jobs=-1,
                 backend='threading')(delayed(partial_mtx_amp_mtx_inner)(p_mic_x[:, band_loop],
                                                                         p_mic_y[:, band_loop],
                                                                         p_mic_z[:, band_loop])
                                      for band_loop in xrange(num_bands))
    return linalg.block_diag(*mtx_amp_lst)


# ===== functions to build R and T matrices =====
def sph_Rmtx_row_major_ri(coef_ri, K, L):
    """
    the right dual matrix associated with the FRI sequence b_nm.
    :param coef_ri: annihilating filter coefficinets represented in terms of
                    real and imaginary parts, i.e., a REAL-VALUED vector!
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    num_real = K + 1
    coef_real = coef_ri[:num_real]
    coef_imag = coef_ri[num_real:]
    R_real = sph_Rmtx_row_major_no_extend(coef_real, K, L)
    R_imag = sph_Rmtx_row_major_no_extend(coef_imag, K, L)
    expand_mtx_real = sph_build_exp_mtx(L, 'real')
    expand_mtx_imag = sph_build_exp_mtx(L, 'imaginary')

    R = np.vstack((np.hstack((np.dot(R_real, expand_mtx_real),
                              -np.dot(R_imag, expand_mtx_imag))),
                   np.hstack((np.dot(R_imag, expand_mtx_real),
                              np.dot(R_real, expand_mtx_imag)))
                   ))
    return R


def sph_Rmtx_row_major_no_extend(coef, K, L):
    """
    the right dual matrix associated with the FRI sequence \tilde{b}.
    :param coef: annihilating filter coefficinets
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    # K = coef.size - 1
    coef = np.squeeze(coef)
    assert K <= L + 1
    Rmtx_half = np.zeros((np.int((L - K + 2) * (L - K + 1) / 2.),
                          np.int((L + K + 2) * (L - K + 1) / 2.)),
                         dtype=coef.dtype)
    v_bg_idx = 0
    h_bg_idx = 0
    for n in xrange(0, L - K + 1):
        # input singal length
        vec_len = L + 1 - n

        blk_size0 = vec_len - K
        if blk_size0 == 1:
            col = coef[-1]
            row = coef[::-1]
        else:
            col = np.concatenate((np.array([coef[-1]]), np.zeros(vec_len - K - 1)))
            row = np.concatenate((coef[::-1], np.zeros(vec_len - K - 1)))

        Rmtx_half[v_bg_idx:v_bg_idx + blk_size0,
        h_bg_idx:h_bg_idx + vec_len] = linalg.toeplitz(col, row)

        v_bg_idx += blk_size0
        h_bg_idx += vec_len
    # only part of the data is longer enough to be annihilated
    # extract that part, which is equiavalent to padding zero columns on both ends
    # expand the data by repeating the data with m = 0 (used for annihilations on both sides)
    mtx_col2row = trans_col2row_pos_ms(L)[:np.int((L + K + 2) * (L - K + 1) / 2.), :]
    return np.kron(np.eye(2), np.dot(Rmtx_half, mtx_col2row))


def sph_Tmtx_row_major_ri(b_ri, K, L):
    """
    The REAL-VALUED data b_ri is assumed to be arranged as a column vector.
    The first half (to be exact, the first (L+1)(L+2)/2 elements) is the real
    part of the FRI sequence; while the second half is the imaginary part of
    the FRI sequence.

    :param b: the annihilable sequence
    :param K: the annihilating filter is of size K + 1
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    num_bands = b_ri.shape[1]
    num_real = (L + 1) ** 2
    blk_sz0 = 2 * (L + 1 - K) * (L - K + 2)
    T = np.zeros((blk_sz0 * num_bands, 2 * (K + 1)), dtype=float)
    idx0 = 0
    for loop in xrange(num_bands):
        b_real = b_ri[:num_real, loop]
        b_imag = b_ri[num_real:, loop]
        T_real = sph_Tmtx_row_major(b_real, K, L, 'real')
        T_imag = sph_Tmtx_row_major(b_imag, K, L, 'imaginary')
        T[idx0:idx0 + blk_sz0, :] = np.vstack((np.hstack((T_real, -T_imag)),
                                               np.hstack((T_imag, T_real))
                                               ))
        idx0 += blk_sz0
    return T


def sph_Tmtx_row_major(b, K, L, option='real'):
    """
    The data b is assumed to be arranged as a column vector
    :param b: the annihilable sequence
    :param K: the annihilating filter is of size K + 1
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    Tmtx_half_pos = np.zeros((np.int((L - K + 2) * (L - K + 1) / 2.),
                              K + 1),
                             dtype=b.dtype)
    Tmtx_half_neg = np.zeros((np.int((L - K + 2) * (L - K + 1) / 2.),
                              K + 1),
                             dtype=b.dtype)
    # exapand the sequence b and rearrange row by row
    expand_mtx = sph_build_exp_mtx(L, option)
    mtx_col2row = trans_col2row_pos_ms(L)
    b_pn = np.dot(expand_mtx, b)
    sz_b_pn = b_pn.size
    assert sz_b_pn % 2 == 0
    b_pos = np.dot(mtx_col2row, b_pn[:np.int(sz_b_pn / 2.)])
    b_neg = np.dot(mtx_col2row, b_pn[np.int(sz_b_pn / 2.):])
    data_bg_idx = 0
    blk_bg_idx = 0
    for n in xrange(0, L - K + 1):
        vec_len = L + 1 - n
        blk_size0 = vec_len - K
        vec_pos_loop = b_pos[data_bg_idx:data_bg_idx + vec_len]
        vec_neg_loop = b_neg[data_bg_idx:data_bg_idx + vec_len]

        Tmtx_half_pos[blk_bg_idx:blk_bg_idx + blk_size0, :] = \
            linalg.toeplitz(vec_pos_loop[K::], vec_pos_loop[K::-1])
        Tmtx_half_neg[blk_bg_idx:blk_bg_idx + blk_size0, :] = \
            linalg.toeplitz(vec_neg_loop[K::], vec_neg_loop[K::-1])

        data_bg_idx += vec_len
        blk_bg_idx += blk_size0
    return np.vstack((Tmtx_half_pos, Tmtx_half_neg))


# -----------------------------------------------
# def sph_Rmtx_col_real_coef_symb(coef, K, L):
#     """
#     the right dual matrix associated with the FRI sequence \tilde{b}.
#     Here we use explicitly the fact that the coeffiients are real.
#     Additionally, b_tilde is Hermitian symmetric.
#     :param coef: annihilating filter coefficinets
#     :param L: maximum degree of the spherical harmonics
#     :return:
#     """
#     coef = np.squeeze(coef)
#     assert K <= L + 1
#     R_sz0 = (L + 1 - K) ** 2
#
#     Rr_sz0 = np.int((L - K + 1) * (L - K + 2) / 2.)
#     Rr_sz1 = np.int((L -K + 1) * (L + K + 2) / 2.)
#
#     Ri_sz0 = np.int((L - K) * (L - K + 1) / 2.)
#     Ri_sz1 = np.int((L - K) * (L + K + 1) / 2.)
#
#     Rr = np.zeros((Rr_sz0, Rr_sz1), dtype=float)
#     Ri = np.zeros((Ri_sz0, Ri_sz1), dtype=float)
#
#     v_bg_idx = 0
#     h_bg_idx = 0
#
#     for m in xrange(-L + K, 1):
#         vec_len = L + 1 - np.abs(m)
#         blk_size0 = vec_len - K
#         if blk_size0 == 1:
#             col = coef[-1]
#             row = coef[::-1]
#         else:
#             col = np.concatenate((np.array([coef[-1]]), np.zeros(vec_len - K - 1)))
#             row = np.concatenate((coef[::-1], np.zeros(vec_len - K - 1)))
#
#         mtx_blk = linalg.toeplitz(col, row)
#         Rr[v_bg_idx:v_bg_idx + blk_size0, h_bg_idx:h_bg_idx + vec_len] = mtx_blk
#         if not m == 0:
#             Ri[v_bg_idx:v_bg_idx + blk_size0, h_bg_idx:h_bg_idx + vec_len] = mtx_blk
#
#     # only part of the data is longer enough to be annihilated
#     # extract that part, which is equiavalent to padding zero columns on both ends
#     extract_mtx_r = np.eye(Rr_sz1, np.int((L + 1) * (L + 2) / 2.),
#                            np.int(K * (K + 1) / 2.))
#     extract_mtx_i = np.eye(Ri_sz1, np.int(L * (L + 1) / 2.),
#                            np.int(K * (K + 1) / 2.))
#     return linalg.block_diag(np.dot(Rr, extract_mtx_r), np.dot(Ri, extract_mtx_i))


def sph_Rmtx_col_major_real_coef(coef, K, L):
    """
    the right dual matrix associated with the FRI sequence \tilde{b}.
    Here we use explicitly the fact that the coefficients are real.
    :param coef: annihilating filter coefficinets
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    return np.kron(np.eye(2), sph_Rmtx_col_major(coef, K, L))


def sph_Rmtx_col_major(coef, K, L):
    """
    the right dual matrix associated with the FRI sequence \tilde{b}.
    :param coef: annihilating filter coefficinets
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    # K = coef.size - 1
    coef = np.squeeze(coef)
    assert K <= L + 1
    Rmtx = np.zeros(((L + 1 - K) ** 2,
                     (K + 1 + L) * (L - K) + L + 1),
                    dtype=coef.dtype)
    v_bg_idx = 0
    h_bg_idx = 0
    for m in xrange(-L + K, L - K + 1):
        # input singal length
        vec_len = L + 1 - np.abs(m)

        blk_size0 = vec_len - K
        if blk_size0 == 1:
            col = coef[-1]
            row = coef[::-1]
        else:
            col = np.concatenate((np.array([coef[-1]]), np.zeros(vec_len - K - 1)))
            row = np.concatenate((coef[::-1], np.zeros(vec_len - K - 1)))

        Rmtx[v_bg_idx:v_bg_idx + blk_size0, h_bg_idx:h_bg_idx + vec_len] = \
            linalg.toeplitz(col, row)

        v_bg_idx += blk_size0
        h_bg_idx += vec_len
    # only part of the data is longer enough to be annihilated
    # extract that part, which is equiavalent to padding zero columns on both ends
    extract_mtx = np.eye((K + 1 + L) * (L - K) + L + 1, (L + 1) ** 2, np.int((K + 1) * K / 2.))
    return np.dot(Rmtx, extract_mtx)


def sph_Tmtx_col_real_coef_symb(b_tilde_ri, K, L):
    """
    build Toeplitz matrix from the given data b_tilde.
    Here we also exploit the fact that b_tilde is Hermitian symmetric.
    :param b_tilde: the given data in matrix form with size: (L + 1)(2L + 1)
    :param K: the filter size is K + 1
    :return:
    """
    num_bands = b_tilde_ri.shape[1]
    num_real = np.int((L + 1) * (L + 2) / 2.)
    Tblk_sz0 = (L + 1 - K) ** 2
    Tr_sz0 = np.int((L - K + 2) * (L - K + 1) / 2.)
    Ti_sz0 = np.int((L - K) * (L - K + 1) / 2.)
    T = np.zeros((Tblk_sz0 * num_bands, K + 1), dtype=float)
    idx0 = 0
    for loop in xrange(num_bands):
        b_tilde_r = b_tilde_ri[:num_real, loop]
        b_tilde_i = b_tilde_ri[num_real:, loop]
        Tr = np.zeros((Tr_sz0, K + 1), dtype=float)
        Ti = np.zeros((Ti_sz0, K + 1), dtype=float)
        v_bg_idx = 0
        data_bg_idx = np.int(K * (K + 1) / 2.)
        for m in xrange(-L + K, 1):
            # input singal length
            vec_len = L + 1 - np.abs(m)
            blk_size0 = vec_len - K
            vec_r_loop = b_tilde_r[data_bg_idx:data_bg_idx + vec_len]
            Tr[v_bg_idx:v_bg_idx + blk_size0, :] = \
                linalg.toeplitz(vec_r_loop[K::], vec_r_loop[K::-1])

            if not m == 0:
                vec_i_loop = b_tilde_i[data_bg_idx:data_bg_idx + vec_len]
                Ti[v_bg_idx:v_bg_idx + blk_size0, :] = \
                    linalg.toeplitz(vec_i_loop[K::], vec_i_loop[K::-1])

            v_bg_idx += blk_size0

        T[idx0:idx0 + Tblk_sz0] = np.vstack((Tr, Ti))
        idx0 += Tblk_sz0
    return T


def sph_Tmtx_col_major_real_coef(b_tilde_ri, K, L):
    """
    build Toeplitz matrix from the given data b_tilde.
    :param b_tilde: the given data in matrix form with size: (L + 1)(2L + 1)
    :param K: the filter size is K + 1
    :return:
    """
    num_bands = b_tilde_ri.shape[1]
    num_real = (L + 1) ** 2
    Tblk_sz0 = 2 * (L + 1 - K) ** 2
    T = np.zeros((Tblk_sz0 * num_bands, K + 1), dtype=float)
    idx0 = 0
    for loop in xrange(num_bands):
        b_tilde_r = up_tri_from_vec(b_tilde_ri[:num_real, loop])
        b_tilde_i = up_tri_from_vec(b_tilde_ri[num_real:, loop])
        Tr = np.zeros(((L + 1 - K) ** 2, K + 1), dtype=float)
        Ti = np.zeros(((L + 1 - K) ** 2, K + 1), dtype=float)
        v_bg_idx = 0
        for m in xrange(-L + K, L - K + 1):
            # input singal length
            vec_len = L + 1 - np.abs(m)
            blk_size0 = vec_len - K
            vec_r_loop = b_tilde_r[:vec_len, m + L]
            vec_i_loop = b_tilde_i[:vec_len, m + L]

            Tr[v_bg_idx:v_bg_idx + blk_size0, :] = \
                linalg.toeplitz(vec_r_loop[K::], vec_r_loop[K::-1])
            Ti[v_bg_idx:v_bg_idx + blk_size0, :] = \
                linalg.toeplitz(vec_i_loop[K::], vec_i_loop[K::-1])

            v_bg_idx += blk_size0

        T[idx0:idx0 + Tblk_sz0] = np.vstack((Tr, Ti))
        idx0 += Tblk_sz0
    return T


# ========= utilities for building G matrix ===========
def sph_mtx_updated_G_col_major_ri(theta_recon, phi_recon, L, p_mic_x,
                                   p_mic_y, p_mic_z, mtx_freq2visibility):
    """
    update the linear transformation matrix that links the FRI sequence to the visibilities.
    Here the input vector to this linear transformation is FRI seuqnce that is re-arranged
    column by column.
    :param theta_recon: reconstructed co-latitude(s)
    :param phi_recon: reconstructed azimuth(s)
    :param L: degree of the spherical harmonics
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    num_bands = p_mic_x.shape[1]
    # the spherical harmonics basis evaluated at the reconstructed Dirac locations
    m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                 np.arange(0, L + 1, step=1, dtype=int))
    m_grid = vec_from_low_tri_col_by_col(m_grid)[:, np.newaxis]
    l_grid = vec_from_low_tri_col_by_col(l_grid)[:, np.newaxis]
    # reshape theta_recon and phi_recon to use broadcasting
    theta_recon = np.reshape(theta_recon, (1, -1), order='F')
    phi_recon = np.reshape(phi_recon, (1, -1), order='F')
    mtx_Ylm_ri = cpx_mtx2real(np.conj(sph_harm_ufnc(l_grid, m_grid, theta_recon, phi_recon)))
    # the mapping from FRI sequence to Diracs amplitude
    mtx_fri2freq_ri = cpx_mtx2real(sph_mtx_fri2freq_col_major(L))
    mtx_Tinv_Y_ri = np.kron(np.eye(num_bands),
                            linalg.solve(mtx_fri2freq_ri, mtx_Ylm_ri))
    # G1
    # mtx_fri2amp_ri = linalg.solve(np.dot(mtx_Tinv_Y_ri.T,
    #                                      mtx_Tinv_Y_ri),
    #                               mtx_Tinv_Y_ri.T)
    mtx_fri2amp_ri = linalg.lstsq(mtx_Tinv_Y_ri, np.eye(mtx_Tinv_Y_ri.shape[0]))[0]
    # the mapping from Diracs amplitude to visibilities
    mtx_amp2visibility_ri = \
        sph_mtx_amp_ri(theta_recon, phi_recon, p_mic_x, p_mic_y, p_mic_z)
    # the mapping from FRI sequence to visibilities (G0)
    mtx_fri2visibility_ri = linalg.block_diag(*[np.dot(cpx_mtx2real(mtx_freq2visibility[G_blk_count]),
                                                       mtx_fri2freq_ri)
                                                for G_blk_count in xrange(num_bands)])
    # projection to the null space of mtx_fri2amp
    # mtx_null_proj = np.eye(2 * num_bands * (L + 1) ** 2) - \
    #                 np.dot(mtx_fri2amp_ri.T,
    #                        linalg.solve(np.dot(mtx_fri2amp_ri,
    #                                            mtx_fri2amp_ri.T),
    #                                     mtx_fri2amp_ri))
    mtx_null_proj = np.eye(2 * num_bands * (L + 1) ** 2) - \
                    np.dot(mtx_fri2amp_ri.T,
                           linalg.lstsq(mtx_fri2amp_ri.T,
                                        np.eye(mtx_fri2amp_ri.shape[1]))[0])
    G_updated = np.dot(mtx_amp2visibility_ri, mtx_fri2amp_ri) + np.dot(mtx_fri2visibility_ri, mtx_null_proj)
    return G_updated


def sph_mtx_updated_G_row_major(theta_recon, phi_recon, L, p_mic_x,
                                p_mic_y, p_mic_z, mtx_freq2visibility):
    """
    update the linear transformation matrix that links the FRI sequence to the visibilities.
    Here the input vector to this linear transformation is FRI seuqnce that is re-arranged
    row by row.
    :param theta_recon: reconstructed co-latitude(s)
    :param phi_recon: reconstructed azimuth(s)
    :param L: degree of the spherical harmonics
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    num_bands = p_mic_x.shape[1]
    # mtx_fri2freq_ri links the FRI sequence to spherical harmonics of degree L
    mtx_fri2freq_ri = sph_mtx_fri2freq_row_major(L)
    # -------------------------------------------------
    # the spherical harmonics basis evaluated at the reconstructed Dirac locations
    m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                 np.arange(0, L + 1, step=1, dtype=int))
    m_grid = vec_from_low_tri_col_by_col(m_grid)[:, np.newaxis]
    l_grid = vec_from_low_tri_col_by_col(l_grid)[:, np.newaxis]
    # reshape theta_recon and phi_recon to use broadcasting
    theta_recon = np.reshape(theta_recon, (1, -1), order='F')
    phi_recon = np.reshape(phi_recon, (1, -1), order='F')
    mtx_Ylm_ri = cpx_mtx2real(np.conj(sph_harm_ufnc(l_grid, m_grid, theta_recon, phi_recon)))
    mtx_Tinv_Y_ri = np.kron(np.eye(num_bands),
                            linalg.solve(mtx_fri2freq_ri, mtx_Ylm_ri))
    # G1
    # mtx_fri2amp_ri = linalg.solve(np.dot(mtx_Tinv_Y_ri.T, mtx_Tinv_Y_ri),
    #                               mtx_Tinv_Y_ri.T)
    mtx_fri2amp_ri = linalg.lstsq(mtx_Tinv_Y_ri, np.eye(mtx_Tinv_Y_ri.shape[0]))[0]
    # -------------------------------------------------
    # the mapping from Diracs amplitude to visibilities (G2)
    mtx_amp2visibility_ri = \
        sph_mtx_amp_ri(theta_recon, phi_recon, p_mic_x, p_mic_y, p_mic_z)
    # -------------------------------------------------
    # the mapping from FRI sequence to visibilities (G0)
    mtx_fri2visibility_ri = linalg.block_diag(*[np.dot(cpx_mtx2real(mtx_freq2visibility[G_blk_count]),
                                                       mtx_fri2freq_ri)
                                                for G_blk_count in xrange(num_bands)])
    # -------------------------------------------------
    # projection to the null space of mtx_fri2amp
    # mtx_null_proj = np.eye(num_bands * 2 * (L + 1) ** 2) - \
    #                 np.dot(mtx_fri2amp_ri.T,
    #                        linalg.solve(np.dot(mtx_fri2amp_ri,
    #                                            mtx_fri2amp_ri.T),
    #                                     mtx_fri2amp_ri))
    mtx_null_proj = np.eye(num_bands * 2 * (L + 1) ** 2) - \
                    np.dot(mtx_fri2amp_ri.T,
                           linalg.lstsq(mtx_fri2amp_ri.T,
                                        np.eye(mtx_fri2amp_ri.shape[1]))[0])
    G_updated = np.dot(mtx_amp2visibility_ri, mtx_fri2amp_ri) + \
                np.dot(mtx_fri2visibility_ri, mtx_null_proj)
    return G_updated


def sph_mtx_freq2visibility(L, p_mic_x, p_mic_y, p_mic_z):
    """
    build the linear mapping matrix from the spherical harmonics to the measured visibility.
    :param L: the maximum degree of spherical harmonics
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' y coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' z coordinates
    :return:
    """
    num_mic_per_band, num_bands = p_mic_x.shape
    m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                 np.arange(0, L + 1, step=1, dtype=int))
    m_grid = np.reshape(vec_from_low_tri_col_by_col(m_grid), (1, -1), order='F')
    l_grid = np.reshape(vec_from_low_tri_col_by_col(l_grid), (1, -1), order='F')
    # parallel implemnetation (over all the subands at once)
    partial_sph_mtx_inner = partial(sph_mtx_freq2visibility_inner, L=L, m_grid=m_grid, l_grid=l_grid)
    G_lst = Parallel(n_jobs=-1, backend='threading')(delayed(partial_sph_mtx_inner)(p_mic_x[:, band_loop],
                                                                                    p_mic_y[:, band_loop],
                                                                                    p_mic_z[:, band_loop])
                                                     for band_loop in xrange(num_bands))
    return G_lst


def sph_mtx_freq2visibility_inner(p_mic_x_loop, p_mic_y_loop, p_mic_z_loop, L, m_grid, l_grid):
    """
    inner loop for the function sph_mtx_freq2visibility
    :param band_loop: the number of the sub-bands considered
    :param p_mic_x_loop: a vector that contains the microphone x-coordinates (multiplied by mid-band frequency)
    :param p_mic_y_loop: a vector that contains the microphone y-coordinates (multiplied by mid-band frequency)
    :param p_mic_z_loop: a vector that contains the microphone z-coordinates (multiplied by mid-band frequency)
    :param L: maximum degree of the spherical harmonics
    :param m_grid: m indices for evaluating the spherical bessel functions and spherical harmonics
    :param l_grid: l indices for evaluating the spherical bessel functions and spherical harmonics
    :return:
    """
    num_mic_per_band = p_mic_x_loop.size
    ells = np.arange(L + 1, dtype=float)[np.newaxis, :]
    G_blk = np.zeros(((num_mic_per_band - 1) * num_mic_per_band, (L + 1) ** 2),
                     dtype=complex, order='C')
    count_G_band = 0
    for q in xrange(num_mic_per_band):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        p_z_outer = p_mic_z_loop[q]
        for qp in xrange(num_mic_per_band):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x_loop[qp]
                p_y_qqp = p_y_outer - p_mic_y_loop[qp]
                p_z_qqp = p_z_outer - p_mic_z_loop[qp]
                norm_p_qqp = np.sqrt(p_x_qqp ** 2 + p_y_qqp ** 2 + p_z_qqp ** 2)
                # compute bessel function up to order L
                sph_bessel_ells = (-1j) ** ells * sp.special.jv(ells + 0.5, norm_p_qqp) / np.sqrt(norm_p_qqp)

                theta_qqp = np.arccos(p_z_qqp / norm_p_qqp)
                phi_qqp = np.arctan2(p_y_qqp, p_x_qqp)
                # here l_grid and m_grid is assumed to be a row vector
                G_blk[count_G_band, :] = sph_bessel_ells[0, l_grid.squeeze()] * \
                                         sph_harm_ufnc(l_grid, m_grid, theta_qqp, phi_qqp)
                count_G_band += 1
    return G_blk * (2 * np.pi) ** 1.5


def cpx_mtx2real(mtx):
    """
    extend complex valued matrix to an extended matrix of real values only
    :param mtx: input complex valued matrix
    :return:
    """
    return np.vstack((np.hstack((mtx.real, -mtx.imag)), np.hstack((mtx.imag, mtx.real))))


def sph_mtx_fri2visibility_col_major(L, mtx_freq2visibility):
    """
    build the linear transformation matrix that links the FRI sequence, which is
    arranged column by column, with the measured visibilities.
    :param L: maximum degree of spherical harmonics
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    mtx_fri2freq = sph_mtx_fri2freq_col_major(L)
    return linalg.block_diag(*[np.dot(mtx_freq2visibility[G_blk_count], mtx_fri2freq)
                               for G_blk_count in xrange(len(mtx_freq2visibility))])


def sph_mtx_fri2visibility_col_major_ri(L, mtx_freq2visibility):
    """
    build the linear transformation matrix that links the FRI sequence, which is
    arranged column by column, with the measured visibilities.
    :param L: maximum degree of spherical harmonics
    :param mtx_freq2visibility: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    mtx_fri2freq = sph_mtx_fri2freq_col_major(L)
    return linalg.block_diag(*[cpx_mtx2real(np.dot(mtx_freq2visibility[G_blk_count],
                                                   mtx_fri2freq))
                               for G_blk_count in xrange(len(mtx_freq2visibility))])


def sph_mtx_fri2freq_col_major(L):
    """
    build the linear transformation for the case of Diracs on the sphere.
    The spherical harmonics are arranged as a matrix of size:
        (L + 1) x (2L + 1)
    :param L: maximum degree of the available spherical harmonics
    :return:
    """
    G = np.zeros(((L + 1) ** 2, (L + 1) ** 2), dtype=float)
    count = 0
    horizontal_idx = 0
    for m in xrange(-L, L + 1):
        abs_m = np.abs(m)
        for l in xrange(abs_m, L + 1):
            Nlm = (-1) ** ((m + abs_m) / 2.) * \
                  np.sqrt((2. * l + 1.) / (4. * np.pi) *
                          (sp.misc.factorial(l - abs_m) / sp.misc.factorial(l + abs_m))
                          )
            G[count, horizontal_idx:horizontal_idx + l - abs_m + 1] = \
                (-1) ** abs_m * Nlm * compute_Pmn_coef(l, m)
            count += 1
        horizontal_idx += L - abs_m + 1
    return G


def sph_mtx_fri2visibility_row_major(L, mtx_freq2visibility_cpx):
    """
    build the linear transformation matrix that links the FRI sequence, which is
    arranged column by column, with the measured visibilities.
    :param L: maximum degree of spherical harmonics
    :param mtx_freq2visibility_cpx: the linear transformation matrix that links the
                spherical harmonics with the measured visibility
    :return:
    """
    # mtx_freq2visibility_cpx
    # matrix with input: the complex-valued spherical harmonics and
    # output: the measured visibilities
    mtx_fri2freq = sph_mtx_fri2freq_row_major(L)
    # -----------------------------------------------------------------
    # cascade the linear mappings
    return linalg.block_diag(*[np.dot(cpx_mtx2real(mtx_freq2visibility_cpx[G_blk_count]),
                                      mtx_fri2freq)
                               for G_blk_count in xrange(len(mtx_freq2visibility_cpx))])


def sph_mtx_fri2freq_row_major(L):
    # -----------------------------------------------------------------
    # the matrix that connects the rearranged spherical harmonics
    # (complex conjugation and reverse indexing for m < 0) to the normal spherical
    # harmonics.
    # the input to this matrix is a concatenation of the real-part and the imaginary
    # part of the rearranged spherical harmonics.
    # size of this matrix: 2(L+1)^2 x 2(L+1)^2
    sz_non_zero_idx = np.int(L * (L + 1) / 2.)
    sz_half_ms = np.int((L + 1) * (L + 2) / 2.)
    # the building matrix for the mapping matrix (negtive m-s)
    mtx_neg_ms_map = np.zeros((sz_non_zero_idx, sz_non_zero_idx), dtype=float)
    idx0 = 0
    for m in xrange(-L, 0):  # m ranges from -L to -1
        blk_sz = L + m + 1
        mtx_neg_ms_map[idx0:idx0 + blk_sz,
        sz_non_zero_idx - idx0 - blk_sz:sz_non_zero_idx - idx0] = \
            np.eye(blk_sz)
        idx0 += blk_sz
    # the building matrix for the mapping matrix (positive m-s)
    mtx_pos_ms_map = np.eye(sz_half_ms)

    mtx_pn_map_real = np.vstack((np.hstack((np.zeros((sz_non_zero_idx, sz_half_ms)),
                                            mtx_neg_ms_map)),
                                 np.hstack((mtx_pos_ms_map,
                                            np.zeros((sz_half_ms, sz_non_zero_idx))))))
    # CAUTION: the negative sign in the first block
    # the rearranged spherical harmonics with negative m-s are complex CONJUGATED.
    # Hence, these components (which corresponds to the imaginary part of the spherical harmonics)
    # are the negative of the original spherical harmonics.
    mtx_pn_map_imag = np.vstack((np.hstack((np.zeros((sz_non_zero_idx, sz_half_ms)),
                                            -mtx_neg_ms_map)),
                                 np.hstack((mtx_pos_ms_map,
                                            np.zeros((sz_half_ms, sz_non_zero_idx))))))

    mtx_freqRemap2freq = linalg.block_diag(mtx_pn_map_real, mtx_pn_map_imag)
    # -----------------------------------------------------------------
    # the matrix that connects the FRI sequence b_mn to the remapped spherical harmonics
    # size of this matrix: 2(L+1)^2 x 2(L+1)^2
    mtx_fri2freqRemap_pos = sph_buildG_positive_ms(L, L)
    mtx_fri2freqRemap_neg = sph_buildG_strict_negative_ms(L, L)
    mtx_fri2freqRemap_pn_cpx = \
        np.vstack((np.hstack((mtx_fri2freqRemap_pos,
                              np.zeros((sz_half_ms, sz_non_zero_idx)))),
                   np.hstack((np.zeros((sz_non_zero_idx, sz_half_ms)),
                              mtx_fri2freqRemap_neg))
                   ))
    # now extend the matrix with real-valued components only
    mtx_fri2freqRemap_pn_real = np.vstack((np.hstack((np.real(mtx_fri2freqRemap_pn_cpx),
                                                      -np.imag(mtx_fri2freqRemap_pn_cpx))),
                                           np.hstack((np.imag(mtx_fri2freqRemap_pn_cpx),
                                                      np.real(mtx_fri2freqRemap_pn_cpx)))
                                           ))
    return np.dot(mtx_freqRemap2freq, mtx_fri2freqRemap_pn_real)


def sph_buildG_positive_ms(L, M):
    """
    build the linear transformation for the case of Diracs on the sphere.
    The spherical harmonics are arranged as a matrix of size:
        (L + 1) x (M + 1)
    :param L: maximum degree of the available spherical harmonics
    :param M: maximum order of the available spherical harmonics: 0 to M
    :return:
    """
    assert M <= L
    sz_G = np.int((2 * L + 2 - M) * (M + 1) / 2)
    G = np.zeros((sz_G, sz_G), dtype=float)
    count = 0
    horizontal_idx = 0
    for m in xrange(0, M + 1):
        for l in xrange(m, L + 1):
            Nlm = (-1) ** m * \
                  np.sqrt((2. * l + 1.) / (4. * np.pi) *
                          (sp.misc.factorial(l - m) / sp.misc.factorial(l + m))
                          )
            G[count, horizontal_idx:horizontal_idx + l - m + 1] = \
                (-1) ** m * Nlm * compute_Pmn_coef(l, m)
            count += 1
        horizontal_idx += L - m + 1
    return G


def sph_buildG_strict_negative_ms(L, M):
    """
    build the linear transformation for the case of Diracs on the sphere.

    The only difference with sph_buildG_positive_ms is the normalisation factor Nlm.

    :param L: maximum degree of the available spherical harmonics
    :param M: maximum order of the available spherical harmonics: -1 to -M
            (NOTICE: the reverse ordering!!)
    :return:
    """
    assert M <= L
    sz_G = np.int((2 * L + 2 - M) * (M + 1) / 2 - (L + 1))
    G = np.zeros((sz_G, sz_G), dtype=float)
    count = 0
    horizontal_idx = 0
    for m in xrange(1, M + 1):
        for l in xrange(m, L + 1):
            Nlm = np.sqrt((2. * l + 1.) / (4. * np.pi) *
                          (sp.misc.factorial(l - m) / sp.misc.factorial(l + m))
                          )
            G[count, horizontal_idx:horizontal_idx + l - m + 1] = \
                (-1) ** m * Nlm * compute_Pmn_coef(l, m)
            count += 1
        horizontal_idx += L - m + 1
    return G


# =========== utilities for spherical harmonics ============
def sph2cart(r, theta, phi):
    """
    spherical to cartesian coordinates
    :param r: radius
    :param theta: co-latitude
    :param phi: azimuth
    :return:
    """
    r_sin_theta = r * np.sin(theta)
    x = r_sin_theta * np.cos(phi)
    y = r_sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def trans_col2row_pos_ms(L):
    """
    transformation matrix to permute a column by column rearranged vector of an
    upper triangle to a row by row rearranged vector.
    :param L: maximum degree of the spherical harmonics.
    :return:
    """
    sz = np.int((L + 1) * (L + 2) / 2.)
    perm_idx = vec_from_up_tri_row_by_row_pos_ms(
        up_tri_from_vec_pos_ms(np.arange(0, sz, dtype=int)))
    return np.eye(sz, dtype=int)[perm_idx, :]


def vec_from_up_tri_row_by_row_pos_ms(mtx):
    """
    extract the data from the matrix, which is of the form:

        x x x x
        x x x 0
        x x 0 0
        x 0 0 0

    row by row.

    The function "extract_vec_from_up_tri" extracts data column by column.

    :param mtx: the matrix to extract the data from
    :return: data vector
    """
    L = mtx.shape[0] - 1
    assert mtx.shape[1] == L + 1
    vec_len = np.int((L + 2) * (L + 1) / 2)
    vec = np.zeros(vec_len, dtype=mtx.dtype)
    bg_idx = 0
    for l in xrange(L + 1):
        vec_len_loop = L - l + 1
        vec[bg_idx:bg_idx + vec_len_loop] = mtx[l, :vec_len_loop]
        bg_idx += vec_len_loop
    return vec


def up_tri_from_vec_pos_ms(vec):
    """
    build a matrix of the form

        x x x x
        x x x 0     (*)
        x x 0 0
        x 0 0 0

    from the given data matrix. This function is the inverse operator
    of the function extract_vec_from_up_tri(mtx).
    :param vec: the data vector
    :return: a matrix of the form (*)
    """
    L = np.int((-3 + np.sqrt(1 + 8 * vec.size)) / 2.)
    assert (L + 1) * (L + 2) == 2 * vec.size
    mtx = np.zeros((L + 1, L + 1), dtype=vec.dtype)
    bg_idx = 0
    for loop in xrange(0, L + 1):
        vec_len_loop = L + 1 - loop
        mtx[0:vec_len_loop, loop] = vec[bg_idx:bg_idx + vec_len_loop]
        bg_idx += vec_len_loop
    return mtx


def sph_build_exp_mtx(L, option='real'):
    """
    build the expansion matrix such that the data matrix which includes the data for
    spherical harmonics with positive orders (m>=0) and complex conjugated ones with
    strictly NEGATIVE ordres (m<0).
    :param L: maximum degree of the spherical harmonics
    :return:
    """
    sz_id_mtx_pos = np.int((L + 1) * (L + 2) / 2.)
    sz_id_mtx_neg = np.int((L + 1) * L / 2.)
    expand_mtx = np.zeros((L + 1 + sz_id_mtx_pos + sz_id_mtx_neg,
                           sz_id_mtx_pos + sz_id_mtx_neg))
    expand_mtx[0:sz_id_mtx_pos, 0:sz_id_mtx_pos] = np.eye(sz_id_mtx_pos)
    if option == 'real':
        expand_mtx[sz_id_mtx_pos:sz_id_mtx_pos + L + 1, 0:L + 1] = np.eye(L + 1)
    else:
        expand_mtx[sz_id_mtx_pos:sz_id_mtx_pos + L + 1, 0:L + 1] = -np.eye(L + 1)
    expand_mtx[sz_id_mtx_pos + L + 1:, sz_id_mtx_pos:] = np.eye(sz_id_mtx_neg)
    return expand_mtx


def up_tri_from_vec(vec):
    """
    build a matrix of the form

        x x x x x x x
        0 x x x x x 0       (*)
        0 0 x x x 0 0
        0 0 0 x 0 0 0

    from the given data matrix. This function is the inverse operator
    of the function vec_from_up_tri_positive_ms(mtx).
    :param vec: the data vector
    :return: a matrix of the form (*)
    """
    L = np.int(np.sqrt(vec.size)) - 1
    assert (L + 1) ** 2 == vec.size
    mtx = np.zeros((L + 1, 2 * L + 1), dtype=vec.dtype)
    bg_idx = 0
    for loop in xrange(2 * L + 1):
        if loop <= L:
            vec_len_loop = loop + 1
        else:
            vec_len_loop = 2 * L + 1 - loop
        mtx[0:vec_len_loop, loop] = vec[bg_idx:bg_idx + vec_len_loop]
        bg_idx += vec_len_loop
    return mtx


def vec_from_low_tri_col_by_col(mtx):
    """
    extract the data from the matrix, which is of the form:

        0 0 0 x 0 0 0
        0 0 x x x 0 0
        0 x x x x x 0
        x x x x x x x

    :param mtx: the matrix to extract the data from
    :return: data vector
    """
    L = mtx.shape[0] - 1
    assert mtx.shape[1] == 2 * L + 1
    vec = np.zeros((L + 1) ** 2, dtype=mtx.dtype)
    bg_idx = 0
    for loop in xrange(2 * L + 1):
        if loop <= L:
            vec_len_loop = loop + 1
        else:
            vec_len_loop = 2 * L + 1 - loop
        vec[bg_idx:bg_idx + vec_len_loop] = mtx[-vec_len_loop:, loop]
        bg_idx += vec_len_loop
    return vec


def sph_harm_ufnc(l, m, theta, phi):
    """
    compute spherical harmonics with the built-in sph_harm function.
    The difference is the name of input angle names as well as the normlisation factor.
    :param l: degree of spherical harmonics
    :param m: order of the spherical harmonics
    :param theta: co-latitude
    :param phi: azimuth
    :return:
    """
    return (-1) ** m * sp.special.sph_harm(m, l, phi, theta)


def compute_Pmn_coef(l, m):
    """
    compute the polynomial coefficients of degree l order m with canonical basis.
    :param l: degree of Legendre polynomial
    :param m: order of Legendre polynomial
    :return:
    """
    abs_m = np.abs(m)
    assert abs_m <= l
    pnm = np.zeros(l - abs_m + 1)
    pn = legendre_poly_canonical_basis_coef(l)
    for n in xrange(l - abs_m + 1):
        pnm[n] = pn[n + abs_m] * sp.misc.factorial(n + abs_m) / sp.misc.factorial(n)
    return pnm


def legendre_poly_canonical_basis_coef(l):
    """
    compute the legendred polynomial expressed in terms of canonical basis.
    :param l: degree of the legendre polynomial
    :return: p_n's: the legendre polynomial coefficients in ascending degree
    """
    p = np.zeros(l + 1, dtype=float)
    l_half = np.int(np.floor(l / 2.))
    # l is even
    if l % 2 == 0:
        for n in xrange(0, l_half + 1):
            p[2 * n] = (-1) ** (l_half - n) * sp.misc.comb(l, l_half - n) * \
                       sp.misc.comb(l + 2 * n, l)
    else:
        for n in xrange(0, l_half + 1):
            p[2 * n + 1] = (-1) ** (l_half - n) * sp.misc.comb(l, l_half - n) * \
                           sp.misc.comb(l + 2 * n + 1, l)
    p /= (2 ** l)
    return p


# ===========plotting functions===========
def sph_plot_diracs(theta_ref, phi_ref, theta, phi,
                    num_mic, P, save_fig=False,
                    file_name='sph_recon_2d_dirac.pdf'):
    dist_recon = sph_distance(1, theta_ref, phi_ref, theta, phi)[0]
    fig = plt.figure(figsize=(6.47, 4), dpi=90)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)

    K = theta_ref.size
    K_est = theta.size
    x = phi.copy()
    y = np.pi / 2. - theta.copy()  # convert CO-LATITUDE to LATITUDE
    x_ref = phi_ref.copy()
    y_ref = np.pi / 2. - theta_ref.copy()  # convert CO-LATITUDE to LATITUDE

    ind = x > np.pi
    x[ind] -= 2 * np.pi  # scale conversion to -pi to pi

    ind = x_ref > np.pi
    x_ref[ind] -= 2 * np.pi

    ax.scatter(x_ref, y_ref,
               c=np.tile([0, 0.447, 0.741], (K, 1)),
               s=70, alpha=0.75,
               marker='^', linewidths=0, cmap='Spectral_r',
               label='Original')
    ax.scatter(x, y,
               c=np.tile([0.850, 0.325, 0.098], (K_est, 1)),
               s=100, alpha=0.75,
               marker='*', linewidths=0, cmap='Spectral_r',
               label='Reconstruction')
    ax.set_xticklabels([u'210\N{DEGREE SIGN}', u'240\N{DEGREE SIGN}',
                        u'270\N{DEGREE SIGN}', u'300\N{DEGREE SIGN}',
                        u'330\N{DEGREE SIGN}', u'0\N{DEGREE SIGN}',
                        u'30\N{DEGREE SIGN}', u'60\N{DEGREE SIGN}',
                        u'90\N{DEGREE SIGN}', u'120\N{DEGREE SIGN}',
                        u'150\N{DEGREE SIGN}'])
    ax.legend(scatterpoints=1, loc=8, fontsize=9,
              ncol=2, bbox_to_anchor=(0.5, -0.13),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)  # framealpha=0.3,
    title_str = r'$K={0}, \mbox{{number of mic.}}={1}, \mbox{{SNR}}={2:.2f}$dB, average error={3:.2e}'
    ax.set_title(title_str.format(repr(K), repr(num_mic), P, dist_recon),
                 fontsize=11)
    ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
    ax.set_ylabel(r'latitude $90^{\circ}-\bm{\theta}$', fontsize=11)
    ax.xaxis.set_label_coords(0.5, 0.52)
    if save_fig:
        plt.savefig(file_name, format='pdf', dpi=300, transparent=True)
    plt.show()


if __name__ == '__main__':
    pass
    # L = 3
    # tmp_r, tmp_i = sph_Hsym_ext(L)
    # plt.imshow(tmp_r, cmap='Spectral_r')
    # plt.imshow(tmp_i, cmap='Spectral_r')
    # print np.dot(tmp_i, np.arange(np.int((L + 1) * (L) / 2.)) + 1)

    # for test purposes
    # radius_array = 0.3
    # K = 3
    # L = 5
    # num_bands = 5
    # alpha_ks, theta_ks, phi_ks = \
    #     sph_gen_diracs_param(K, num_bands=num_bands, positive_amp=True, log_normal_amp=True,
    #                          semisphere=True, save_param=False)[:3]
    # p_mic_x, p_mic_y, p_mic_z = \
    #     sph_gen_mic_array(radius_array, num_mic_per_band=8, num_bands=num_bands,
    #                       max_ratio_omega=5, save_layout=False)[:3]
    # a = sph_gen_visibility(alpha_ks, theta_ks, phi_ks, p_mic_x, p_mic_y, p_mic_z)
    #
    # # G_freq2vis_lst = sph_mtx_freq2visibility(L, p_mic_x, p_mic_y, p_mic_z)
    # # G_row = sph_mtx_fri2visibility_row_major(L, G_freq2vis_lst)
    # # G_col = sph_mtx_fri2visibility_col_major(L, G_freq2vis_lst)
    # # G_col_ri = cpx_mtx2real(G_col)
    # #
    # # amp_mtx = sph_mtx_amp(theta_ks, phi_ks, p_mic_x, p_mic_y, p_mic_z)
    # #
    # # G_updated_row = sph_mtx_updated_G_row_major(theta_ks, phi_ks, L, p_mic_x,
    # #                                             p_mic_y, p_mic_z, G_freq2vis_lst)
    # # G_updated_col = sph_mtx_updated_G_col_major_ri(theta_ks, phi_ks, L, p_mic_x,
    # #                                                p_mic_y, p_mic_z, G_freq2vis_lst)
    # noise_level = 1e-10
    # thetak_recon, phik_recon, alphak_recon = \
    #     sph_recon_2d_dirac(a, p_mic_x, p_mic_y, p_mic_z, K, L, noise_level,
    #                        max_ini=20, stop_cri='max_iter', num_rotation=2,
    #                        verbose=True, update_G=True, G_iter=4)
    # print(np.degrees(theta_ks))
    # print(np.degrees(phi_ks))
