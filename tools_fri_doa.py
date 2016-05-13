from __future__ import division
import datetime
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


def sph_gen_diracs_param(K, positive_amp=True,
                         log_normal_amp=False,
                         semisphere=True,
                         save_param=True):
    """
    Randomly generate Diracs' parameters.
    :param K: number of Diracs
    :param positive_amp: whether the Diracs have positive amplitudes or not.
    :param log_normal_amp: whether the Diracs amplitudes follow log-normal distribution.
    :param save_param: whether to save the Diracs' parameter or not.
    :return:
    """
    # amplitudes
    if log_normal_amp:
        positive_amp = True
    if not positive_amp:
        alpha_ks = np.sign(np.random.randn(K)) * (0.7 + (np.random.rand(K) - 0.5) / 1.)
    elif log_normal_amp:
        alpha_ks = np.random.lognormal(mean=np.log(2), sigma=0.7, size=(K,))
    else:
        alpha_ks = 0.7 + (np.random.rand(K) - 0.5) / 1.

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
                 phi_ks=phi_ks, K=K, time_stamp=time_stamp)
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
    radius_sph = 20. * radius_array
    pos_array_norm = np.linspace(0, radius_array, num=num_mic_per_band, dtype=float)
    pos_array_angle = np.linspace(0, 12. * np.pi, num=num_mic_per_band, dtype=float)

    pos_mic_x = pos_array_norm * np.cos(pos_array_angle)
    pos_mic_y = pos_array_norm * np.sin(pos_array_angle)

    phi_mic = np.arctan2(pos_mic_y, pos_mic_x)
    theta_mic = np.arcsin(np.sqrt(pos_mic_x ** 2 + pos_mic_y ** 2) / radius_sph)

    p_mic0_x, p_mic0_y, p_mic0_z = sph2cart(radius_sph, theta_mic, phi_mic)
    # reshape to use broadcasting
    p_mic0_x = np.reshape(p_mic0_x, (-1, 1), order='F')
    p_mic0_y = np.reshape(p_mic0_y, (-1, 1), order='F')
    p_mic0_z = np.reshape(p_mic0_z, (-1, 1), order='F')

    # mid-band frequencies of different sub-bands
    mid_band_freq = np.random.rand(1, num_bands) * (max_ratio_omega - 1) + 1

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


def sph_mtx_freq2visibility(L, p_mic_x, p_mic_y, p_mic_z):
    """
    build the linear mapping matrix from the spherical harmonics to the measured visibility.
    :param L: the maximum degree of spherical harmonics
    :param p_mic_x: a num_mic x num_bands matrix that contains microphones' x coordinates
    :param p_mic_y: a num_mic y num_bands matrix that contains microphones' x coordinates
    :param p_mic_z: a num_mic z num_bands matrix that contains microphones' x coordinates
    :return:
    """
    num_mic_per_band, num_bands = p_mic_x.shape
    m_grid, l_grid = np.meshgrid(np.arange(-L, L + 1, step=1, dtype=int),
                                 np.arange(0, L + 1, step=1, dtype=int))
    m_grid = np.reshape(vec_from_low_tri_col_by_col(m_grid), (1, -1), order='F')
    l_grid = np.reshape(vec_from_low_tri_col_by_col(l_grid), (1, -1), order='F')
    # parallel implemnetation (over all the subands at once)
    # TODO: change the G_lst if needed
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
    G_blk = np.zeros((num_mic_per_band - 1, (L + 1) ** 2), dtype=complex, order='C')
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
                G_blk[count_G_band, :] = sph_bessel_ells[l_grid.squeeze()] * \
                                         sph_harm_ufnc(l_grid, m_grid, theta_qqp, phi_qqp)
                count_G_band += 1
    return G_blk * (2 * np.pi) ** 1.5


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
