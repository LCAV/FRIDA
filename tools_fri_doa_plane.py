from __future__ import division
import datetime
import time
import numpy as np
import scipy as sp
from scipy import linalg
import scipy.special
import scipy.optimize
from functools import partial
from joblib import Parallel, delayed
import os

if os.environ.get('DISPLAY') is None:
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib import rcParams

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{bm}"]


def polar_distance(x1, x2):
    """
    Given two arrays of numbers x1 and x2, pairs the cells that are the
    closest and provides the pairing matrix index: x1(index(1,:)) should be as
    close as possible to x2(index(2,:)). The function outputs the average of the
    absolute value of the differences abs(x1(index(1,:))-x2(index(2,:))).
    :param x1: vector 1
    :param x2: vector 2
    :return: d: minimum distance between d
             index: the permutation matrix
    """
    x1 = np.reshape(x1, (1, -1), order='F')
    x2 = np.reshape(x2, (1, -1), order='F')
    N1 = x1.size
    N2 = x2.size
    diffmat = np.arccos(np.cos(x1 - np.reshape(x2, (-1, 1), order='F')))
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
        d = np.mean(np.arccos(np.cos(x1[:, index[:, 0]] - x2[:, index[:, 1]])))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index


def polar2cart(rho, phi):
    """
    convert from polar to cartesian coordinates
    :param rho: radius
    :param phi: azimuth
    :return:
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def gen_diracs_param(K, positive_amp=True, log_normal_amp=False,
                     semicircle=True, save_param=True):
    """
    randomly generate Diracs' parameters
    :param K: number of Diracs
    :param positive_amp: whether Diracs have positive amplitudes or not
    :param log_normal_amp: whether the Diracs amplitudes follow log-normal distribution.
    :param semicircle: whether the Diracs are located on half of the circle or not.
    :param save_param: whether to save the Diracs' parameter or not.
    :return:
    """
    # amplitudes
    if log_normal_amp:
        positive_amp = True
    if not positive_amp:
        alpha_ks = np.sign(np.random.randn(K)) * (0.7 + 0.6 * (np.random.rand(K) - 0.5) / 1.)
    elif log_normal_amp:
        alpha_ks = np.random.lognormal(mean=np.log(2), sigma=0.7, size=K)
    else:
        alpha_ks = 0.7 + 0.6 * (np.random.rand(K) - 0.5) / 1.

    # location on the circle (S^1)
    if semicircle:
        factor = 1
    else:
        factor = 2

    exp_rnd = np.random.exponential(1. / (K - 1), K - 1)
    min_sep = 1. / 30
    phi_ks = np.cumsum(min_sep + (1 - (K - 1) * min_sep) *
                       (1. - 0.1 * np.random.rand(1, 1)) /
                       np.sum(exp_rnd) * exp_rnd)
    phi_ks = factor * np.pi * np.append(phi_ks, np.random.rand() * phi_ks[0] / 2.)

    time_stamp = datetime.datetime.now().strftime('%d-%m_%H_%M')
    if save_param:
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        file_name = './data/polar_Dirac_' + time_stamp + '.npz'
        np.savez(file_name, alpha_ks=alpha_ks,
                 phi_ks=phi_ks, K=K, time_stamp=time_stamp)
    return alpha_ks, phi_ks, time_stamp


def load_dirac_param(file_name):
    """
    load stored Diracs' parameters
    :param file_name: the file name that the parameters are stored
    :return:
    """
    stored_param = np.load(file_name)
    alpha_ks = stored_param['alpha_ks']
    phi_ks = stored_param['phi_ks']
    time_stamp = stored_param['time_stamp'].tostring()
    return alpha_ks, phi_ks, time_stamp


def gen_mic_array_2d(radius_array, num_mic=3, save_layout=True,
                     divi=3, plt_layout=False, **kwargs):
    """
    generate microphone array layout randomly
    :param radius_array: microphones are contained within a cirle of this radius
    :param num_mic: number of microphones
    :param save_layout: whether to save the microphone array layout or not
    :return:
    """
    # pos_array_norm = np.linspace(0, radius_array, num=num_mic, dtype=float)
    # pos_array_angle = np.linspace(0, 5 * np.pi, num=num_mic, dtype=float)
    num_seg = np.ceil(num_mic / divi)
    # radius_stepsize = radius_array / num_seg

    # pos_array_norm = np.append(np.repeat((np.arange(num_seg) + 1) * radius_stepsize,
    #                                      divi)[:num_mic-1], 0)
    pos_array_norm = np.linspace(0, radius_array, num=num_mic, endpoint=False)

    # pos_array_angle = np.append(np.tile(np.pi * 2 * np.arange(divi) / divi, num_seg)[:num_mic-1], 0)
    pos_array_angle = np.reshape(np.tile(np.pi * 2 * np.arange(divi) / divi, num_seg),
                                 (divi, -1), order='F') + \
                      np.linspace(0, 2 * np.pi / divi,
                                  num=num_seg, endpoint=False)[np.newaxis, :]
    pos_array_angle = np.insert(pos_array_angle.flatten('F')[:num_mic - 1], 0, 0)

    pos_array_angle += np.random.rand() * np.pi / divi
    # pos_array_norm = np.random.rand(num_mic) * radius_array
    # pos_array_angle = 2 * np.pi * np.random.rand(num_mic)
    pos_mic_x = pos_array_norm * np.cos(pos_array_angle)
    pos_mic_y = pos_array_norm * np.sin(pos_array_angle)

    layout_time_stamp = datetime.datetime.now().strftime('%d-%m')
    if save_layout:
        directory = './data/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = directory + 'mic_layout_' + layout_time_stamp + '.npz'
        np.savez(file_name, pos_mic_x=pos_mic_x, pos_mic_y=pos_mic_y,
                 layout_time_stamp=layout_time_stamp)

    if plt_layout:
        plt.figure(figsize=(3, 3), dpi=90)
        plt.plot(pos_mic_x, pos_mic_y, 'x')
        plt.axis('image')
        plt.xlim([-radius_array, radius_array])
        plt.ylim([-radius_array, radius_array])
        plt.title('microphone array layout', fontsize=11)

        if 'save_fig' in kwargs:
            save_fig = kwargs['save_fig']
        else:
            save_fig = False
        if 'fig_dir' in kwargs and save_fig:
            fig_dir = kwargs['fig_dir']
        else:
            fig_dir = './result/'
        if save_fig:
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            fig_name = (fig_dir + 'polar_numMic_{0}_layout' +
                        layout_time_stamp + '.pdf').format(repr(num_mic))
            plt.savefig(fig_name, format='pdf', dpi=300, transparent=True)

        # plt.show()
    return pos_mic_x, pos_mic_y, layout_time_stamp


def load_mic_array_param(file_name):
    """
    load stored microphone array parameters
    :param file_name: file that stored these parameters
    :return:
    """
    stored_param = np.load(file_name)
    pos_mic_x = stored_param['pos_mic_x']
    pos_mic_y = stored_param['pos_mic_y']
    layout_time_stamp = stored_param['layout_time_stamp'].tostring()
    return pos_mic_x, pos_mic_y, layout_time_stamp


def gen_sig_at_mic(sigmak2_k, phi_k, pos_mic_x,
                   pos_mic_y, SNR, Ns=256):
    """
    generate complex base-band signal received at microphones
    :param sigmak2_k: the variance of the circulant complex Gaussian signal
                emitted by the K sources
    :param phi_k: source locations (azimuths)
    :param pos_mic_x: a vector that contains microphones' x coordinates
    :param pos_mic_y: a vector that contains microphones' y coordinates
    :param SNR: SNR for the received signal at microphones
    :param Ns: number of snapshots used to estimate the covariance matrix
    :return: y_mic: received (complex) signal at microphones
    """
    xk, yk = polar2cart(1, phi_k)  # source locations in cartesian coordinates
    # reshape to use bordcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')
    pos_mic_x = np.reshape(pos_mic_x, (-1, 1), order='F')
    pos_mic_y = np.reshape(pos_mic_y, (-1, 1), order='F')

    # TODO: adapt to realistic settings
    omega_band = 1
    t = np.reshape(np.linspace(0, 10 * np.pi, num=Ns), (1, -1), order='F')
    K = sigmak2_k.size
    sigmak2_k = np.reshape(sigmak2_k, (-1, 1), order='F')

    # x_tilde_k size: K x lenthg_of_t
    # circular complex Gaussian process
    x_tilde_k = np.sqrt(sigmak2_k / 2.) * (np.random.randn(K, Ns) + 1j *
                                           np.random.randn(K, Ns))
    y_mic = np.dot(np.exp(-1j * (xk * pos_mic_x + yk * pos_mic_y)),
                   x_tilde_k * np.exp(1j * omega_band * t))
    signal_energy = linalg.norm(y_mic, 'fro') ** 2
    noise_energy = signal_energy / 10 ** (SNR * 0.1)
    sigma2_noise = noise_energy / Ns
    noise = np.sqrt(sigma2_noise / 2.) * (np.random.randn(*y_mic.shape) + 1j *
                                          np.random.randn(*y_mic.shape))
    y_mic_noisy = y_mic + noise
    return y_mic_noisy, y_mic


def cov_mtx_est(y_mic):
    """
    estimate covariance matrix
    :param y_mic: received signal (complex based band representation) at microphoens
    :return:
    """
    # Q: total number of microphones
    # num_snapshot: number of snapshots used to estimate the covariance matrix
    Q, num_snapshot = y_mic.shape
    cov_mtx = np.zeros((Q, Q), dtype=complex, order='F')
    for q in xrange(Q):
        y_mic_outer = y_mic[q, :]
        for qp in xrange(Q):
            y_mic_inner = y_mic[qp, :]
            cov_mtx[qp, q] = np.dot(y_mic_outer, y_mic_inner.T.conj())
    return cov_mtx / num_snapshot


def extract_off_diag(mtx):
    """
    extract off diagonal entries in mtx.
    The output vector is order in a column major manner.
    :param mtx: input matrix to extract the off diagonal entries
    :return:
    """
    # we transpose the matrix because the function np.extract will first flatten the matrix
    # withe ordering convention 'C' instead of 'F'!!
    extract_cond = np.reshape((1 - np.eye(*mtx.shape)).T.astype(bool), (-1, 1), order='F')
    return np.reshape(np.extract(extract_cond, mtx.T), (-1, 1), order='F')


def gen_visibility(alphak, phi_k, pos_mic_x, pos_mic_y):
    """
    generate visibility from the Dirac parameter and microphone array layout
    :param alphak: Diracs' amplitudes
    :param phi_k: azimuths
    :param pos_mic_x: a vector that contains microphones' x coordinates
    :param pos_mic_y: a vector that contains microphones' y coordinates
    :return:
    """
    xk, yk = polar2cart(1, phi_k)
    num_mic = pos_mic_x.size
    visi = np.zeros((num_mic, num_mic), dtype=complex)
    for q in xrange(num_mic):
        p_x_outer = pos_mic_x[q]
        p_y_outer = pos_mic_y[q]
        for qp in xrange(num_mic):
            p_x_qqp = p_x_outer - pos_mic_x[qp]  # a scalar
            p_y_qqp = p_y_outer - pos_mic_y[qp]  # a scalar
            visi[qp, q] = np.dot(np.exp(-1j * (xk * p_x_qqp + yk * p_y_qqp)), alphak)
    return visi


def add_noise(visi_noiseless, var_noise, num_mic, Ns=256):
    """
    add noise to the noiselss visibility
    :param visi_noiseless: noiseless visibilities
    :param var_noise: variance of noise
    :param num_mic: number of microphones
    :param Ns: number of samples used to estimate the covariance matrix
    :return:
    """
    sigma_mtx = visi_noiseless + var_noise * np.eye(*visi_noiseless.shape)
    wischart_mtx = np.kron(sigma_mtx.conj(), sigma_mtx) / Ns
    # the noise vairance matrix is given by the Cholesky decomposition
    noise_conv_mtx_sqrt = np.linalg.cholesky(wischart_mtx)
    visi_noiseless_vec = np.reshape(visi_noiseless, (-1, 1), order='F')
    noise = np.dot(noise_conv_mtx_sqrt,
                   np.random.randn(*visi_noiseless_vec.shape) +
                   1j * np.random.randn(*visi_noiseless_vec.shape)) / np.sqrt(2)
    # a matrix form
    visi_noisy = np.reshape(visi_noiseless_vec + noise, visi_noiseless.shape, order='F')
    # extract the off-diagonal entries
    visi_noisy = extract_off_diag(visi_noisy)
    visi_noiseless_off_diag = extract_off_diag(visi_noiseless)
    # calculate the equivalent SNR
    noise = visi_noisy - visi_noiseless_off_diag
    P = 20 * np.log10(linalg.norm(visi_noiseless_off_diag) / linalg.norm(noise))
    return visi_noisy, P, noise, visi_noiseless_off_diag


def gen_dirty_img(visi, pos_mic_x, pos_mic_y, phi_plt):
    """
    Compute the dirty image associated with the given measurements. Here the Fourier transform
    that is not measured by the microphone array is taken as zero.
    :param visi: the measured visibilites
    :param pos_mic_x: a vector contains microphone array locations (x-coordinates)
    :param pos_mic_y: a vector contains microphone array locations (y-coordinates)
    :param phi_plt: plotting grid (azimuth on the circle) to show the dirty image
    :return:
    """
    img = np.zeros(phi_plt.size, dtype=complex)
    x_plt, y_plt = polar2cart(1, phi_plt)
    num_mic = pos_mic_x.size
    count_visi = 0
    for q in xrange(num_mic):
        p_x_outer = pos_mic_x[q]
        p_y_outer = pos_mic_y[q]
        for qp in xrange(num_mic):
            if not q == qp:
                p_x_qqp = p_x_outer - pos_mic_x[qp]  # a scalar
                p_y_qqp = p_y_outer - pos_mic_y[qp]  # a scalar
                img += visi[count_visi] * np.exp(1j * (p_x_qqp * x_plt + p_y_qqp * y_plt))
                count_visi += 1
    return img / phi_plt.size


def mtx_freq2visi(M, p_mic_x, p_mic_y):
    """
    build the matrix that maps the Fourier series to the visibility
    :param M: the Fourier series expansion is limited from -M to M
    :param p_mic_x: a vector that constains microphones x coordinates
    :param p_mic_y: a vector that constains microphones y coordinates
    :return:
    """
    num_mic = p_mic_x.size
    ms = np.reshape(np.arange(-M, M + 1, step=1), (1, -1), order='F')
    G = np.zeros((num_mic * (num_mic - 1), 2 * M + 1), dtype=complex, order='C')
    count_G = 0
    for q in xrange(num_mic):
        p_x_outer = p_mic_x[q]
        p_y_outer = p_mic_y[q]
        for qp in xrange(num_mic):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x[qp]
                p_y_qqp = p_y_outer - p_mic_y[qp]
                norm_p_qqp = np.sqrt(p_x_qqp ** 2 + p_y_qqp ** 2)
                phi_qqp = np.arctan2(p_y_qqp, p_x_qqp)
                G[count_G, :] = (-1j) ** ms * sp.special.jv(ms, norm_p_qqp) * \
                                np.exp(1j * ms * phi_qqp)
                count_G += 1
    return G


def mtx_fri2visi_ri(M, p_mic_x, p_mic_y, D1, D2):
    """
    build the matrix that maps the Fourier series to the visibility in terms of
    REAL-VALUDED entries only. (matrix size double)
    :param M: the Fourier series expansion is limited from -M to M
    :param p_mic_x: a vector that constains microphones x coordinates
    :param p_mic_y: a vector that constains microphones y coordinates
    :param D1: expansion matrix for the real-part
    :param D2: expansion matrix for the imaginary-part
    :return:
    """
    return np.dot(cpx_mtx2real(mtx_freq2visi(M, p_mic_x, p_mic_y)),
                  linalg.block_diag(D1, D2))


def cpx_mtx2real(mtx):
    """
    extend complex valued matrix to an extended matrix of real values only
    :param mtx: input complex valued matrix
    :return:
    """
    return np.vstack((np.hstack((mtx.real, -mtx.imag)), np.hstack((mtx.imag, mtx.real))))


def hermitian_expan(half_vec_len):
    """
    expand a real-valued vector to a Hermitian symmetric vector.
    The input vector is a concatenation of the real parts with NON-POSITIVE indices and
    the imaginary parts with STRICTLY-NEGATIVE indices.
    :param half_vec_len: length of the first half vector
    :return:
    """
    D0 = np.eye(half_vec_len)
    D1 = np.vstack((D0, D0[1:, ::-1]))
    D2 = np.vstack((D0, -D0[1:, ::-1]))
    D2 = D2[:, :-1]
    return D1, D2


def coef_expan_mtx(K):
    """
    expansion matrix for an annihilating filter of size K + 1
    :param K: number of Dirac. The filter size is K + 1
    :return:
    """
    if K % 2 == 0:
        D0 = np.eye(K / 2. + 1)
        D1 = np.vstack((D0, D0[1:, ::-1]))
        D2 = np.vstack((D0, -D0[1:, ::-1]))[:, :-1]
    else:
        D0 = np.eye((K + 1) / 2.)
        D1 = np.vstack((D0, D0[:, ::-1]))
        D2 = np.vstack((D0, -D0[:, ::-1]))
    return D1, D2


def Tmtx_ri(b_ri, K, D, L):
    """
    build convolution matrix associated with b_ri
    :param b_ri: a real-valued vector
    :param K: number of Diracs
    :param D1: exapansion matrix for the real-part
    :param D2: exapansion matrix for the imaginary-part
    :return:
    """
    b_ri = np.dot(D, b_ri)
    b_r = b_ri[:L]
    b_i = b_ri[L:]
    Tb_r = linalg.toeplitz(b_r[K:], b_r[K::-1])
    Tb_i = linalg.toeplitz(b_i[K:], b_i[K::-1])
    return np.vstack((np.hstack((Tb_r, -Tb_i)), np.hstack((Tb_i, Tb_r))))


def Tmtx_ri_half(b_ri, K, D, L, D_coef):
    return np.dot(Tmtx_ri(b_ri, K, D, L), D_coef)


def Rmtx_ri(coef_ri, K, D, L):
    coef_ri = np.squeeze(coef_ri)
    coef_r = coef_ri[:K + 1]
    coef_i = coef_ri[K + 1:]
    R_r = linalg.toeplitz(np.concatenate((np.array([coef_r[-1]]),
                                          np.zeros(L - K - 1))),
                          np.concatenate((coef_r[::-1],
                                          np.zeros(L - K - 1)))
                          )
    R_i = linalg.toeplitz(np.concatenate((np.array([coef_i[-1]]),
                                          np.zeros(L - K - 1))),
                          np.concatenate((coef_i[::-1],
                                          np.zeros(L - K - 1)))
                          )
    return np.dot(np.vstack((np.hstack((R_r, -R_i)), np.hstack((R_i, R_r)))), D)


def Rmtx_ri_half(coef_half, K, D, L, D_coef):
    coef_ri = np.dot(D_coef, coef_half)
    return Rmtx_ri(coef_ri, K, D, L)


def build_mtx_amp(phi_k, p_mic_x, p_mic_y):
    """
    the matrix that maps Diracs' amplitudes to the visibility
    :param phi_k: Diracs' location (azimuth)
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :return:
    """
    xk, yk = polar2cart(1, phi_k)
    num_mic = p_mic_x.size
    K = phi_k.size
    mtx = np.zeros((num_mic * (num_mic - 1), K), dtype=complex, order='C')
    count_inner = 0
    for q in xrange(num_mic):
        p_x_outer = p_mic_x[q]
        p_y_outer = p_mic_y[q]
        for qp in xrange(num_mic):
            if not q == qp:
                p_x_qqp = p_x_outer - p_mic_x[qp]  # a scalar
                p_y_qqp = p_y_outer - p_mic_y[qp]  # a scalar
                mtx[count_inner, :] = np.exp(-1j * (xk * p_x_qqp + yk * p_y_qqp))
                count_inner += 1
    return mtx


def mtx_updated_G(phi_recon, M, mtx_amp2visi_ri, mtx_fri2visi_ri):
    """
    Update the linear transformation matrix that links the FRI sequence to the
    visibilities by using the reconstructed Dirac locations.
    :param phi_recon: the reconstructed Dirac locations (azimuths)
    :param M: the Fourier series expansion is between -M to M
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param mtx_freq2visi: the linear mapping from Fourier series to visibilities
    :return:
    """
    L = 2 * M + 1
    ms_half = np.reshape(np.arange(-M, 1, step=1), (-1, 1), order='F')
    phi_recon = np.reshape(phi_recon, (1, -1), order='F')
    mtx_amp2freq = np.exp(-1j * ms_half * phi_recon)  # size: (M + 1) x K
    mtx_amp2freq_ri = np.vstack((mtx_amp2freq.real, mtx_amp2freq.imag[:-1, :]))  # size: (2M + 1) x K
    mtx_fri2amp_ri = linalg.lstsq(mtx_amp2freq_ri, np.eye(L))[0]
    # projection mtx_freq2visi to the null space of mtx_fri2amp
    mtx_null_proj = np.eye(L) - np.dot(mtx_fri2amp_ri.T,
                                       linalg.lstsq(mtx_fri2amp_ri.T, np.eye(L))[0])
    G_updated = np.dot(mtx_amp2visi_ri, mtx_fri2amp_ri) + \
                np.dot(mtx_fri2visi_ri, mtx_null_proj)
    return G_updated


def dirac_recon_ri(G, a_ri, K, M, noise_level, max_ini=100, stop_cri='mse'):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements
    :param G: the linear transformation matrix that links the visibilities to
                uniformly sampled sinusoids
    :param a_ri: the visibility measurements
    :param K: number of Diracs
    :param M: the Fourier series expansion is between -M and M
    :param noise_level: level of noise (ell_2 norm) in the measurements
    :param max_ini: maximum number of initialisations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :return:
    """
    L = 2 * M + 1  # length of the (complex-valued) b vector
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    GtG = np.dot(G.T, G)
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    # the least-square solution
    beta_ri = linalg.lstsq(G, a_ri)[0]
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    # size of Tbeta_ri: 2(L - K) x 2(K + 1)
    Tbeta_ri = Tmtx_ri(beta_ri, K, D, L)

    # size of various matrices / vectors
    sz_G1 = L

    sz_Tb0 = 2 * (L - K)
    sz_Tb1 = 2 * (K + 1)

    sz_Rc0 = 2 * (L - K)
    sz_Rc1 = L

    sz_coef = 2 * (K + 1)
    sz_bri = L

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))

    # the main iteration with different random initialisations
    for ini in xrange(max_ini):
        c_ri = np.random.randn(sz_coef, 1)
        c0_ri = c_ri.copy()
        error_seq = np.zeros(max_iter, dtype=float)
        R_loop = Rmtx_ri(c_ri, K, D, L)
        for inner in xrange(max_iter):
            mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)),
                                             Tbeta_ri.T,
                                             np.zeros((sz_coef, sz_Rc1)),
                                             c0_ri)),
                                  np.hstack((Tbeta_ri,
                                             np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop,
                                             np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
                                             -R_loop.T,
                                             GtG,
                                             np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0_ri.T,
                                             np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be symmetric
            mtx_loop = (mtx_loop + mtx_loop.T) / 2.
            c_ri = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]
            # c_ri = linalg.solve(mtx_loop, rhs)[:sz_coef]

            R_loop = Rmtx_ri(c_ri, K, D, L)
            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                    ))
            mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
            b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]
            # b_recon_ri = linalg.solve(mtx_brecon, rhs_bl)[:sz_bri]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt, min_error, b_opt, ini


def dirac_recon_ri_half(G, a_ri, K, M, noise_level, max_ini=100, stop_cri='mse'):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements.
    Here we enforce hermitian symmetry in the annihilating filter coefficients so that
    roots on the unit circle are encouraged.
    :param G: the linear transformation matrix that links the visibilities to
                uniformly sampled sinusoids
    :param a_ri: the visibility measurements
    :param K: number of Diracs
    :param M: the Fourier series expansion is between -M and M
    :param noise_level: level of noise (ell_2 norm) in the measurements
    :param max_ini: maximum number of initialisations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :return:
    """
    L = 2 * M + 1  # length of the (complex-valued) b vector
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    GtG = np.dot(G.T, G)
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    # the least-square solution
    beta_ri = linalg.lstsq(G, a_ri)[0]
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)
    # size of Tbeta_ri: 2(L - K) x 2(K + 1)
    Tbeta_ri = Tmtx_ri_half(beta_ri, K, D, L, D_coef)

    # size of various matrices / vectors
    sz_G1 = L

    sz_Tb0 = 2 * (L - K)
    sz_Tb1 = K + 1

    sz_Rc0 = 2 * (L - K)
    sz_Rc1 = L

    sz_coef = K + 1
    sz_bri = L

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))

    # the main iteration with different random initialisations
    for ini in xrange(max_ini):
        c_ri_half = np.random.randn(sz_coef, 1)
        c0_ri_half = c_ri_half.copy()
        error_seq = np.zeros(max_iter, dtype=float)
        R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
        for inner in xrange(max_iter):
            mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)),
                                             Tbeta_ri.T,
                                             np.zeros((sz_coef, sz_Rc1)),
                                             c0_ri_half)),
                                  np.hstack((Tbeta_ri,
                                             np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop,
                                             np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
                                             -R_loop.T,
                                             GtG,
                                             np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0_ri_half.T,
                                             np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be symmetric
            mtx_loop = (mtx_loop + mtx_loop.T) / 2.
            c_ri_half = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]
            # c_ri = linalg.solve(mtx_loop, rhs)[:sz_coef]

            R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                    ))
            mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
            b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]
            # b_recon_ri = linalg.solve(mtx_brecon, rhs_bl)[:sz_bri]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
                c_ri = np.dot(D_coef, c_ri_half)
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt, min_error, b_opt, ini


def dirac_recon_ri_half_parallel(G, a_ri, K, M, max_ini=100):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements.
    Here we enforce hermitian symmetry in the annihilating filter coefficients so that
    roots on the unit circle are encouraged.
    We use parallel implementation when stop_cri == 'max_iter'
    :param G: the linear transformation matrix that links the visibilities to
                uniformly sampled sinusoids
    :param a_ri: the visibility measurements
    :param K: number of Diracs
    :param M: the Fourier series expansion is between -M and M
    :param noise_level: level of noise (ell_2 norm) in the measurements
    :param max_ini: maximum number of initialisations
    :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
    :return:
    """
    L = 2 * M + 1  # length of the (complex-valued) b vector
    a_ri = a_ri.flatten('F')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50

    # the least-square solution
    beta_ri = linalg.lstsq(G, a_ri)[0]
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)

    # size of Tbeta_ri: 2(L - K) x 2(K + 1)
    Tbeta_ri = Tmtx_ri_half(beta_ri, K, D, L, D_coef)

    # size of various matrices / vectors
    sz_G1 = L

    sz_Tb0 = 2 * (L - K)

    sz_Rc0 = 2 * (L - K)

    sz_coef = K + 1

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))

    # the main iteration with different random initialisations
    partial_dirac_recon = partial(dirac_recon_ri_inner, a_ri=a_ri, rhs=rhs,
                                  rhs_bl=rhs_bl, K=K, M=M, D1=D1, D2=D2, D_coef=D_coef,
                                  Tbeta_ri=Tbeta_ri, G=G, max_iter=max_iter)

    # generate all the random initialisations
    c_ri_half_all = np.random.randn(sz_coef, max_ini)

    res_all = Parallel(n_jobs=-1)(
        delayed(partial_dirac_recon)(c_ri_half_all[:, loop][:, np.newaxis])
        for loop in xrange(max_ini))

    # find the one with smallest error
    min_idx = np.array(zip(*res_all)[1]).argmin()
    c_opt, min_error, b_opt = res_all[min_idx]

    return c_opt, min_error, b_opt


def dirac_recon_ri_inner(c_ri_half, a_ri, rhs, rhs_bl, K, M,
                         D1, D2, D_coef, Tbeta_ri, G, max_iter):
    min_error = float('inf')
    # size of various matrices / vectors
    L = 2 * M + 1  # length of the (complex-valued) b vector

    sz_Tb0 = 2 * (L - K)
    sz_Tb1 = K + 1

    sz_Rc0 = 2 * (L - K)
    sz_Rc1 = L

    sz_coef = K + 1
    sz_bri = L

    GtG = np.dot(G.T, G)
    D = linalg.block_diag(D1, D2)

    c0_ri_half = c_ri_half.copy()
    error_seq = np.zeros(max_iter, dtype=float)
    R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
    for inner in xrange(max_iter):
        mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)),
                                         Tbeta_ri.T,
                                         np.zeros((sz_coef, sz_Rc1)),
                                         c0_ri_half)),
                              np.hstack((Tbeta_ri,
                                         np.zeros((sz_Tb0, sz_Tb0)),
                                         -R_loop,
                                         np.zeros((sz_Rc0, 1))
                                         )),
                              np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
                                         -R_loop.T,
                                         GtG,
                                         np.zeros((sz_Rc1, 1))
                                         )),
                              np.hstack((c0_ri_half.T,
                                         np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                         ))
                              ))
        # matrix should be symmetric
        mtx_loop = (mtx_loop + mtx_loop.T) / 2.
        c_ri_half = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]

        R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
        mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                ))
        mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
        b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]

        error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
        if error_seq[inner] < min_error:
            min_error = error_seq[inner]
            b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
            c_ri = np.dot(D_coef, c_ri_half)
            c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts
    return c_opt, min_error, b_opt


def pt_src_recon(a, p_mic_x, p_mic_y, K, M, noise_level, max_ini=50,
                 stop_cri='mse', update_G=False, verbose=False, **kwargs):
    """
    reconstruct point sources on the circle from the visibility measurements
    :param a: the measured visibilities
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param K: number of point sources
    :param M: the Fourier series expansion is between -M to M
    :param noise_level: noise level in the measured visibilities
    :param max_ini: maximum number of random initialisation used
    :param stop_cri: either 'mse' or 'max_iter'
    :param update_G: update the linear mapping that links the uniformly sampled
                sinusoids to the visibility or not.
    :param kwargs: possible optional input: G_iter: number of iterations for the G updates
    :return:
    """
    assert M >= K
    num_mic = p_mic_x.size
    assert num_mic * (num_mic - 1) >= 2 * M + 1
    a_ri = np.concatenate((a.real, a.imag))
    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    # initialisation
    min_error = float('inf')
    phik_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # expansion matrices to take Hermitian symmetry into account
    D1, D2 = hermitian_expan(M + 1)
    G = mtx_fri2visi_ri(M, p_mic_x, p_mic_y, D1, D2)
    for loop_G in xrange(max_loop_G):
        # c_recon, error_recon = dirac_recon_ri(G, a_ri, K_est, M, noise_level, max_ini, stop_cri)[:2]
        # c_recon, error_recon = dirac_recon_ri_half(G, a_ri, K, M, noise_level, max_ini, stop_cri)[:2]
        c_recon, error_recon = dirac_recon_ri_half_parallel(G, a_ri, K, M, max_ini)[:2]

        if verbose:
            print('noise level: {0:.3e}'.format(noise_level))
            print('objective function value: {0:.3e}'.format(error_recon))
        # recover Dirac locations
        uk = np.roots(np.squeeze(c_recon))
        uk /= np.abs(uk)
        phik_recon = np.mod(-np.angle(uk), 2. * np.pi)

        # use least square to reconstruct amplitudes
        amp_mtx = build_mtx_amp(phik_recon, p_mic_x, p_mic_y)
        amp_mtx_ri = np.vstack((amp_mtx.real, amp_mtx.imag))
        alphak_recon = sp.optimize.nnls(amp_mtx_ri, a_ri.squeeze())[0]
        error_loop = linalg.norm(a.flatten('F') - np.dot(amp_mtx, alphak_recon))

        if error_loop < min_error:
            min_error = error_loop
            # amp_sort_idx = np.argsort(alphak_recon)[-K:]
            # phik_opt, alphak_opt = phik_recon[amp_sort_idx], alphak_recon[amp_sort_idx]
            phik_opt, alphak_opt = phik_recon, alphak_recon

        # update the linear transformation matrix
        if update_G:
            G = mtx_updated_G(phik_recon, M, amp_mtx_ri, G)

    return phik_opt, alphak_opt


def pt_src_recon_rotate(a, p_mic_x, p_mic_y, K, M, noise_level, max_ini=50,
                        stop_cri='mse', update_G=False, num_rotation=1,
                        verbose=False, **kwargs):
    """
    reconstruct point sources on the circle from the visibility measurements.
    Here we apply random rotations to the coordiantes.
    :param a: the measured visibilities
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param K: number of point sources
    :param M: the Fourier series expansion is between -M to M
    :param noise_level: noise level in the measured visibilities
    :param max_ini: maximum number of random initialisation used
    :param stop_cri: either 'mse' or 'max_iter'
    :param update_G: update the linear mapping that links the uniformly sampled
                sinusoids to the visibility or not.
    :param kwargs: possible optional input: G_iter: number of iterations for the G updates
    :return:
    """
    assert M >= K
    num_mic = p_mic_x.size
    assert num_mic * (num_mic - 1) >= 2 * M + 1
    a_ri = np.concatenate((a.real, a.imag))
    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    # expansion matrices to take Hermitian symmetry into account
    D1, D2 = hermitian_expan(M + 1)

    # initialisation
    min_error = float('inf')
    phik_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # random rotation angles
    rotate_angle_all = np.random.rand(num_rotation) * np.pi * 2.
    for rand_rotate in xrange(num_rotation):
        rotate_angle_loop = rotate_angle_all[rand_rotate]
        rotate_mtx = np.array([[np.cos(rotate_angle_loop), -np.sin(rotate_angle_loop)],
                               [np.sin(rotate_angle_loop), np.cos(rotate_angle_loop)]])

        # rotate microphone steering vector
        # (due to change of coordinate w.r.t. the random rotation)
        p_mic_xy_rotated = np.dot(rotate_mtx,
                                  np.row_stack((p_mic_x.flatten('F'),
                                                p_mic_y.flatten('F')))
                                  )
        p_mic_x_rotated = np.reshape(p_mic_xy_rotated[0, :], p_mic_x.shape, order='F')
        p_mic_y_rotated = np.reshape(p_mic_xy_rotated[1, :], p_mic_y.shape, order='F')

        # linear transformation matrix that maps uniform samples
        # of sinusoids to visibilities
        G = mtx_fri2visi_ri(M, p_mic_x_rotated, p_mic_y_rotated, D1, D2)

        for loop_G in xrange(max_loop_G):
            c_recon, error_recon = dirac_recon_ri_half(G, a_ri, K, M, noise_level,
                                                       max_ini, stop_cri)[:2]
            if verbose:
                print('noise level: {0:.3e}'.format(noise_level))
                print('objective function value: {0:.3e}'.format(error_recon))

            # recover Dirac locations
            uk = np.roots(np.squeeze(c_recon))
            uk /= np.abs(uk)
            phik_recon_rotated = np.mod(-np.angle(uk), 2. * np.pi)

            # use least square to reconstruct amplitudes
            amp_mtx = build_mtx_amp(phik_recon_rotated, p_mic_x_rotated, p_mic_y_rotated)
            amp_mtx_ri = np.vstack((amp_mtx.real, amp_mtx.imag))
            alphak_recon = sp.optimize.nnls(amp_mtx_ri, a_ri.squeeze())[0]
            error_loop = linalg.norm(a.flatten('F') - np.dot(amp_mtx, alphak_recon))

            if error_loop < min_error:
                min_error = error_loop

                # rotate back
                # transform to cartesian coordinate
                xk_recon_rotated, yk_recon_rotated = polar2cart(1, phik_recon_rotated)
                xy_rotate_back = linalg.solve(rotate_mtx,
                                              np.row_stack((xk_recon_rotated.flatten('F'),
                                                            yk_recon_rotated.flatten('F')))
                                              )
                xk_recon = np.reshape(xy_rotate_back[0, :], xk_recon_rotated.shape, 'F')
                yk_recon = np.reshape(xy_rotate_back[1, :], yk_recon_rotated.shape, 'F')
                # transform back to polar coordinate
                phik_recon = np.mod(np.arctan2(yk_recon, xk_recon), 2. * np.pi)

                phik_opt, alphak_opt = phik_recon, alphak_recon

            # update the linear transformation matrix
            if update_G:
                G = mtx_updated_G(phik_recon_rotated, M, amp_mtx_ri, G)

    return phik_opt, alphak_opt


def polar_plt_diracs(phi_ref, phi_recon, num_mic, P, save_fig=False, **kwargs):
    """
    plot Diracs in the polar coordinate
    :param phi_ref: ground truth Dirac locations (azimuths)
    :param phi_recon: reconstructed Dirac locations (azimuths)
    :param num_mic: number of microphones
    :param P: PSNR in the visibility measurements
    :param save_fig: whether save the figure or not
    :param kwargs: optional input argument(s), include:
            file_name: file name used to save figure
    :return:
    """
    dist_recon = polar_distance(phi_ref, phi_recon)[0]
    if 'dirty_img' in kwargs and 'phi_plt' in kwargs:
        plt_dirty_img = True
        dirty_img = kwargs['dirty_img']
        phi_plt = kwargs['phi_plt']
    else:
        plt_dirty_img = False
    fig = plt.figure(figsize=(5, 4), dpi=90)
    ax = fig.add_subplot(111, projection='polar')
    K = phi_ref.size
    K_est = phi_recon.size
    ax.scatter(phi_ref, np.ones(K), c=np.tile([0, 0.447, 0.741], (K, 1)), s=70,
               alpha=0.75, marker='^', linewidths=0, label='original')
    ax.scatter(phi_recon, np.ones(K_est), c=np.tile([0.850, 0.325, 0.098], (K_est, 1)), s=100,
               alpha=0.75, marker='*', linewidths=0, label='reconstruction')
    if plt_dirty_img:
        dirty_img = dirty_img.real
        min_val = dirty_img.min()
        max_val = dirty_img.max()
        # color_lines = cm.spectral_r((dirty_img - min_val) / (max_val - min_val))
        # ax.scatter(phi_plt, 1 + dirty_img, edgecolor='none', linewidths=0,
        #         c=color_lines, label='dirty image')  # 1 is for the offset
        ax.plot(phi_plt, 1 + dirty_img, linewidth=1.5,
                linestyle='-', color=[0.466, 0.674, 0.188], label='dirty image')
    ax.legend(scatterpoints=1, loc=8, fontsize=9,
              ncol=1, bbox_to_anchor=(0.5, 0.32),
              handletextpad=.2, columnspacing=1.7, labelspacing=0.1)
    title_str = r'$K={0}$, $\mbox{{\# of mic.}}={1}$, $\mbox{{SNR}}={2:.1f}$dB, average error={3:.1e}'
    ax.set_title(title_str.format(repr(K), repr(num_mic), P, dist_recon),
                 fontsize=11)
    ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.set_yticks(np.linspace(0,  1, 2))
    ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
    ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
    if plt_dirty_img:
        ax.set_ylim([0, 1.05 + max_val])
    else:
        ax.set_ylim([0, 1.05])
    if save_fig:
        if 'file_name' in kwargs:
            file_name = kwargs['file_name']
        else:
            file_name = 'polar_recon_dirac.pdf'
        plt.savefig(file_name, format='pdf', dpi=300, transparent=True)
    # plt.show()


if __name__ == '__main__':
    pass
    # K = 3
    # M = 4
    # L = 2 * M + 1
    # radius_array = 0.1
    # num_mic = 5
    #
    # alpha_ks, phi_ks = \
    #     gen_diracs_param(K, positive_amp=True, log_normal_amp=False,
    #                      semicircle=False, save_param=False)[:2]
    #
    # p_mic_x, p_mic_y = gen_mic_array_2d(radius_array, num_mic, save_layout=False)[:2]
    # visi_noiseless = \
    #     add_noise(gen_visibility(alpha_ks, phi_ks, p_mic_x, p_mic_y),
    #               10, num_mic, Ns=256)[3]
    # # a_ri = np.concatenate((visi_noiseless.real, visi_noiseless.imag))
    #
    # # D1, D2 = hermitian_expan(M + 1)
    # # G = mtx_freq2visi_ri(M, p_mic_x, p_mic_y, D1, D2)
    # #
    # # ms = np.reshape(np.arange(-M, M + 1, step=1), (-1, 1), order='F')
    # # b_noiseless = np.dot(np.exp(-1j * ms * np.reshape(phi_ks, (1, -1), order='F')),
    # #                      np.reshape(alpha_ks, (-1, 1), order='F'))
    # #
    # # b_ri = np.concatenate((np.real(b_noiseless[:M+1]), np.imag(b_noiseless[:M])))
    # # Gb = np.dot(G, b_ri)
    # # Gb_cpx = Gb[:num_mic*(num_mic -1)] + 1j *Gb[num_mic*(num_mic -1):]
    # # print linalg.norm(Gb_cpx - visi_noiseless)
    #
    # plt.plot(p_mic_x, p_mic_y, 'x')
    # plt.axis('image')
    # plt.xlim([-radius_array, radius_array])
    # plt.ylim([-radius_array, radius_array])
    # plt.show()
    #
    # # plt.figure(figsize=(8, 2), dpi=300)
    # # plt.subplot(1, 2, 1)
    # # plt.plot(np.real(visi_noiseless), 'r')
    # # plt.plot(np.real(Gb_cpx), 'b', hold=True)
    # # plt.subplot(1, 2, 2)
    # # plt.plot(np.imag(visi_noiseless), 'r')
    # # plt.plot(np.imag(Gb_cpx), 'b', hold=True)
    # # plt.show()
    # #
    # # beta_opt = b_ri
    # noise_level = 1e-10
    # # c_opt, min_error, b_opt, ini =\
    # #     dirac_recon_ri(G, a_ri, K, M, noise_level, max_ini=100, stop_cri='mse')
    # phik_recon, alphak_recon = \
    #     pt_src_recon(visi_noiseless, p_mic_x, p_mic_y, K, M, noise_level, max_ini=50,
    #                  stop_cri='mse', update_G=True, G_iter=5)
    # recon_err, sort_idx = polar_distance(phik_recon, phi_ks)
    # print('Original azimuths       : {0}'.format(phi_ks[sort_idx[:, 1]]))
    # print('Reconstructed azimuths  : {0}'.format(phik_recon[sort_idx[:, 0]]))
    # print('Original amplitudes     : {0}'.format(alpha_ks[sort_idx[:, 1]]))
    # print('Reconstructed amplitudes: {0}'.format(alphak_recon[sort_idx[:, 0]]))
    #
    # P = float('inf')
    # polar_plt_diracs(phi_ks, phik_recon, num_mic, P, save_fig=False)
