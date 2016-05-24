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
    if K == 1:
        phi_ks = factor * np.pi * np.random.rand()
    else:
        phi_ks = factor * np.pi * np.random.rand(K)

    time_stamp = datetime.datetime.now().strftime('%d-%m_%H_%M')
    if save_param:
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        file_name = './data/sph_Dirac_' + time_stamp + '.npz'
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


def gen_mic_array_2d(radius_array, num_mic=3, save_layout=True):
    """
    generate microphone array layout randomly
    :param radius_array: microphones are contained within a cirle of this radius
    :param num_mic: number of microphones
    :param save_layout: whether to save the microphone array layout or not
    :return:
    """
    # pos_array_norm = np.linspace(0, radius_array, num=num_mic, dtype=float)
    # pos_array_angle = np.linspace(0, 5 * np.pi, num=num_mic, dtype=float)
    num_seg = np.ceil(num_mic / 3.)
    radius_stepsize = radius_array / num_seg
    pos_array_norm = np.append(np.repeat((np.arange(num_seg) + 1) * radius_stepsize, 3)[:num_mic-1], 0)
    pos_array_angle = np.append(np.tile(np.pi * 2 * np.arange(3) / 3., 3)[:num_mic-1], 0)
    pos_array_angle += np.random.rand() * np.pi / 3.
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


# def gen_visi_inner(alphak_loop, p_mic_x_loop, p_mic_y_loop, xk, yk):
#     num_mic = p_mic_x_loop.size
#     visi_qqp = np.zeros(num_mic ** 2, dtype=complex)
#     count_inner = 0
#     for q in xrange(num_mic):
#         p_x_outer = p_mic_x_loop[q]
#         p_y_outer = p_mic_y_loop[q]
#         for qp in xrange(num_mic):
#             p_x_qqp = p_x_outer - p_mic_x_loop[qp]  # a scalar
#             p_y_qqp = p_y_outer - p_mic_y_loop[qp]  # a scalar
#             visi_qqp[count_inner] =  np.dot(-1j * (xk * p_x_qqp + yk * p_y_qqp),
#                                             alphak_loop)
#             count_inner += 1
#     return visi_qqp


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
    # we transpose the matrix because the function extract will first flatten the matrix
    # withe ordering convention 'C' instead of 'F'!!
    extract_cond = np.reshape((1 - np.eye(num_mic)).T.astype(bool), (-1, 1), order='F')
    visi_noisy = np.reshape(np.extract(extract_cond, visi_noisy.T), (-1, 1), order='F')
    visi_noiseless_off_diag = \
        np.reshape(np.extract(extract_cond, visi_noiseless.T), (-1, 1), order='F')
    # calculate the equivalent SNR
    noise = visi_noisy - visi_noiseless_off_diag
    P = 20 * np.log10(linalg.norm(visi_noiseless_off_diag) / linalg.norm(noise))
    return visi_noisy, P, noise, visi_noiseless_off_diag


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


def mtx_freq2visi_ri(M, p_mic_x, p_mic_y, D1, D2):
    """

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
    D2 = np.vstack((D0, -D0[1::, ::-1]))
    D2 = D2[:, :-1]
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
                b_opt = b_recon_ri
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt, min_error, b_opt, ini

if __name__ == '__main__':
    K = 3
    M = 4
    L = 2 * M + 1
    radius_array = 0.1
    num_mic = 5

    alpha_ks, phi_ks = \
        gen_diracs_param(K, positive_amp=True, log_normal_amp=False,
                         semicircle=True, save_param=False)[:2]

    p_mic_x, p_mic_y = gen_mic_array_2d(radius_array, num_mic, save_layout=False)[:2]
    visi_noiseless = \
        add_noise(gen_visibility(alpha_ks, phi_ks, p_mic_x, p_mic_y),
                  10, num_mic, Ns=256)[3]
    a_ri = np.concatenate((visi_noiseless.real, visi_noiseless.imag))

    D1, D2 = hermitian_expan(M + 1)
    G = mtx_freq2visi_ri(M, p_mic_x, p_mic_y, D1, D2)

    ms = np.reshape(np.arange(-M, M + 1, step=1), (-1, 1), order='F')
    b_noiseless = np.dot(np.exp(-1j * ms * np.reshape(phi_ks, (1, -1), order='F')),
                         np.reshape(alpha_ks, (-1, 1), order='F'))

    b_ri = np.concatenate((np.real(b_noiseless[:M+1]), np.imag(b_noiseless[:M])))
    Gb = np.dot(G, b_ri)
    Gb_cpx = Gb[:num_mic*(num_mic -1)] + 1j *Gb[num_mic*(num_mic -1):]
    
    plt.plot(p_mic_x, p_mic_y, 'x')
    plt.axis('image')
    plt.xlim([-radius_array, radius_array])
    plt.ylim([-radius_array, radius_array])
    plt.show()
    
    plt.figure(figsize=(8, 2), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(np.real(visi_noiseless), 'r')
    plt.plot(np.real(Gb_cpx), 'b', hold=True)
    plt.subplot(1, 2, 2)
    plt.plot(np.imag(visi_noiseless), 'r')
    plt.plot(np.imag(Gb_cpx), 'b', hold=True)
    plt.show()

    beta_opt = b_ri
    noise_level = 1e-10
    c_opt, min_error, b_opt, ini =\
        dirac_recon_ri(G, a_ri, K, M, noise_level, max_ini=100, stop_cri='mse')
    
    print np.sort(phi_ks)
    print np.sort(np.mod(np.real(np.log(np.roots(c_opt))/(-1j)), 2* np.pi))
