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

from utils import polar2cart


def cov_mtx_est(y_mic):
    """
    estimate covariance matrix
    :param y_mic: received signal (complex based band representation) at microphones
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


def mtx_fri2visi_ri_multiband(M, p_mic_x_all, p_mic_y_all, D1, D2):
    """
    build the matrix that maps the Fourier series to the visibility in terms of
    REAL-VALUED entries only. (matrix size double)
    :param M: the Fourier series expansion is limited from -M to M
    :param p_mic_x_all: a matrix that contains microphones x coordinates
    :param p_mic_y_all: a matrix that contains microphones y coordinates
    :param D1: expansion matrix for the real-part
    :param D2: expansion matrix for the imaginary-part
    :return:
    """
    num_bands = p_mic_x_all.shape[1]
    return linalg.block_diag(*[mtx_fri2visi_ri(M, p_mic_x_all[:, band_count],
                                              p_mic_y_all[:, band_count], D1, D2)
                              for band_count in xrange(num_bands)])


def mtx_fri2visi_ri(M, p_mic_x, p_mic_y, D1, D2):
    """
    build the matrix that maps the Fourier series to the visibility in terms of
    REAL-VALUED entries only. (matrix size double)
    :param M: the Fourier series expansion is limited from -M to M
    :param p_mic_x: a vector that contains microphones x coordinates
    :param p_mic_y: a vector that contains microphones y coordinates
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


def output_shrink(K, L):
    """
    shrink the convolution output to half the size.
    used when both the annihilating filter and the uniform samples of sinusoids satisfy
    Hermitian symmetric.
    :param K: the annihilating filter size: K + 1
    :param L: length of the (complex-valued) b vector
    :return:
    """
    out_len = L - K
    if out_len % 2 == 0:
        half_out_len = np.int(out_len / 2.)
        mtx_r = np.hstack((np.eye(half_out_len),
                           np.zeros((half_out_len, half_out_len))))
        mtx_i = mtx_r
    else:
        half_out_len = np.int((out_len + 1) / 2.)
        mtx_r = np.hstack((np.eye(half_out_len),
                           np.zeros((half_out_len, half_out_len - 1))))
        mtx_i = np.hstack((np.eye(half_out_len - 1),
                           np.zeros((half_out_len - 1, half_out_len))))
    return linalg.block_diag(mtx_r, mtx_i)


def coef_expan_mtx(K):
    """
    expansion matrix for an annihilating filter of size K + 1
    :param K: number of Dirac. The filter size is K + 1
    :return:
    """
    if K % 2 == 0:
        D0 = np.eye(np.int(K / 2. + 1))
        D1 = np.vstack((D0, D0[1:, ::-1]))
        D2 = np.vstack((D0, -D0[1:, ::-1]))[:, :-1]
    else:
        D0 = np.eye(np.int((K + 1) / 2.))
        D1 = np.vstack((D0, D0[:, ::-1]))
        D2 = np.vstack((D0, -D0[:, ::-1]))
    return D1, D2


def Tmtx_ri(b_ri, K, D, L):
    """
    build convolution matrix associated with b_ri
    :param b_ri: a real-valued vector
    :param K: number of Diracs
    :param D1: expansion matrix for the real-part
    :param D2: expansion matrix for the imaginary-part
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


def Tmtx_ri_half_out_half(b_ri, K, D, L, D_coef, mtx_shrink):
    """
    if both b and annihilation filter coefficients are Hermitian symmetric, then
    the output will also be Hermitian symmetric => the effectively output is half the size
    """
    return np.dot(np.dot(mtx_shrink, Tmtx_ri(b_ri, K, D, L)), D_coef)


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
    return Rmtx_ri(np.dot(D_coef, coef_half), K, D, L)


def Rmtx_ri_half_out_half(coef_half, K, D, L, D_coef, mtx_shrink):
    """
    if both b and annihilation filter coefficients are Hermitian symmetric, then
    the output will also be Hermitian symmetric => the effectively output is half the size
    """
    return np.dot(mtx_shrink, Rmtx_ri(np.dot(D_coef, coef_half), K, D, L))


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


def build_mtx_amp_ri(p_mic_x, p_mic_y, phi_k):
    mtx = build_mtx_amp(phi_k, p_mic_x, p_mic_y)
    return np.vstack((mtx.real, mtx.imag))


def mtx_updated_G_multiband(phi_recon, M, mtx_amp2visi_ri,
                            mtx_fri2visi_ri, num_bands):
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
    G_updated = np.dot(mtx_amp2visi_ri,
                       linalg.block_diag(*([mtx_fri2amp_ri] * num_bands))
                       ) + \
                np.dot(mtx_fri2visi_ri,
                       linalg.block_diag(*([mtx_null_proj] * num_bands))
                       )
    return G_updated


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


# def dirac_recon_ri_half_parallel(G, a_ri, K, M, max_ini=100):
#     """
#     Reconstruct point sources' locations (azimuth) from the visibility measurements.
#     Here we enforce hermitian symmetry in the annihilating filter coefficients so that
#     roots on the unit circle are encouraged.
#     We use parallel implementation when stop_cri == 'max_iter'
#     :param G: the linear transformation matrix that links the visibilities to
#                 uniformly sampled sinusoids
#     :param a_ri: the visibility measurements
#     :param K: number of Diracs
#     :param M: the Fourier series expansion is between -M and M
#     :param noise_level: level of noise (ell_2 norm) in the measurements
#     :param max_ini: maximum number of initialisations
#     :param stop_cri: stopping criterion, either 'mse' or 'max_iter'
#     :return:
#     """
#     L = 2 * M + 1  # length of the (complex-valued) b vector
#     a_ri = a_ri.flatten('F')
#     # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
#     assert not np.iscomplexobj(G)  # G should be real-valued
#     Gt_a = np.dot(G.T, a_ri)
#     # maximum number of iterations with each initialisation
#     max_iter = 50
#
#     # the least-square solution
#     beta_ri = linalg.lstsq(G, a_ri)[0]
#     D1, D2 = hermitian_expan(M + 1)
#     D = linalg.block_diag(D1, D2)
#     D_coef1, D_coef2 = coef_expan_mtx(K)
#     D_coef = linalg.block_diag(D_coef1, D_coef2)
#
#     # size of Tbeta_ri: 2(L - K) x 2(K + 1)
#     Tbeta_ri = Tmtx_ri_half(beta_ri, K, D, L, D_coef)
#
#     # size of various matrices / vectors
#     sz_G1 = L
#
#     sz_Tb0 = 2 * (L - K)
#
#     sz_Rc0 = 2 * (L - K)
#
#     sz_coef = K + 1
#
#     rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
#     rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))
#
#     # the main iteration with different random initialisations
#     partial_dirac_recon = partial(dirac_recon_ri_inner, a_ri=a_ri, rhs=rhs,
#                                   rhs_bl=rhs_bl, K=K, M=M, D1=D1, D2=D2, D_coef=D_coef,
#                                   Tbeta_ri=Tbeta_ri, G=G, max_iter=max_iter)
#
#     # generate all the random initialisations
#     c_ri_half_all = np.random.randn(sz_coef, max_ini)
#
#     res_all = Parallel(n_jobs=-1)(
#         delayed(partial_dirac_recon)(c_ri_half_all[:, loop][:, np.newaxis])
#         for loop in xrange(max_ini))
#
#     # find the one with smallest error
#     min_idx = np.array(zip(*res_all)[1]).argmin()
#     c_opt, min_error, b_opt = res_all[min_idx]
#
#     return c_opt, min_error, b_opt
#
#
# def dirac_recon_ri_inner(c_ri_half, a_ri, rhs, rhs_bl, K, M,
#                          D1, D2, D_coef, Tbeta_ri, G, max_iter):
#     min_error = float('inf')
#     # size of various matrices / vectors
#     L = 2 * M + 1  # length of the (complex-valued) b vector
#
#     sz_Tb0 = 2 * (L - K)
#     sz_Tb1 = K + 1
#
#     sz_Rc0 = 2 * (L - K)
#     sz_Rc1 = L
#
#     sz_coef = K + 1
#     sz_bri = L
#
#     GtG = np.dot(G.T, G)
#     D = linalg.block_diag(D1, D2)
#
#     c0_ri_half = c_ri_half.copy()
#     error_seq = np.zeros(max_iter, dtype=float)
#     R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
#
#     # first row of mtx_loop
#     mtx_loop_first_row = np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta_ri.T,
#                                     np.zeros((sz_coef, sz_Rc1)), c0_ri_half))
#     # last row of mtx_loop
#     mtx_loop_last_row = np.hstack((c0_ri_half.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))))
#
#     for inner in xrange(max_iter):
#         mtx_loop = np.vstack((mtx_loop_first_row,
#                               np.hstack((Tbeta_ri,
#                                          np.zeros((sz_Tb0, sz_Tb0)),
#                                          -R_loop,
#                                          np.zeros((sz_Rc0, 1))
#                                          )),
#                               np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
#                                          -R_loop.T,
#                                          GtG,
#                                          np.zeros((sz_Rc1, 1))
#                                          )),
#                               mtx_loop_last_row
#                               ))
#         # matrix should be symmetric
#         # mtx_loop = (mtx_loop + mtx_loop.T) / 2.
#         mtx_loop += mtx_loop.T
#         mtx_loop *= 0.5
#         c_ri_half = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]
#
#         R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
#         mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
#                                 np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
#                                 ))
#         # mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
#         mtx_brecon += mtx_brecon.T
#         mtx_brecon *= 0.5
#         b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]
#
#         error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
#         if error_seq[inner] < min_error:
#             min_error = error_seq[inner]
#             b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
#             c_ri = np.dot(D_coef, c_ri_half)
#             c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts
#     return c_opt, min_error, b_opt


def dirac_recon_ri_half_multiband(G, a_ri, K, M, max_ini=100):
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
    num_bands = a_ri.shape[1]  # number of bands considered
    L = 2 * M + 1  # length of the (complex-valued) b vector for each band
    a_ri = a_ri.flatten('F')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50

    # the least-square solution
    beta_ri = np.reshape(linalg.lstsq(G, a_ri)[0], (-1, num_bands), order='F')
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink = output_shrink(K, L)

    # size of Tbeta_ri: (L - K)num_bands x 2(K + 1)
    Tbeta_ri = np.vstack([Tmtx_ri_half_out_half(beta_ri[:, band_count],
                                                K, D, L, D_coef, mtx_shrink)
                          for band_count in range(num_bands)])

    # size of various matrices / vectors
    sz_G1 = L * num_bands

    sz_Tb0 = (L - K) * num_bands  # the only effective size because of Hermitian symmetry

    sz_Rc0 = (L - K) * num_bands

    sz_coef = K + 1

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=float)))

    min_error = float('inf')
    # size of various matrices / vectors
    L = 2 * M + 1  # length of the (complex-valued) b vector

    sz_Tb0 = (L - K) * num_bands  # the only effective size because of Hermitian symmetry
    sz_Tb1 = K + 1

    sz_Rc0 = (L - K) * num_bands
    sz_Rc1 = L * num_bands

    sz_coef = K + 1
    sz_bri = L * num_bands

    GtG = np.dot(G.T, G)
    D = linalg.block_diag(D1, D2)

    for ini_loop in range(max_ini):
        c_ri_half = np.random.randn(sz_coef)
        c0_ri_half = c_ri_half.copy()[:, np.newaxis]

        Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
        R_loop = linalg.block_diag(*([Rmtx_band] * num_bands))

        # first row of mtx_loop
        mtx_loop_first_row = np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta_ri.T,
                                        np.zeros((sz_coef, sz_Rc1)), c0_ri_half))
        # last row of mtx_loop
        mtx_loop_last_row = np.hstack((c0_ri_half.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))))

        for inner in range(max_iter):
            mtx_loop = np.vstack((mtx_loop_first_row,
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
                                  mtx_loop_last_row
                                  ))

            # matrix should be symmetric
            mtx_loop += mtx_loop.T
            mtx_loop *= 0.5
            c_ri_half = linalg.solve(mtx_loop, rhs)[:sz_coef]

            Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
            R_loop = linalg.block_diag(*([Rmtx_band] * num_bands))

            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                    ))

            mtx_brecon += mtx_brecon.T
            mtx_brecon *= 0.5
            b_recon_ri = linalg.solve(mtx_brecon, rhs_bl)[:sz_bri]

            error_seq_loop = linalg.norm(a_ri - np.dot(G, b_recon_ri))
            if error_seq_loop < min_error:
                min_error = error_seq_loop
                b_recon_ri = np.reshape(b_recon_ri, (-1, num_bands), order='F')
                b_opt = np.dot(D1, b_recon_ri[:M + 1, :]) + \
                        1j * np.dot(D2, b_recon_ri[M + 1:, :])
                c_ri = np.dot(D_coef, c_ri_half)
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts

    return c_opt, min_error, b_opt


def dirac_recon_ri_half_multiband_parallel(G, a_ri, K, M, max_ini=100, use_parallel=False):
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
    num_bands = a_ri.shape[1]  # number of bands considered
    L = 2 * M + 1  # length of the (complex-valued) b vector for each band
    a_ri = a_ri.flatten('F')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50

    # the least-square solution
    beta_ri = np.reshape(linalg.lstsq(G, a_ri)[0], (-1, num_bands), order='F')
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink = output_shrink(K, L)

    # size of Tbeta_ri: (L - K)num_bands x 2(K + 1)
    Tbeta_ri = np.vstack([Tmtx_ri_half_out_half(beta_ri[:, band_count],
                                                K, D, L, D_coef, mtx_shrink)
                          for band_count in xrange(num_bands)])

    # size of various matrices / vectors
    sz_G1 = L * num_bands

    sz_Tb0 = (L - K) * num_bands  # the only effective size because of Hermitian symmetry

    sz_Rc0 = (L - K) * num_bands

    sz_coef = K + 1

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=float)))

    # the main iteration with different random initialisations
    partial_dirac_recon = partial(dirac_recon_ri_multiband_inner,
                                  a_ri=a_ri, num_bands=num_bands, rhs=rhs,
                                  rhs_bl=rhs_bl, K=K, M=M, D1=D1, D2=D2,
                                  D_coef=D_coef, mtx_shrink=mtx_shrink,
                                  Tbeta_ri=Tbeta_ri, G=G, max_iter=max_iter)

    # generate all the random initialisations
    c_ri_half_all = np.random.randn(sz_coef, max_ini)

    if use_parallel:
        res_all = Parallel(n_jobs=-1)(
            delayed(partial_dirac_recon)(c_ri_half_all[:, loop][:, np.newaxis])
            for loop in xrange(max_ini))
    else:
        res_all = []
        for loop in range(max_ini):
            res_all.append(
                partial_dirac_recon(c_ri_half_all[:, loop][:, np.newaxis])
            )

    # find the one with smallest error
    min_idx = np.array(zip(*res_all)[1]).argmin()
    c_opt, min_error, b_opt = res_all[min_idx]

    return c_opt, min_error, b_opt


def dirac_recon_ri_multiband_inner(c_ri_half, a_ri, num_bands, rhs, rhs_bl, K, M,
                                   D1, D2, D_coef, mtx_shrink, Tbeta_ri, G, max_iter):
    min_error = float('inf')
    # size of various matrices / vectors
    L = 2 * M + 1  # length of the (complex-valued) b vector

    sz_Tb0 = (L - K) * num_bands  # the only effective size because of Hermitian symmetry
    sz_Tb1 = K + 1

    sz_Rc0 = (L - K) * num_bands
    sz_Rc1 = L * num_bands

    sz_coef = K + 1
    sz_bri = L * num_bands

    GtG = np.dot(G.T, G)
    D = linalg.block_diag(D1, D2)

    c0_ri_half = c_ri_half.copy()
    error_seq = np.zeros(max_iter, dtype=float)

    Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
    R_loop = linalg.block_diag(*([Rmtx_band] * num_bands))

    # first row of mtx_loop
    mtx_loop_first_row = np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta_ri.T,
                                    np.zeros((sz_coef, sz_Rc1)), c0_ri_half))
    # last row of mtx_loop
    mtx_loop_last_row = np.hstack((c0_ri_half.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))))

    for inner in xrange(max_iter):
        mtx_loop = np.vstack((mtx_loop_first_row,
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
                              mtx_loop_last_row
                              ))

        # matrix should be symmetric
        mtx_loop += mtx_loop.T
        mtx_loop *= 0.5
        c_ri_half = linalg.solve(mtx_loop, rhs)[:sz_coef]

        Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
        R_loop = linalg.block_diag(*([Rmtx_band] * num_bands))

        mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                ))

        mtx_brecon += mtx_brecon.T
        mtx_brecon *= 0.5
        b_recon_ri = linalg.solve(mtx_brecon, rhs_bl)[:sz_bri]

        error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
        if error_seq[inner] < min_error:
            min_error = error_seq[inner]
            b_recon_ri = np.reshape(b_recon_ri, (-1, num_bands), order='F')
            b_opt = np.dot(D1, b_recon_ri[:M + 1, :]) + \
                    1j * np.dot(D2, b_recon_ri[M + 1:, :])
            c_ri = np.dot(D_coef, c_ri_half)
            c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts
    return c_opt, min_error, b_opt


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

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink = output_shrink(K, L)

    # size of Tbeta_ri: (L - K) x 2(K + 1)
    Tbeta_ri = Tmtx_ri_half_out_half(beta_ri, K, D, L, D_coef, mtx_shrink)

    # size of various matrices / vectors
    sz_G1 = L

    sz_Tb0 = L - K  # the only effective size because of Hermitian symmetry

    sz_Rc0 = L - K

    sz_coef = K + 1

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))

    # the main iteration with different random initialisations
    partial_dirac_recon = partial(dirac_recon_ri_inner, a_ri=a_ri, rhs=rhs,
                                  rhs_bl=rhs_bl, K=K, M=M, D1=D1, D2=D2,
                                  D_coef=D_coef, mtx_shrink=mtx_shrink,
                                  Tbeta_ri=Tbeta_ri, G=G, max_iter=max_iter)

    # generate all the random initialisations
    c_ri_half_all = np.random.randn(sz_coef, max_ini)

    res_all = Parallel(n_jobs=1)(
        delayed(partial_dirac_recon)(c_ri_half_all[:, loop][:, np.newaxis])
        for loop in xrange(max_ini))

    # find the one with smallest error
    min_idx = np.array(zip(*res_all)[1]).argmin()
    c_opt, min_error, b_opt = res_all[min_idx]

    return c_opt, min_error, b_opt


def dirac_recon_ri_inner(c_ri_half, a_ri, rhs, rhs_bl, K, M,
                         D1, D2, D_coef, mtx_shrink, Tbeta_ri, G, max_iter):
    min_error = float('inf')
    # size of various matrices / vectors
    L = 2 * M + 1  # length of the (complex-valued) b vector

    sz_Tb0 = L - K  # the only effective size because of Hermitian symmetry
    sz_Tb1 = K + 1

    sz_Rc0 = L - K
    sz_Rc1 = L

    sz_coef = K + 1
    sz_bri = L

    GtG = np.dot(G.T, G)
    D = linalg.block_diag(D1, D2)

    c0_ri_half = c_ri_half.copy()
    error_seq = np.zeros(max_iter, dtype=float)
    R_loop = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)

    # first row of mtx_loop
    mtx_loop_first_row = np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta_ri.T,
                                    np.zeros((sz_coef, sz_Rc1)), c0_ri_half))
    # last row of mtx_loop
    mtx_loop_last_row = np.hstack((c0_ri_half.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))))

    for inner in xrange(max_iter):
        mtx_loop = np.vstack((mtx_loop_first_row,
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
                              mtx_loop_last_row
                              ))
        # matrix should be symmetric
        # mtx_loop = (mtx_loop + mtx_loop.T) / 2.
        mtx_loop += mtx_loop.T
        mtx_loop *= 0.5
        # c_ri_half = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]
        c_ri_half = linalg.solve(mtx_loop, rhs)[:sz_coef]

        R_loop = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
        mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                ))
        # mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
        mtx_brecon += mtx_brecon.T
        mtx_brecon *= 0.5
        # b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]
        b_recon_ri = linalg.solve(mtx_brecon, rhs_bl)[:sz_bri]

        error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
        if error_seq[inner] < min_error:
            min_error = error_seq[inner]
            b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
            c_ri = np.dot(D_coef, c_ri_half)
            c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts
    return c_opt, min_error, b_opt


def pt_src_recon_multiband(a, p_mic_x, p_mic_y, omega_bands, sound_speed,
                           K, M, noise_level, max_ini=50,
                           update_G=False, verbose=False, **kwargs):
    """
    reconstruct point sources on the circle from the visibility measurements
    from multi-bands.
    :param a: the measured visibilities in a matrix form, where the second dimension
              corresponds to different subbands
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param omega_bands: mid-band (ANGULAR) frequencies [radian/sec]
    :param sound_speed: speed of sound
    :param K: number of point sources
    :param M: the Fourier series expansion is between -M to M
    :param noise_level: noise level in the measured visibilities
    :param max_ini: maximum number of random initialisation used
    :param update_G: update the linear mapping that links the uniformly sampled
                sinusoids to the visibility or not.
    :param verbose: whether output intermediate results for debugging or not
    :param kwargs: possible optional input: G_iter: number of iterations for the G updates
    :return:
    """
    p_mic_x = np.squeeze(p_mic_x)
    p_mic_y = np.squeeze(p_mic_y)

    num_bands = np.array(omega_bands).size  # number of bands considered
    num_mic = p_mic_x.size

    assert M >= K
    assert num_mic * (num_mic - 1) >= 2 * M + 1
    if len(a.shape) == 2:
        assert a.shape[1] == num_bands

    a_ri = np.row_stack((a.real, a.imag))

    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    if 'use_parallel' in kwargs:
        use_parallel = kwargs['use_parallel']
    else:
        use_parallel = False

    # initialisation
    min_error = float('inf')
    phik_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # expansion matrices to take Hermitian symmetry into account
    norm_factor = np.reshape(sound_speed / omega_bands, (1, -1), order='F')
    # normalised antenna coordinates in a matrix form
    # each column corresponds to the location in one subband,
    # i.e., the second dimension corresponds to different subbands
    p_mic_x_normalised = np.reshape(p_mic_x, (-1, 1), order='F') / norm_factor
    p_mic_y_normalised = np.reshape(p_mic_y, (-1, 1), order='F') / norm_factor

    D1, D2 = hermitian_expan(M + 1)
    G = mtx_fri2visi_ri_multiband(M, p_mic_x_normalised, p_mic_y_normalised, D1, D2)

    for loop_G in range(max_loop_G):
        c_recon, error_recon = \
            dirac_recon_ri_half_multiband_parallel(G, a_ri, K, M, max_ini, use_parallel=use_parallel)[:2]

        if verbose:
            print('noise level: {0:.3e}'.format(noise_level))
            print('objective function value: {0:.3e}'.format(error_recon))
        # recover Dirac locations
        uk = np.roots(np.squeeze(c_recon))
        uk /= np.abs(uk)
        phik_recon = np.mod(-np.angle(uk), 2. * np.pi)

        # use least square to reconstruct amplitudes
        partial_build_mtx_amp = partial(build_mtx_amp_ri, phi_k=phik_recon)
        amp_mtx_ri = \
            linalg.block_diag(
                *Parallel(n_jobs=-1)(
                    delayed(partial_build_mtx_amp)(
                        p_mic_x_normalised[:, band_count],
                        p_mic_y_normalised[:, band_count])
                    for band_count in range(num_bands))
            )

        alphak_recon = sp.optimize.nnls(amp_mtx_ri, a_ri.flatten('F'))[0]
        error_loop = linalg.norm(a_ri.flatten('F') - np.dot(amp_mtx_ri, alphak_recon))

        if error_loop < min_error:
            min_error = error_loop
            phik_opt = phik_recon
            alphak_opt = np.reshape(alphak_recon, (-1, num_bands), order='F')

        # update the linear transformation matrix
        if update_G:
            G = mtx_updated_G_multiband(phik_recon, M, amp_mtx_ri, G, num_bands)

    # convert progagation vector to DOA
    phik_doa = np.mod(phik_opt - np.pi, 2. * np.pi)
    return phik_doa , alphak_opt


def pt_src_recon(a, p_mic_x, p_mic_y, omega_band, sound_speed,
                 K, M, noise_level, max_ini=50,
                 stop_cri='mse', update_G=False, verbose=False, **kwargs):
    """
    reconstruct point sources on the circle from the visibility measurements
    :param a: the measured visibilities
    :param p_mic_x: a vector that contains microphones' x-coordinates
    :param p_mic_y: a vector that contains microphones' y-coordinates
    :param omega_band: mid-band (ANGULAR) frequency [radian/sec]
    :param sound_speed: speed of sound
    :param K: number of point sources
    :param M: the Fourier series expansion is between -M to M
    :param noise_level: noise level in the measured visibilities
    :param max_ini: maximum number of random initialisation used
    :param stop_cri: either 'mse' or 'max_iter'
    :param update_G: update the linear mapping that links the uniformly sampled
                sinusoids to the visibility or not.
    :param verbose: whether output intermediate results for debugging or not
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
    p_mic_x_normalised = p_mic_x / (sound_speed / omega_band)
    p_mic_y_normalised = p_mic_y / (sound_speed / omega_band)
    D1, D2 = hermitian_expan(M + 1)
    G = mtx_fri2visi_ri(M, p_mic_x_normalised, p_mic_y_normalised, D1, D2)
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
        amp_mtx = build_mtx_amp(phik_recon, p_mic_x_normalised, p_mic_y_normalised)
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
    :param num_rotation: number of random rotations
    :param verbose: whether output intermediate results for debugging or not
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
