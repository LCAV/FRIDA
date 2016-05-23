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


def sph_gen_diracs_param(K, positive_amp=True, log_normal_amp=False,
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
    pos_array_norm = np.linspace(0, radius_array, num=num_mic, dtype=float)
    pos_array_angle = np.linspace(0, 12 * np.pi, num=num_mic, dtype=float)
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
    return pos_mic_x, pos_mic_y,layout_time_stamp


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
    # reshape xk, yk to use broadcasting
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')


def gen_visi_inner(alphak_loop, p_mic_x_loop, p_mic_y_loop, xk, yk):
    num_mic = p_mic_x_loop.size
    visi_qqp = np.zeros(num_mic ** 2, dtype=complex)
    count_inner = 0
    for q in xrange(num_mic):
        p_x_outer = p_mic_x_loop[q]
        p_y_outer = p_mic_y_loop[q]
        for qp in xrange(num_mic):
            p_x_qqp = p_x_outer - p_mic_x_loop[qp]
            p_y_qqp = p_y_outer - p_mic_y_loop[qp]
            visi_qqp[count_inner] =  np.dot()
            count_inner += 1
    return visi_qqp
