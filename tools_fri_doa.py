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
# from mayavi import mlab
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
    calculate
    :param r: radius of the sphere
    :param theta1: colatitude of point 1
    :param phi1: azimuth of point 1
    :param theta2: colatitude of point 2
    :param phi2: azimuth of point 2
    :return: great-circle distance
    """
    # theta1 = theta1.flatten('F')
    # phi1 = phi1.flatten('F')
    # theta2 = theta2.flatten('F')
    # phi2 = phi2.flatten('F')
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
                         save_param=False):
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