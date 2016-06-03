from __future__ import division

import numpy as np

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

