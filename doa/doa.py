# Author: Eric Bezzam
# Date: Feb 15, 2016

"""Direction of Arrival (DoA) estimation."""

import numpy as np
import math, sys
import warnings
from abc import ABCMeta, abstractmethod

try:
    import matplotlib as mpl
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

if matplotlib_available:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

tol = 1e-14

class DOA(object):
    """

    Parent class for Direction of Arrival (DoA) algorithms. After creating an object (SRP, MUSIC, CSSM, WAVES, or TOPS), run locate_sources() to apply the corresponding algorithm.

    .. note:: Think of this as a virtual class in C++. You should not create a DOA object!

    :param L: Microphone array positions. Each column should correspond to the cartesian coordinates of a single microphone.
    :type L: numpy array
    :param fs: Sampling frequency.
    :type fs: float
    :param nfft: FFT length.
    :type nfft: int
    :param c: Speed of sound.
    :type c: float
    :param num_sources: Number of sources to detect. Default is 1.
    :type num_sources: int
    :param mode: 'far' (default) or 'near' for far-field or near-field detection respectively.
    :type mode: str
    :param r: Candidate distances from the origin. Default is r = np.ones(1) corresponding to far-field.
    :type r: numpy array
    :param theta: Candidate azimuth angles (in radians) with respect to x-axis.
    :type theta: numpy array
    :param phi: Candidate elevation angles (in radians) with respect to z-axis. Default value is phi = pi/2 as to search on the xy plane.
    :type phi: numpy array

    """

    __metaclass__ = ABCMeta


    def __init__(self, L, fs, nfft, c, num_sources, mode, theta, r=None, phi=None):

        self.M = L.shape[1]     # number of microphones
        self.D = L.shape[0]     # number of dimensions (x,y,z)
        self.fs = fs            # sampling frequency
        self.L = L              # locations of mics
        self.c = c              # speed of sound

        self.nfft = nfft        # FFT size
        self.nbin = self.nfft/2+1
        self.freq = None
        self.num_freq = None

        self.num_sources = self._check_num_sources(num_sources)
        self.sources = np.zeros([self.D, self.num_sources])
        self.src_idx = np.zeros(self.num_sources, dtype=np.int)
        self.phi_recon = None

        # default is azimuth search on xy plane for far-field search
        self.mode = mode
        if self.mode is 'far': 
            self.r = np.ones(1)
        elif r is None:
            self.r = np.ones(1)
            self.mode = 'far'
        else:
            self.r = r
            if r == np.ones(1):
                mode = 'far'
        if theta is None:
            self.theta = np.linspace(-180.,180.,30)*np.pi/180
        else:  
            self.theta = theta
        if phi is None:
            self.phi = np.pi/2*np.ones(1)
        else:
            self.phi = phi

        # spatial spectrum / dirty image
        self.P = None

        # build lookup table to candidate locations from r, theta, phi 
        from fri import FRI
        if not isinstance(self, FRI):
            self.loc = None
            self.num_loc = None
            self.build_lookup()
            # compute mode vectors
            self.mode_vec = None
            self.compute_mode()
        else:
            self.num_loc = len(self.theta)

    def locate_sources(self, X, num_sources=None, freq_range=[500.0, 4000.0], freq = None):
        """
        Locate source(s) using corresponding algorithm.

        :param X: Set of signals in the frequency (RFFT) domain for current frame. Size should be M x F x S, where M should correspond to the number of microphones, F to nfft/2+1, and S is the number of snapshots (user-defined). It is recommended to have S >> M.
        :type X: 3D numpy array
        :param num_sources: Number of sources to detect. Default is value given to object constructor.
        :type num_sources: int
        :param freq_range: Frequency range on which to run DoA: [fmin, fmax].
        :type freq_range: list of floats, length 2
        :param freq: List of individual frequencies on which to run DoA. If defined by user, it will **not** take into consideration freq_range.
        :type freq: list of floats
        """

        # check validity of inputs
        if num_sources is not None and num_sources != self.num_sources:
            self.num_sources = self._check_num_sources(num_sources)
            self.sources = np.zeros([self.num_sources, self.D])
            self.src_idx = np.zeros(self.num_sources, dtype=np.int)
            self.angle_of_arrival = None
        if (X.shape[0] != self.M):
            raise ValueError('Number of signals (rows) does not match the number of microphones.')
        if (X.shape[1] != self.nbin):
            raise ValueError("Mismatch in FFT length.")

        # frequency bins on which to apply DOA
        if freq is not None:
            self.freq = [int(np.round(f/self.fs*self.nfft)) for f in freq]
        else:
            freq_range = [int(np.round(f/self.fs*self.nfft)) for f in freq_range]
            self.freq = np.arange(freq_range[0],freq_range[1])
        self.freq = self.freq[self.freq<self.nbin]
        self.freq = self.freq[self.freq>=0]
        self.num_freq = len(self.freq)

        # search for DoA according to desired algorithm
        self.P = np.zeros(self.num_loc)
        self._process(X)

        # locate sources
        if self.phi_recon is None:  # not FRI
            self._peaks1D()


    def polar_plt_dirac(self, phi_ref, save_fig=False, file_name=None, plt_dirty_img=True):

        phi_recon = self.phi_recon
        num_mic = self.M
        phi_plt = self.theta
        dirty_img = self.P

        # make amplitudes and angles line up
        order = np.argsort(phi_recon)
        phi_ref = np.sort(phi_ref)
        phi_recon = np.sort(phi_recon)
        alpha_recon = self.P[self.src_idx][order]
        alpha_ref = alpha_recon # might need alpha_ref has same size as phi_ref

        if max(self.P)==0:  # FRI
            dirty_img = self.gen_dirty_img()

        fig = plt.figure(figsize=(5, 4), dpi=90)
        ax = fig.add_subplot(111, projection='polar')
        K = phi_ref.size
        K_est = phi_recon.size

        ax.scatter(phi_ref, 1 + alpha_ref, c=np.tile([0, 0.447, 0.741], (K, 1)), s=70, alpha=0.75, marker='^', linewidths=0, label='original')
        ax.scatter(phi_recon, 1 + alpha_recon, c=np.tile([0.850, 0.325, 0.098], (K_est, 1)), s=100, alpha=0.75, marker='*', linewidths=0, label='reconstruction')
        for k in xrange(K):
            ax.plot([phi_ref[k], phi_ref[k]], [1, 1 + alpha_ref[k]], linewidth=1.5, linestyle='-', color=[0, 0.447, 0.741], alpha=0.6)
        for k in xrange(K_est):
            ax.plot([phi_recon[k], phi_recon[k]], [1, 1 + alpha_recon[k]], linewidth=1.5, linestyle='-', color=[0.850, 0.325, 0.098], alpha=0.6)

        if plt_dirty_img:
            dirty_img = dirty_img.real
            min_val = dirty_img.min()
            max_val = dirty_img.max()
            ax.plot(phi_plt, 1 + dirty_img, linewidth=1, alpha=0.55,
                    linestyle='-', color=[0.466, 0.674, 0.188], label='spatial spectrum')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:3], framealpha=0.5,
                  scatterpoints=1, loc=8, fontsize=9,
                  ncol=1, bbox_to_anchor=(0.9, -0.17),
                  handletextpad=.2, columnspacing=1.7, labelspacing=0.1)
        # title_str = r'$K={0}$, $\mbox{{\# of mic.}}={1}$, $\mbox{{SNR}}={2:.1f}$dB, average error={3:.1e}'
        # ax.set_title(title_str.format(repr(K), repr(num_mic), P, dist_recon),
        #              fontsize=11)
        ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
        ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
        ax.set_ylim([0, 1.05 + max_val])
        # if plt_dirty_img:
        #     ax.set_ylim([0, 1.05 + np.max(np.append(np.concatenate((alpha_ref, alpha_recon)), max_val))])
        # else:
        #     ax.set_ylim([0, 1.05 + np.max(np.concatenate((alpha_ref, alpha_recon)))])
        if save_fig:
            if file_name is None:
                file_name = 'polar_recon_dirac.pdf'
            plt.savefig(file_name, format='pdf', dpi=300, transparent=True)

    def build_lookup(self, r=None, theta=None, phi=None):
        """
        Construct lookup table for given candidate locations (in spherical coordinates). Each row is a location in cartesian coordinates.

        :param r: Candidate distances from the origin.
        :type r: numpy array
        :param theta: Candidate azimuth angles with respect to x-axis.
        :type theta: numpy array
        :param phi: Candidate elevation angles with respect to z-axis.
        :type phi: numpy array
        """
        if theta is not None:
            self.theta = theta
        if phi is not None:
            self.phi = phi
        if r is not None:
            self.r = r
            if self.r == np.ones(1):
                self.mode = 'far'
            else:
                self.mode = 'near'
        self.loc = np.zeros([self.D,len(self.r)*len(self.theta)*len(self.phi)])
        self.num_loc = self.loc.shape[1]
        # convert to cartesian
        for i in range(len(self.r)):
            r_s = self.r[i]
            for j in range(len(self.theta)):
                theta_s = self.theta[j]
                for k in range(len(self.phi)):
                    # spher = np.array([r_s,theta_s,self.phi[k]])
                    self.loc[:,i*len(self.theta)+j*len(self.phi)+k] = \
                        spher2cart(r_s, theta_s, self.phi[k])[0:self.D]

    def compute_mode(self):
        """
        Pre-compute mode vectors from candidate locations (in spherical 
        coordinates).

        .. note:: build_lookup() **must** be run beforehand.
        """
        self.mode_vec = np.empty((self.nbin,self.M,self.num_loc), 
            dtype='complex64')
        if (self.nfft % 2 == 1):
            raise ValueError('Signal length must be even.')
        f = 1.0/self.nfft*np.linspace(0,self.nfft/2,self.nbin)*1j*2*np.pi
        for i in range(self.num_loc):
            p_s = self.loc[:,i]
            for m in range(self.M):
                p_m = self.L[:,m]
                if(self.mode=='near'):
                    dist = np.linalg.norm(p_m-p_s, axis=1)
                if(self.mode=='far'):
                    dist = np.dot(p_s,p_m)
                # tau = np.round(self.fs*dist/self.c) # discrete - jagged
                tau = self.fs*dist/self.c           # "continuous" - smoother
                self.mode_vec[:,m,i] = np.exp(f*tau)

    def _check_num_sources(self, num_sources):
        """
        Check validity of number of sources.

        Parameters
        -----------
        num_sources : int
            Number of desired sources to detect.
        
        Returns
        -----------
        valid : int
            Valid number of sources that can be detected.
        """

        # check validity of inputs
        if num_sources > self.M:
            warnings.warn('Number of sources cannot be more than number of microphones. Changing number of sources to ' + str(self.M) + '.')
            num_sources = self.M
        if num_sources < 1:
            warnings.warn('Number of sources must be at least 1. Changing number of sources to 1.')
            num_sources = 1
        valid = num_sources
        return valid


    def _peaks1D(self):
        if self.num_sources==1:
            self.src_idx[0] = np.argmax(self.P)
            self.sources[:,0] = self.loc[:,self.src_idx[0]]
            self.phi_recon = self.theta[self.src_idx[0]]
        else:
            peak_idx = []
            for i in range(self.num_loc):
                if i==(self.num_loc-1):
                    if self.P[i]>self.P[i-1] and self.P[i]>self.P[0]:
                        peak_idx.append(i)  
                else:
                    if self.P[i]>self.P[i-1] and self.P[i]>self.P[i+1]:
                        peak_idx.append(i)
            peaks = self.P[peak_idx]
            max_idx = np.argsort(peaks)[-self.num_sources:]
            self.src_idx = [peak_idx[k] for k in max_idx]
            self.sources = self.loc[:,self.src_idx]
            self.phi_recon = self.theta[self.src_idx]

#------------------Miscellaneous Functions---------------------#

def spher2cart(r, theta, phi):
    """
    Convert a spherical point to cartesian coordinates.
    """
    # convert to cartesian
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return np.array([x, y, z])
    