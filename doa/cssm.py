# Author: Eric Bezzam
# Date: July 15, 2016

from music import *

class CSSM(MUSIC):
    """
    Class to apply the Coherent Signal-Subspace method (CSSM) [H. Wang and M. 
    Kaveh] for Direction of Arrival (DoA) estimation.

    .. note:: Run locate_source() to apply the CSSM algorithm.

    :param L: Microphone array positions. Each row should correspond to the cartesian coordinates of a single microphone.
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
    :param num_iter: Number of iterations for CSSM. Default is 1 but around 3 is suggested.
    :type num_iter: int
    """
    def __init__(self, L, fs, nfft, c=343.0, num_sources=1, mode='far', r=None, theta=None, phi=None, num_iter=1, **kwargs):
        MUSIC.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_sources=num_sources, mode=mode, r=r, theta=theta, phi=phi)
        self.iter = num_iter

    def _process(self, X):
        """
        Perform CSSM for given frame in order to estimate steered response spectrum.
        """

        # compute empirical cross correlation matrices
        C_hat = self._compute_correlation_matrices(X)

        # compute initial estimates
        beta = []
        invalid = []
        for k in range(self.num_freq):
            self.P = 1 / self._compute_spatial_spectrum(C_hat[k,:,:],self.freq[k])
            self._peaks1D()
            if len(self.src_idx) < self.num_sources:    # remove frequency
                invalid.append(k)
            # else:
            beta.append(self.src_idx)
        desired_freq = np.delete(self.freq, invalid)
        # self.num_freq = self.num_freq - len(invalid)

        # compute reference frequency (take bin with max amplitude)
        f0 = np.argmax(np.sum(np.sum(abs(X[:,self.freq,:]),axis=0),axis=1))
        f0 = self.freq[f0]

        # iterate to find DOA
        i = 0
        # B = np.concatenate((np.zeros([self.M-self.num_sources, self.num_sources]), np.identity(self.M-self.num_sources)), axis=1).T
        while(i < self.iter or (len(self.src_idx) < self.num_sources and i < 20)):
            # coherent sum
            R = self._coherent_sum(C_hat, f0, beta)
            # subspace decomposition
            Es, En, ws, wn = self._subspace_decomposition(R)
            # compute spatial spectrum
            cross = np.dot(En,np.conjugate(En).T)
            # cross = np.identity(self.M) - np.dot(Es, np.conjugate(Es).T)
            self.P = self._compute_spatial_spectrum(cross,f0)
            self._peaks1D()
            beta = np.tile(self.src_idx, (self.num_freq, 1))
            i += 1

    def _coherent_sum(self, C_hat, f0, beta):
        R = np.zeros((self.M,self.M))
        # coherently sum frequencies
        for j in range(len(self.freq)):
            k = self.freq[j]
            Aj = self.mode_vec[k,:,beta[j]].T
            A0 = self.mode_vec[f0,:,beta[j]].T
            B = np.concatenate((np.zeros([self.M-len(beta[j]), len(beta[j])]), np.identity(self.M-len(beta[j]))), axis=1).T
            Tj = np.dot(np.c_[A0, B], np.linalg.inv(np.c_[Aj, B]))
            R = R + np.dot(np.dot(Tj,C_hat[j,:,:]),np.conjugate(Tj).T)
        return R


