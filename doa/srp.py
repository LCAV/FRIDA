# Author: Eric Bezzam
# Date: July 15, 2016

from doa import *

class SRP(DOA):
    """
    Class to apply Steered Response Power (SRP) direction-of-arrival (DoA) for a particular microphone array.

    .. note:: Run locate_source() to apply the SRP-PHAT algorithm.

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
    def __init__(self, L, fs, nfft, c=343.0, num_sources=1, mode='far', r=None,theta=None, phi=None, **kwargs):
        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_sources=num_sources, mode=mode, r=r, theta=theta, phi=phi)
        self.num_pairs = self.M*(self.M-1)/2
        self.mode_vec = np.conjugate(self.mode_vec)

    def _process(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response spectrum.
        """
        # average over snapshots
        S = X.shape[2]
        for s in range(S):
            X_s = X[:,self.freq,s]
            absX = abs(X_s)
            absX[absX<tol] = tol 
            pX = X_s/absX
            # grid search
            for k in range(self.num_loc):
                Yk = pX.T * self.mode_vec[self.freq,:,k]
                CC = np.dot(np.conj(Yk).T, Yk)
                self.P[k] = self.P[k]+abs(np.sum(np.triu(CC,1)))/S/self.num_freq/self.num_pairs
