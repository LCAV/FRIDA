# Author: Eric Bezzam
# Date: July 15, 2016

from music import *

class WAVES(MUSIC):
    """
    Class to apply Weighted Average of Signal Subspaces (WAVES) for Direction of
    Arrival (DoA) estimation.

    .. note:: Run locate_source() to apply the WAVES algorithm.

    :param L: Microphone array positions. Each row should correspond to the 
    cartesian coordinates of a single microphone.
    :type L: numpy array
    :param fs: Sampling frequency.
    :type fs: float
    :param nfft: FFT length.
    :type nfft: int
    :param c: Speed of sound.
    :type c: float
    :param num_src: Number of sources to detect. Default is 1.
    :type num_src: int
    :param mode: 'far' (default) or 'near' for far-field or near-field detection
    respectively.
    :type mode: str
    :param r: Candidate distances from the origin. Default is r = np.ones(1) 
    corresponding to far-field.
    :type r: numpy array
    :param theta: Candidate azimuth angles (in radians) with respect to x-axis.
    :type theta: numpy array
    :param phi: Candidate elevation angles (in radians) with respect to z-axis. 
    Default value is phi = pi/2 as to search on the xy plane.
    :type phi: numpy array
    :param num_iter: Number of iterations for WAVES.
    :type num_iter: int
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,
        theta=None, phi=None, num_iter=1, **kwargs):
        MUSIC.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, theta=theta, phi=phi)
        self.iter = num_iter
        self.Z = None

    def _process(self, X):
        """
        Perform WAVES for given frame in order to estimate steered response 
        spectrum.
        """

        # compute empirical cross correlation matrices
        C_hat = self._compute_correlation_matrices(X)

        # compute initial estimates
        beta = []
        for k in range(self.num_freq):
            self.P = self._compute_spatial_spectrum(C_hat[k,:,:],
                self.freq_bins[k])
            self._peaks1D()
            beta.append(self.src_idx)

        # compute reference frequency (take bin with max amplitude)
        f0 = np.argmax(np.sum(np.sum(abs(X[:,self.freq_bins,:]),axis=0),axis=1))
        f0 = self.freq_bins[f0]

        # iterate to find DOA (but max 20)
        i = 0
        self.Z = np.empty((self.M,len(self.freq_bins)*self.num_src), 
            dtype='complex64')
        while(i < self.iter or (len(self.src_idx) < self.num_src and i < 20)):
            # construct waves matrix
            self._construct_waves_matrix(C_hat, f0, beta)
            # subspace decomposition with svd
            u,s,v = np.linalg.svd(self.Z)
            Un = u[:,self.num_src:]
            # compute spatial spectrum
            cross = np.dot(Un,np.conjugate(Un).T)
            self.P = self._compute_spatial_spectrum(cross,f0)
            self._peaks1D()
            beta = np.tile(self.src_idx, (self.num_freq, 1))
            i += 1

    def _construct_waves_matrix(self, C_hat, f0, beta):
        for j in range(len(self.freq_bins)):
            k = self.freq_bins[j]
            Aj = self.mode_vec[k,:,beta[j]].T
            A0 = self.mode_vec[f0,:,beta[j]].T
            B = np.concatenate((np.zeros([self.M-len(beta[j]), len(beta[j])]), 
                np.identity(self.M-len(beta[j]))), axis=1).T
            Tj = np.dot(np.c_[A0, B], np.linalg.inv(np.c_[Aj, B]))
            # estimate signal subspace
            Es, En, ws, wn = self._subspace_decomposition(C_hat[j,:,:])
            P = (ws-wn[-1])/np.sqrt(ws*wn[-1]+1)
            # form WAVES matrix
            idx1 = j*self.num_src
            idx2 = (j+1)*self.num_src
            self.Z[:,idx1:idx2] = np.dot(Tj, Es*P)

            
