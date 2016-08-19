# Author: Eric Bezzam
# Date: July 15, 2016

from doa import *

class MUSIC(DOA):
    """
    Class to apply MUltiple SIgnal Classication (MUSIC) direction-of-arrival (DoA) for a particular microphone array.

    .. note:: Run locate_source() to apply the MUSIC algorithm.

    :param L: Microphone array positions. Each column should correspond to the cartesian coordinates of a single microphone.
    :type L: numpy array
    :param fs: Sampling frequency.
    :type fs: float
    :param nfft: FFT length.
    :type nfft: int
    :param c: Speed of sound.
    :type c: float
    :param num_src: Number of sources to detect. Default is 1.
    :type num_src: int
    :param mode: 'far' (default) or 'near' for far-field or near-field detection respectively.
    :type mode: str
    :param r: Candidate distances from the origin. Default is r = np.ones(1) corresponding to far-field.
    :type r: numpy array
    :param theta: Candidate azimuth angles (in radians) with respect to x-axis.
    :type theta: numpy array
    :param phi: Candidate elevation angles (in radians) with respect to z-axis. Default value is phi = pi/2 as to search on the xy plane.
    :type phi: numpy array
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,theta=None, phi=None, **kwargs):
        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, mode=mode, r=r, theta=theta, phi=phi)
        self.Pssl = None

    def _process(self, X):
        """
        Perform MUSIC for given frame in order to estimate steered response spectrum.
        """

        # compute steered response
        self.Pssl = np.zeros((self.num_freq,self.num_loc))
        num_freq = self.num_freq
        C_hat = self._compute_correlation_matrices(X)
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            # subspace decomposition
            Es, En, ws, wn = self._subspace_decomposition(C_hat[i,:,:])
            # compute spatial spectrum
            # cross = np.dot(En,np.conjugate(En).T)
            cross = np.identity(self.M) - np.dot(Es, np.conjugate(Es).T) 
            self.Pssl[i,:] = self._compute_spatial_spectrum(cross,k)
        self.P = sum(self.Pssl)/num_freq

    def plot_individual_spectrum(self):
        """
        Plot the steered response for each frequency.
        """
        # check if matplotlib imported
        if matplotlib_available is False:
            warnings.warn('Could not import matplotlib.')
            return
        # only for 2D
        if len(self.theta)!=1 and len(self.phi)==1 and len(self.r)==1:
            pass
        else:
            warnings.warn('Only for 2D.')
            return
        # plot
        for k in range(self.num_freq):
            freq = float(self.freq_bins[k])/self.nfft*self.fs
            azimuth = self.theta*180/np.pi
            plt.plot(azimuth, self.Pssl[k,0:len(azimuth)])
            plt.ylabel('Magnitude')
            plt.xlabel('Azimuth [degrees]')
            plt.xlim(min(azimuth),max(azimuth))
            plt.title('Steering Response Spectrum - ' + str(freq) 
                + ' Hz' )
            plt.grid(True)
            plt.show()

    def _compute_spatial_spectrum(self,cross,k):
        # Dc = np.array(self.mode_vec[:,k,n],ndmin=2).T
        # Dc_H = np.conjugate(np.array(self.mode_vec[:,k,n],ndmin=2))
        # denom = np.dot(np.dot(Dc_H,cross),Dc)
        # return 1/abs(denom)
        P = np.zeros(self.num_loc)
        for n in range(self.num_loc):
            Dc = np.array(self.mode_vec[k,:,n],ndmin=2).T
            Dc_H = np.conjugate(np.array(self.mode_vec[k,:,n],ndmin=2))
            denom = np.dot(np.dot(Dc_H,cross),Dc)
            P[n] = 1/abs(denom)
        return P

    def _compute_correlation_matrices(self, X):
        S = X.shape[2]
        C_hat = np.zeros([self.num_freq,self.M,self.M], dtype=complex)
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            for s in range(S):
                C_hat[i,:,:] = C_hat[i,:,:] + np.outer(X[:,k,s], np.conjugate(X[:,k,s]))
        return C_hat/S

    def _subspace_decomposition(self, R):
        w,v = np.linalg.eig(R)
        eig_order = np.flipud(np.argsort(abs(w)))
        sig_space = eig_order[:self.num_src]
        noise_space = eig_order[self.num_src:]
        ws = w[sig_space]
        wn = w[noise_space]
        Es = v[:,sig_space]
        En = v[:,noise_space]
        return Es, En, ws, wn
