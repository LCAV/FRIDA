from doa import *

from scipy import linalg
from tools_fri_doa_plane import pt_src_recon_multiband, extract_off_diag, cov_mtx_est

import os
if os.environ.get('DISPLAY') is None:
    import matplotlib
    matplotlib.use('Agg')

from matplotlib import rcParams

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{bm}"]

class FRI(DOA):

    def __init__(self, L, fs, nfft, max_four, c=343.0, num_src=1, theta=None):

        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, mode='far', theta=theta)
        self.max_four = max_four
        self.visi_noisy_all = None
        self.alpha_recon = np.array(num_src, dtype=float)

    def _process(self, X):

        # loop over all subbands
        self.num_freq = self.freq_bins.shape[0]
        visi_noisy_all = []
        for band_count in range(self.num_freq):
            # Estimate the covariance matrix and extract off-diagonal entries
            visi_noisy = extract_off_diag(cov_mtx_est(X[:,self.freq_bins[band_count],:]))
            visi_noisy_all.append(visi_noisy)

        # stack as columns (NOT SUBTRACTING NOISELESS)
        self.visi_noisy_all = np.column_stack(visi_noisy_all)

        # reconstruct point sources with FRI
        max_ini = 50  # maximum number of random initialisation
        noise_level = 1e-10
        self.phi_recon, self.alpha_recon = pt_src_recon_multiband(self.visi_noisy_all, self.L[0,:], self.L[1,:], 2*np.pi*self.freq_hz, self.c, self.num_src, self.max_four, noise_level, max_ini, update_G=True, G_iter=3, verbose=False)

    def _gen_dirty_img(self):
        """
        Compute the dirty image associated with the given measurements. Here the Fourier transform
        that is not measured by the microphone array is taken as zero.
        :param visi: the measured visibilites
        :param pos_mic_x: a vector contains microphone array locations (x-coordinates)
        :param pos_mic_y: a vector contains microphone array locations (y-coordinates)
        :param omega_band: mid-band (ANGULAR) frequency [radian/sec]
        :param sound_speed: speed of sound
        :param phi_plt: plotting grid (azimuth on the circle) to show the dirty image
        :return:
        """
        # TODO: average over subbands instead of taking 0
        visi = self.visi_noisy_all[:, 0]
        pos_mic_x = self.L[0,:]
        pos_mic_y = self.L[1, :]
        omega_band = 2*np.pi*self.freq_hz[0]
        sound_speed = self.c
        phi_plt = self.theta
        num_mic = self.M

        img = np.zeros(phi_plt.size, dtype=complex)
        x_plt, y_plt = polar2cart(1, phi_plt)

        pos_mic_x_normalised = pos_mic_x / (sound_speed / omega_band)
        pos_mic_y_normalised = pos_mic_y / (sound_speed / omega_band)

        count_visi = 0
        for q in range(num_mic):
            p_x_outer = pos_mic_x_normalised[q]
            p_y_outer = pos_mic_y_normalised[q]
            for qp in range(num_mic):
                if not q == qp:
                    p_x_qqp = p_x_outer - pos_mic_x_normalised[qp]  # a scalar
                    p_y_qqp = p_y_outer - pos_mic_y_normalised[qp]  # a scalar
                    # <= the negative sign converts DOA to propagation vector
                    img += visi[count_visi] * \
                           np.exp(-1j * (p_x_qqp * x_plt + p_y_qqp * y_plt))
                    count_visi += 1
        return img / (num_mic * (num_mic - 1))

#-------------MISC--------------#

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

