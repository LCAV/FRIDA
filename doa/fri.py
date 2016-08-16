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

    def __init__(self, L, fs, nfft, num_bands, max_four, c=343.0, num_sources=1, mode='far', r=None,theta=None, phi=None):

        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_sources=num_sources, mode=mode, r=r, theta=theta, phi=phi)
        self.num_bands = num_bands
        self.fc = np.array(num_bands, dtype=float)
        self.max_four = max_four
        self.visi_noisy_all = None
        self.fft_bins = None
        self.alpha_recon = np.array(num_sources, dtype=float)

    def _process(self, X):

        # # Subband selection
        bands_pwr = np.mean(np.mean(np.abs(X[:,self.freq,:]) ** 2, axis=0), axis=1)+self.freq[0]
        self.fft_bins = np.argsort(bands_pwr)[-self.num_bands:]
        self.fc = self.fft_bins*self.fs/float(self.nfft)
        print('Selected bins: {0} Hertz'.format(self.fc))

        # loop over all subbands
        visi_noisy_all = []
        for band_count in xrange(self.fft_bins.size):
            # Estimate the covariance matrix and extract off-diagonal entries
            visi_noisy = extract_off_diag(cov_mtx_est(X[:,self.fft_bins[band_count],:]))
            visi_noisy_all.append(visi_noisy)

        # stack as columns (NOT SUBTRACTING NOISELESS)
        self.visi_noisy_all = np.column_stack(visi_noisy_all)

        # reconstruct point sources with FRI
        max_ini = 50  # maximum number of random initialisation
        noise_level = 1e-10
        self.phi_recon, self.alpha_recon = pt_src_recon_multiband(self.visi_noisy_all, self.L[0,:], self.L[1,:], 2*np.pi*self.fc, self.c, self.num_sources, self.max_four, noise_level, max_ini, update_G=True, G_iter=3, verbose=False)

    def plot_polar(self, plt_show=False):
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

        pos_mic_x = self.L[0,:]
        pos_mic_y = self.L[1,:]
        sound_speed = self.c
        omega_band = 2*np.pi*self.fc
        phi_plt = self.theta

        img = np.zeros(phi_plt.size, dtype=complex)
        x_plt, y_plt = polar2cart(1, phi_plt)
        num_mic = pos_mic_x.size

        for k in range(self.num_bands):
            count_visi = 0
            pos_mic_x_normalised = pos_mic_x / (sound_speed / omega_band[k])
            pos_mic_y_normalised = pos_mic_y / (sound_speed / omega_band[k])
            for q in xrange(num_mic):
                p_x_outer = pos_mic_x_normalised[q]
                p_y_outer = pos_mic_y_normalised[q]
                for qp in xrange(num_mic):
                    if not q == qp:
                        p_x_qqp = p_x_outer - pos_mic_x_normalised[qp]  # a scalar
                        p_y_qqp = p_y_outer - pos_mic_y_normalised[qp]  # a scalar
                        img += self.response[:,k][count_visi] * \
                               np.exp(1j * (p_x_qqp * x_plt + p_y_qqp * y_plt))
                        count_visi += 1
            img / (num_mic * (num_mic - 1))
        # TODO: average over dirty images

        self._polar_plt_diracs(phi_plt=phi_plt, dirty_img=img)

    def _polar_plt_diracs(self, save_fig=False, **kwargs):
        """
        plot Diracs in the polar coordinate
        :param phi_ref: ground truth Dirac locations (azimuths)
        :param phi_recon: reconstructed Dirac locations (azimuths)
        :param alpha_ref: ground truth Dirac amplitudes
        :param alpha_recon: reconstructed Dirac amplitudes
        :param num_mic: number of microphones
        :param P: PSNR in the visibility measurements
        :param save_fig: whether save the figure or not
        :param kwargs: optional input argument(s), include:
                file_name: file name used to save figure
        :return:
        """
        phi_recon = self.phi_recon
        alpha_recon = np.mean(self.alpha_recon, axis=1)
        num_mic = self.M

        # dist_recon = polar_distance(phi_ref, phi_recon)[0]
        if 'dirty_img' in kwargs and 'phi_plt' in kwargs:
            plt_dirty_img = True
            dirty_img = kwargs['dirty_img']
            phi_plt = kwargs['phi_plt']
        else:
            plt_dirty_img = False
        fig = plt.figure(figsize=(5, 4), dpi=90)
        ax = fig.add_subplot(111, projection='polar')

        # ax.scatter(phi_ref, 1 + alpha_ref, c=np.tile([0, 0.447, 0.741], (K, 1)), s=70,
        #            alpha=0.75, marker='^', linewidths=0, label='original')

        ax.scatter(phi_recon, 1+alpha_recon, c=np.tile([0.850, 0.325, 0.098], (self.num_sources, 1)), s=100, alpha=0.75, marker='*', linewidths=0, label='reconstruction')
        # for k in xrange(K):
        #     ax.plot([phi_ref[k], phi_ref[k]], [1, 1 + alpha_ref[k]],
        #             linewidth=1.5, linestyle='-', color=[0, 0.447, 0.741], alpha=0.6)

        for k in xrange(self.num_sources):
            ax.plot([phi_recon[k], phi_recon[k]], [1, 1 + alpha_recon[k]],
                    linewidth=1.5, linestyle='-', color=[0.850, 0.325, 0.098], alpha=0.6)

        plt_dirty_img = False
        if plt_dirty_img:
            dirty_img = dirty_img.real
            min_val = dirty_img.min()
            max_val = dirty_img.max()
            # color_lines = cm.spectral_r((dirty_img - min_val) / (max_val - min_val))
            # ax.scatter(phi_plt, 1 + dirty_img, edgecolor='none', linewidths=0,
            #         c=color_lines, label='dirty image')  # 1 is for the offset
            ax.plot(phi_plt, 1 + dirty_img, linewidth=1, alpha=0.55,
                    linestyle='-', color=[0.466, 0.674, 0.188], label='dirty image')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:3], framealpha=0.5,
                  scatterpoints=1, loc=8, fontsize=9,
                  ncol=1, bbox_to_anchor=(0.9, -0.17),
                  handletextpad=.2, columnspacing=1.7, labelspacing=0.1)
        title_str = r'$K={0}$, $\mbox{{\# of mic.}}={1}$, $\mbox{{SNR}}={2:.1f}$dB, average error={3:.1e}'
        # ax.set_title(title_str.format(repr(K), repr(num_mic), P, dist_recon),
        #              fontsize=11)
        # ax.set_title(title_str.format(repr(self.num_sources), repr(self.M), fontsize=11))
        ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
        ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
        if plt_dirty_img:
            ax.set_ylim([0, 1.05 + np.max(np.append(alpha_recon, max_val))])
        else:
            ax.set_ylim([0, 1.05 + np.max(alpha_recon)])
        if save_fig:
            if 'file_name' in kwargs:
                file_name = kwargs['file_name']
            else:
                file_name = 'polar_recon_dirac.pdf'
            plt.savefig(file_name, format='pdf', dpi=300, transparent=True)

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

