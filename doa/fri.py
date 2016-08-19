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

    def __init__(self, L, fs, nfft, num_bands, max_four, c=343.0, num_sources=1, theta=None, G_iter=None, **kwargs):

        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_sources=num_sources, mode='far', theta=theta)
        self.num_bands = num_bands
        self.fc = np.array(num_bands, dtype=float)
        self.max_four = max_four
        self.visi_noisy_all = None
        self.fft_bins = None
        self.alpha_recon = np.array(num_sources, dtype=float)

        # Set the number of updates of the mapping matrix
        self.update_G = True if G_iter is not None and G_iter > 0 else False
        self.G_iter = G_iter if self.update_G else 1

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
        self.phi_recon, self.alpha_recon = pt_src_recon_multiband(self.visi_noisy_all, 
                self.L[0,:], self.L[1,:],
                2*np.pi*self.fc, self.c, 
                self.num_sources, self.max_four,
                noise_level, max_ini, 
                update_G=self.update_G, G_iter=self.G_iter, 
                verbose=False)

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
        omega_band = 2*np.pi*self.fc[0]
        sound_speed = self.c
        phi_plt = self.theta
        num_mic = self.M

        img = np.zeros(phi_plt.size, dtype=complex)
        x_plt, y_plt = polar2cart(1, phi_plt)

        pos_mic_x_normalised = pos_mic_x / (sound_speed / omega_band)
        pos_mic_y_normalised = pos_mic_y / (sound_speed / omega_band)

        count_visi = 0
        for q in xrange(num_mic):
            p_x_outer = pos_mic_x_normalised[q]
            p_y_outer = pos_mic_y_normalised[q]
            for qp in xrange(num_mic):
                if not q == qp:
                    p_x_qqp = p_x_outer - pos_mic_x_normalised[qp]  # a scalar
                    p_y_qqp = p_y_outer - pos_mic_y_normalised[qp]  # a scalar
                    # <= the negative sign converts DOA to propagation vector
                    img += visi[count_visi] * \
                           np.exp(-1j * (p_x_qqp * x_plt + p_y_qqp * y_plt))
                    count_visi += 1
        return img / (num_mic * (num_mic - 1))

    def polar_plt_dirac(self, phi_ref, alpha_ref, save_fig=False, file_name=None, plt_dirty_img=True):

        phi_recon = self.phi_recon
        alpha_recon = np.mean(self.alpha_recon, axis=1)
        num_mic = self.M
        phi_plt = self.theta
        dirty_img = self._gen_dirty_img()

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
        ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
        ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
        ax.set_ylim([0, 1.05 + max_val])
        if plt_dirty_img:
            ax.set_ylim([0, 1.05 + np.max(np.append(np.concatenate((alpha_ref, alpha_recon)), max_val))])
        else:
            ax.set_ylim([0, 1.05 + np.max(np.concatenate((alpha_ref, alpha_recon)))])
        if save_fig:
            if file_name is None:
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

