from __future__ import division, print_function

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from tools import rfft
import pyroomacoustics as pra

def select_bands(samples, freq_range, fs, nfft, win, n_bands):

    if win is not None and isinstance(win, bool):
        if win:
            win = np.hanning(nfft)
        else:
            win = None

    # Read the signals in a single array
    sig = [wavfile.read(s)[1] for s in samples]
    L = max([s.shape[0] for s in sig])
    signals = np.zeros((L,len(samples)), dtype=np.float32)
    for i in range(signals.shape[1]):
        signals[:sig[i].shape[0],i] = sig[i] / np.std(sig[i][sig[i] > 1e-2])

    sum_sig = np.sum(signals, axis=1)

    sum_STFT = pra.stft(sum_sig, nfft, nfft, win=win, transform=rfft).T
    sum_STFT_avg = np.mean(np.abs(sum_STFT)**2, axis=1)

    # Do some band selection
    bnds = np.linspace(freq_range[0], freq_range[1], n_bands+1)

    freq_hz = np.zeros(n_bands)
    freq_bins = np.zeros(n_bands, dtype=int)

    for i in range(n_bands):

        bl = int(bnds[i] / fs * nfft)
        bh = int(bnds[i+1] / fs * nfft)

        k = np.argmax(sum_STFT_avg[bl:bh])

        freq_hz[i] = (bl + k) / nfft * fs
        freq_bins[i] = k + bl

    return np.unique(freq_hz), np.unique(freq_bins)

'''
print('freq_hz = [' + ','.join([str(f) for f in freq_hz]) + ']', sep=',')

# Plot FFT of all signals
freqvec = np.fft.rfftfreq(L)
S = np.fft.rfft(signals, axis=0)

STFT = np.array([pra.stft(s, nfft, nfft, win=win, transform=rfft) for s in signals.T]).T
STFT_avg = np.array([np.mean(np.abs(st)**2, axis=0) for st in STFT])
f_stft = np.fft.rfftfreq(nfft)

plt.subplot(2,1,1)
plt.plot(fs*freqvec, pra.dB(S))

plt.subplot(2,1,2)
plt.plot(fs*f_stft, pra.dB(STFT_avg))
plt.plot(fs*f_stft, pra.dB(sum_STFT_avg), '--')
plt.plot(freq_hz, pra.dB(sum_STFT_avg[freq_bins]), '*')

plt.show()
'''
