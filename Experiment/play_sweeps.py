 # Experiment from 2016/07/26

import numpy as np
import sounddevice as sd
import time
from scipy.io import wavfile

def sine_sweep(T, fs, f_low=50., f_high=22000.):

    f1 = f_low     # Start frequency in [Hz]
    f2 = f_high  # End frequency in [Hz]
    Ts = 1./fs   # Sampling period in [s]
    N = np.floor(T/Ts)
    n  = np.arange(0, N, dtype='float64')  # Sample index

    om1 = 2*np.pi*f1
    om2 = 2*np.pi*f2

    x_exp = 0.95*np.sin(om1*N*Ts / np.log(om2/om1) * (np.exp(n/N*np.log(om2/om1)) - 1))

    t_win = 0.2
    n_win = np.floor(t_win*fs)
    win = np.hanning(t_win*fs)
    x_exp[:n_win/2] *= win[:n_win/2]
    x_exp[-n_win/2:] *= win[-n_win/2:]

    return x_exp


if __name__ == "__main__":

    # Signal parameters
    fs = 16000.
    T = 6.
    T_sweep = 0.95*T
    T_sleep = 1.
    
    # Setup device
    sd.default.device = 2
    sd.default.samplerate = fs
    sd.default.channels = 8
    
    # create sweep signal
    sweep = np.zeros(int(T*fs))

    # save the sweep to deconvolve later
    sweep[:int(T_sweep*fs)] = sine_sweep(T_sweep, fs, f_low=50., f_high=7500.)
    wavfile.write("sweep.wav", fs, sweep)

    # Create multichannel signal
    signal = np.zeros((sweep.shape[0], sd.default.channels[0]))

    # Play the sweep on each channel
    for ch in range(sd.default.channels[0]):
        signal[:,ch] = sweep
        sd.play(signal, fs, blocking=True)
        signal[:,ch] = 0.
        time.sleep(1.)

