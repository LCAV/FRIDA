 # Experiment from 2016/08/31

import numpy as np
import sounddevice as sd
import time
from scipy.io import wavfile

def window(signal, n_win):
    ''' window the signal at beginning and end with window of size n_win/2 '''

    win = np.hanning(2*n_win)

    sig_copy = signal.copy()

    sig_copy[:n_win] *= win[:n_win]
    sig_copy[-n_win:] *= win[-n_win:]

    return sig_copy

def sine_sweep(T, fs, f_low=50., f_high=22000.):

    f1 = f_low     # Start frequency in [Hz]
    f2 = f_high  # End frequency in [Hz]
    Ts = 1./fs   # Sampling period in [s]
    N = np.floor(T/Ts)
    n  = np.arange(0, N, dtype='float64')  # Sample index

    om1 = 2*np.pi*f1
    om2 = 2*np.pi*f2

    x_exp = 0.95*np.sin(om1*N*Ts / np.log(om2/om1) * (np.exp(n/N*np.log(om2/om1)) - 1))
    x_exp = x_exp[::-1]

    return x_exp


if __name__ == "__main__":

    from realtimeaudio import AudioGetter

    # Signal parameters
    fs = 48000.
    T = 3.
    T_sweep = 0.96*T
    T_sleep = 1.
    
    # Setup device
    sd.default.device = 2
    sd.default.samplerate = fs
    sd.default.channels = (8,1)

    # create sweep signal
    sweep = np.zeros(int(T*fs))

    # save the sweep to deconvolve later
    short_sweep = sine_sweep(T_sweep, fs, f_low=20., f_high=fs/2-100.)
    win_sweep = window(short_sweep, int(fs*0.01))
    sweep[:int(T_sweep*fs)] = win_sweep
    wavfile.write("short_sweep.wav", fs, short_sweep)
    wavfile.write("win_sweep.wav", fs, win_sweep)

    # Create multichannel signal
    signal = np.zeros((sd.default.channels[0]*sweep.shape[0], sd.default.channels[0]))

    compactsix_server = AudioGetter("192.168.2.11:8888", sd.default.channels[0]*T+1, fs=fs, channels=6)
    compactsix_server.start()

    # Play the sweep on each channel
    for ch in range(sd.default.channels[0]):
        s = ch*sweep.shape[0]
        e = (ch+1)*sweep.shape[0]

        signal[s:e,ch] = sweep


    data_in = sd.playrec(signal, fs, blocking=True)

    compactsix_server.join()
