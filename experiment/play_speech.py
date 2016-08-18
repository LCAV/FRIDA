# Experiment from 2016/07/26
from __future__ import division

import numpy as np
import sounddevice as sd
import time
from scipy.io import wavfile

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

r,s1 = wavfile.read('fq_sample1.wav')
r,s2 = wavfile.read('fq_sample2.wav')

fs = r

# Setup device
sd.default.device = 2
sd.default.samplerate = fs
sd.default.channels = 8

T_sleep = 1.

length = np.maximum(s1.shape[0], s2.shape[0])
max_amp = np.maximum(np.abs(s1).max(), np.abs(s2).max())

s1 *= 0.95/max_amp
s2 *= 0.95/max_amp

signal = np.zeros((length, sd.default.channels[1]))

total_time = (
        (length/fs + T_sleep)
        * ( sd.default.channels[1]
            + sd.default.channels[1] * (sd.default.channels[1]-1) / 2. )
        )
print 'Total time to play all signals',total_time
            
time_counter = 0.

# Play the speech  on each channel
for ch in range(sd.default.channels[1]):
    signal[:s1.shape[0],ch] = s1
    sd.play(signal, fs, blocking=True)
    signal[:,ch] = 0.
    time.sleep(T_sleep)

    time_counter += length/fs + T_sleep
    if time_counter > 70.:
        pause()
        time_counter = 0.
        

# Play both signals on all pairs
for ch1 in range(sd.default.channels[1]):
    for ch2 in range(ch1+1, sd.default.channels[1]):
        signal[:s1.shape[0],ch1] = s1
        signal[:s2.shape[0],ch2] = s2

        sd.play(signal, fs, blocking=True)

        signal[:,ch1] = 0.
        signal[:,ch2] = 0.

        time.sleep(T_sleep)

        time_counter += length/fs + T_sleep
        if time_counter > 70.:
            pause()
            time_counter = 0.

