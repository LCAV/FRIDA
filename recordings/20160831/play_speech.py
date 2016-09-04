# Experiment from 2016/08/31
from __future__ import division

import numpy as np
import sounddevice as sd
import time
from scipy.io import wavfile

from realtimeaudio import AudioGetter

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

now = time.strftime("%Y%m%d-%H%M%S")
recording_folder = './Recordings/'
recording_filename = now + '-compactsix-speech-{}.wav'
sample_folder = './samples/'

samples = [ 'fq_sample{}.wav'.format(i) for i in range(3) ]

# the list contains tuples of (fs, signal)
fs_out, s = wavfile.read(sample_folder + samples[0])  # just to get the sampling rate
signals = [wavfile.read(sample_folder + sample)[1] for sample in samples]

# the microphones have high sampling rate
fs_in = 48000

# Setup output device
sd.default.device = 2
sd.default.samplerate = fs_out
sd.default.channels = 8

# time between two samples
T_sleep = 0.2

# We record in chunks because of limitations of pyramic array
chunk_length = 70

# Normalize the amplitude and length of speech signals
length = np.max([s.shape[0] for s in signals])
max_amp = np.max([np.max(np.abs(s)) for s in signals])

for s in signals:
    s *= 0.95 / max_amp

signal = np.zeros((length, sd.default.channels[1]))

n_spkrs = sd.default.channels[1]

n_comb_2_spkrs = n_spkrs * (n_spkrs - 1) / 2.
n_comb_3_spkrs = n_spkrs * (n_spkrs - 1) * (n_spkrs - 2) / (3 * 2)

total_time = (
        (length/fs_out + T_sleep)
        * ( n_spkrs + n_comb_2_spkrs + n_comb_3_spkrs )
        )
print 'Total time to play all signals',total_time
pause()

# The time counter for the chunks
time_counter = 0.
file_counter = 3

# start the compactsix array recorder
compactsix_server = AudioGetter("192.168.2.11:8888", chunk_length+5, fs=fs_in, channels=6)
compactsix_server.start()

# here is a skip counter for when the experiments fail midway but we don't
# want to redo everything
skip = 32
skip_counter = 0

print 'Don''t forget, we''ll be skipping the first {} signals'.format(skip)

# Play the speech  on each channel
s1 = signals[0]
for ch in range(sd.default.channels[1]):

    if skip_counter < skip:
        skip_counter += 1
        continue

    signal[:s1.shape[0],ch] = s1
    sd.play(signal, fs_out, blocking=True)
    signal[:,ch] = 0.
    time.sleep(T_sleep)

    time_counter += length/fs_out + T_sleep
    if time_counter > chunk_length:
        # wait for the file from the remote array
        compactsix_server.join()
        # and save it
        wavfile.write(
                recording_folder + recording_filename.format(file_counter), 
                compactsix_server.fs, 
                compactsix_server.audio)
        file_counter += 1

        # then manually wait for the pyramic to finish
        pause()
        time_counter = 0.
        
        # restart compactsix recording
        compactsix_server = AudioGetter("192.168.2.11:8888", chunk_length+5, fs=fs_in, channels=6)
        compactsix_server.start()

# Play both signals on all pairs
s1 = signals[0]
s2 = signals[1]
for ch1 in range(sd.default.channels[1]):
    for ch2 in range(ch1+1, sd.default.channels[1]):

        if skip_counter < skip:
            skip_counter += 1
            continue

        signal[:s1.shape[0],ch1] = s1
        signal[:s2.shape[0],ch2] = s2

        sd.play(signal, fs_out, blocking=True)

        signal[:,ch1] = 0.
        signal[:,ch2] = 0.

        time.sleep(T_sleep)

        time_counter += length/fs_out + T_sleep
        if time_counter > chunk_length:
            # wait for the file from the remote array
            compactsix_server.join()
            # and save it
            wavfile.write(
                    recording_folder + recording_filename.format(file_counter), 
                    compactsix_server.fs, 
                    compactsix_server.audio)
            file_counter += 1

            # then manually wait for the pyramic to finish
            pause()
            time_counter = 0.
            
            # restart compactsix recording
            compactsix_server = AudioGetter("192.168.2.11:8888", chunk_length+5, fs=fs_in, channels=6)
            compactsix_server.start()

# Play both signals on all pairs
s1 = signals[0]
s2 = signals[1]
s3 = signals[2]
for ch1 in range(n_spkrs):
    for ch2 in range(ch1+1, n_spkrs):
        for ch3 in range(ch2+1, n_spkrs):

            if skip_counter < skip:
                skip_counter += 1
                continue

            signal[:s1.shape[0],ch1] = s1
            signal[:s2.shape[0],ch2] = s2
            signal[:s3.shape[0],ch3] = s3

            sd.play(signal, fs_out, blocking=True)

            signal[:,ch1] = 0.
            signal[:,ch2] = 0.
            signal[:,ch3] = 0.

            time.sleep(T_sleep)

            time_counter += length/fs_out + T_sleep
            if time_counter > chunk_length:
                # wait for the file from the remote array
                compactsix_server.join()
                # and save it
                wavfile.write(
                        recording_folder + recording_filename.format(file_counter), 
                        compactsix_server.fs, 
                        compactsix_server.audio)
                file_counter += 1

                # then manually wait for the pyramic to finish
                pause()
                time_counter = 0.
                
                # restart compactsix recording
                compactsix_server = AudioGetter("192.168.2.11:8888", chunk_length+5, fs=fs_in, channels=6)
                compactsix_server.start()

# wait for compact six to finish
compactsix_server.join()

# and save it
wavfile.write(
        recording_folder + recording_filename.format(file_counter), 
        compactsix_server.fs, 
        compactsix_server.audio)
file_counter += 1

