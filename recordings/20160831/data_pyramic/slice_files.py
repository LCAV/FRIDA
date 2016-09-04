from __future__ import division
from scipy.io import wavfile
import numpy as np
import os, sys, getopt

from scikits.samplerate import resample

def segment_files(argv):

    txtfile = ''
    output_dir = ''
    try:
        opts, args = getopt.getopt(argv,"hf:o:r:",["file=","output_dir=","samplerate="])
    except getopt.GetoptError:
        print 'slice_files.py -f <txtfile> -o <output_dir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'slice_files.py -f <txtfile> -o <output_dir> -r <samplerate>'
            sys.exit()
        elif opt in ("-f", "--file"):
            txtfile = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-r", "--samplerate"):
            fs_out = int(arg)

    f = open(txtfile, "r")
    param = []
    for line in f:
        param.append(line.split())
    directory = param[0][0]
    start_times = map(float,param[1])
    end_times = map(float,param[2])
    labels = param[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for start_time, end_time, label in zip(start_times, end_times, labels):
        # read audio files
        mics = []
        for i in range(48):
            file_name = directory + '/Mic_' + str(i) + '.wav'
            fs_reported, audio = wavfile.read(file_name)
            mics.append(audio)

        # For pyramic (as of 2016/08/31) the sampling frequency is
        # wrongly reported as 48000 Hertz
        # The correct value is close to
        fs = 47718.6069

        # convert to numpy
        sigs = np.array(mics)
        # select slice
        segs = []
        for i in range(48):
            # The segmentation was done based on fs = 48 kHz
            seg = select_slice(sigs[i,:],start_time, end_time, fs_reported)
            # Resample the signals to something reasonable
            if fs != fs_out:
                seg = resample(seg, fs_out/fs, 'sinc_best')
            segs.append(seg)
        segment = np.array(segs).T

        # write to WAV
        file_name = output_dir + '/' + label + '.wav'
        wavfile.write(file_name, fs_out, segment)

def select_slice(x, start_time, end_time, fs):
    start_sample = int(start_time*fs)
    end_sample = int(end_time*fs)
    if len(np.shape(x))==1:
        return x[start_sample:end_sample]
    else:
        return x[:,start_sample:end_sample]

if __name__ == "__main__":
    segment_files(sys.argv[1:])


