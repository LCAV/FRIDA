from scipy.io import wavfile
import numpy as np
import os, sys, getopt

def segment_files(argv):

    txtfile = ''
    output_dir = ''
    try:
      opts, args = getopt.getopt(argv,"hf:o:",["file=","output_dir="])
    except getopt.GetoptError:
      print 'slice_files.py -f <txtfile> -o <output_dir>'
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'slice_files.py -f <txtfile> -o <output_dir>'
            sys.exit()
        elif opt in ("-f", "--file"):
            txtfile = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg

    f = open(txtfile, "r")
    param = []
    for line in f:
        param.append(line.split())
    filename = param[0][0]
    start_times = map(float,param[1])
    end_times = map(float,param[2])
    labels = param[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read audio file
    fs, audio = wavfile.read(filename)
    audio = np.array(audio)

    for start_time, end_time, label in zip(start_times, end_times, labels):
        seg = select_slice(audio, start_time, end_time, fs)
        # write to wav
        file_name = output_dir + '/' + label + '.wav'
        wavfile.write(file_name, fs, seg)

def select_slice(x, start_time, end_time, fs):
    start_sample = int(start_time*fs)
    end_sample = int(end_time*fs)
    if len(np.shape(x))==1:
        return x[start_sample:end_sample]
    else:
        return x[start_sample:end_sample,:]

if __name__ == "__main__":
    segment_files(sys.argv[1:])


