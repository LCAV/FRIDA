from __future__ import division

import numpy as np
from scipy import linalg as la
import scikits.samplerate as sr
from scipy.io import wavfile
import json

import sys

import matplotlib.pyplot as plt

import theaudioexperimentalist as tae

from experiment import PointCloud, arrays

temp = 25.3
humidity = 57.4
pressure = 1000.
c = tae.calculate_speed_of_sound(temp, humidity, pressure)

data_dir = '/Users/scheibler/switchdrive/LCAV-Audio/Recordings/'

fn_sweep = data_dir + '20160831/20160831_short_sweep.wav'

# Get the speakers and microphones geometry
exp_folder = data_dir + '20160831/'
sys.path.append(exp_folder)
from edm_to_positions import twitters
# Open the protocol json file
with open(exp_folder + 'protocol.json') as fd:
    exp_data = json.load(fd)

# open the sweep
r_sweep, sweep = wavfile.read(fn_sweep)

#array_type = 'BBB'
array_type = 'FPGA'

# open all recordings
if array_type == 'FPGA':
    R = arrays['pyramic_tetrahedron']

    # FPGA array reference point offset
    ref_pt_offset = 0.01  # meters

    # Adjust the z-offset of Pyramic
    R[2,:] += ref_pt_offset - R[2,0]

    # Localize microphones in new reference frame
    R += twitters[['pyramic']]

    # labels of the speakers
    labels = twitters.labels

    seg_len = 2.5
    offset = 1.5
    #fn_rec = data_dir + '20160726/data_pyramic/Test_Sweep/Mic_'
    fn_rec = data_dir + '20160831/data_pyramic/raw/20160831_sweeps/Mic_'
    rec = {}
    r_rec = 0
    for l,lbl in enumerate(labels):
        rec[lbl] = []
        for i in range(R.shape[1]):
            r_rec,s = wavfile.read(fn_rec + str(i) + '.wav')
            # The true sample rate of pyramic is fs = 1/0.00002094 = 47.76 kHz :(
            #r_rec = 47718.263
            #r_rec = 47500
            r_rec = 47718.6069
            b = int(r_rec * (offset + l*seg_len) )
            e = int(r_rec * (offset + (l+1)*seg_len) )
            rec[lbl].append(s[b:e])
        rec[lbl] = np.array(rec[lbl], dtype=np.float32).T/(2**15-1)

elif array_type == 'BBB':
    R = arrays['compactsix_circular_1']
    fn_rec = data_dir + 'BBB-blue/jul26/sweeps/speaker-7.wav'
    r_rec,rec = wavfile.read(fn_rec)

if r_sweep != r_rec:
    sweep = sr.resample(sweep, r_rec/r_sweep, 'sinc_best')

fs = r_rec

print 'Deconvolving'
h = {}
for lbl in ['7']:
    temp = []
    for mic in range(rec[lbl].shape[1]):
        temp.append(tae.deconvolve(rec[lbl][:,mic], sweep, thresh=0.05))
    h[lbl] = np.array(temp).T

print 'TDOA'
tdoa = [0]
lbl = '7'
for i in range(1,rec[lbl].shape[1]):
    tdoa.append(tae.tdoa(rec[lbl][:,i], rec[lbl][:,0], fs=fs, interp=1, phat=False))
tdoa = np.array(tdoa)

delay_d = tdoa * c
delay_d -= delay_d[0]

x0 = np.zeros(4)
x0[:3] = twitters[lbl]
x0[3] = la.norm(twitters[lbl] - R[:,0])
print 'Doing localization'
loc = np.array([tae.tdoa_loc(R, tdoa, c, x0=x0)]).T

tdoa2 = la.norm(R - loc, axis=0) / c
tdoa2 -= tdoa2[0]

tdoa3 = la.norm(R - twitters[[lbl]], axis=0) / c
tdoa3 -= tdoa3[0]

R = np.concatenate((R, loc), axis=1)
pc = PointCloud(X=R)
pc.labels[-1] = 'spkr'

plt.figure()
plt.plot(tdoa)
plt.plot(tdoa2)
plt.plot(tdoa3)
plt.legend(['TDOA measured','TDOA reconstructed','TDOA hand measured location'])


axes = pc.plot()
twitters.plot(axes=axes, c='r')
plt.axis('equal')
plt.show()
