from __future__ import division

import numpy as np
from scipy import linalg as la
import scikits.samplerate as sr
from scipy.io import wavfile
import json
import sys
import matplotlib.pyplot as plt

import theaudioexperimentalist as tae
from experiment import PointCloud, arrays, calculate_speed_of_sound

exp_dir = '/Users/scheibler/switchdrive/LCAV-Audio/Recordings/20160831'

fn_sweep = exp_dir + '/20160831_short_sweep.wav'

# Get the speakers and microphones geometry
sys.path.append(exp_dir)
from edm_to_positions import twitters

# labels of the speakers
labels = twitters.labels

# Open the protocol json file
with open(exp_dir + '/protocol.json') as fd:
    exp_data = json.load(fd)

temp = exp_data['conditions']['temperature']
hum = exp_data['conditions']['humidity']
c = calculate_speed_of_sound(temp, hum)

# open the sweep
r_sweep, sweep = wavfile.read(fn_sweep)

spkr = ['16']
#array_type = 'BBB'
#array_type = 'FPGA'
array_type = 'FPGA_speech'

# open all recordings
if array_type == 'FPGA':
    R = arrays['pyramic_tetrahedron'].copy()

    # Localize microphones in new reference frame
    R += twitters[['pyramic']]

    seg_len = 17.8 / 6
    offset = 3.85 - seg_len
    fn_rec = exp_dir + '/data_pyramic/raw/20160831_sweeps/Mic_'
    rec = {}
    r_rec = 0
    for l,lbl in enumerate(labels):
        rec[lbl] = []
        for i in range(R.shape[1]):
            r_rec,s = wavfile.read(fn_rec + str(i) + '.wav')
            r_rec = 47718.6069
            #r_rec = 47760.
            b = int(r_rec * (offset + l*seg_len) )
            e = int(r_rec * (offset + (l+1)*seg_len) )
            rec[lbl].append(s[b:e])
        rec[lbl] = np.array(rec[lbl], dtype=np.float32).T/(2**15-1)

elif array_type == 'FPGA_speech':
    R = arrays['pyramic_tetrahedron'].copy()
    R += twitters[['pyramic']]

    mics = PointCloud(X=R)
    D = np.sqrt(mics.EDM())

    rec = {}
    for lbl in labels[:-2]:
        fn_rec = exp_dir + '/data_pyramic/segmented/one_speaker/{}.wav'.format(lbl)
        r_rec, s = wavfile.read(fn_rec)
        #r_rec = 47718.6069

        # segment the file
        rec[lbl] = s

elif array_type == 'BBB':
    R = arrays['compactsix_circular_1'].copy()
    R += twitters[['compactsix']]

    fn_rec = exp_dir + '/data_compactsix/raw/20160831_compactsix_sweeps.wav'
    r_rec, s = wavfile.read(fn_rec)

    # segment the file
    seg_len = 3.
    offset = 0.
    rec = {}
    for l,lbl in enumerate(labels):
        rec[lbl] = []
        b = int(r_rec * (offset + l*seg_len) )
        e = int(r_rec * (offset + (l+1)*seg_len) )
        rec[lbl] = s[b:e,:] / (2**15-1)

if r_sweep != r_rec:
    print 'Resample sweep'
    sweep = sr.resample(sweep, r_rec/r_sweep, 'sinc_best')

fs = r_rec

print 'TDOA'

if array_type == 'FPGA_speech':

    tdoa = []
    for i in range(0,rec[spkr[0]].shape[1]):
       tdoa.append(tae.tdoa(rec[spkr[0]][:,i], rec[spkr[0]][:,0], fs=fs, interp=4, phat=True))
    tdoa = np.array(tdoa)
    tdoa -= tdoa[0]

else:

    print 'Deconvolving'
    h = {}
    for lbl in spkr:
        temp = []
        for mic in range(rec[lbl].shape[1]):
            temp.append(tae.deconvolve(rec[lbl][:,mic], sweep, thresh=0.1))
        h[lbl] = np.array(temp).T

    print 'TDOA'
    tdoa = []
    for i in range(0,rec[spkr[0]].shape[1]):
        #tdoa.append(tae.tdoa(rec[spkr[0]][:,i], rec[spkr[0]][:,0], fs=fs, interp=1, phat=True))
        k = np.argmax(np.abs(h[spkr[0]][:,i]))
        if k > h[spkr[0]].shape[0]/2:
            k -= h[spkr[0]].shape[0]
        tdoa.append(k/fs)
    tdoa = np.array(tdoa)
    tdoa -= tdoa[0]

delay_d = tdoa * c
delay_d -= delay_d[0]

x0 = np.zeros(4)
x0[:3] = twitters[spkr[0]]
x0[3] = la.norm(twitters[spkr[0]] - R[:,0])
print 'Doing localization'

remove = [32, 47]
if array_type == 'BBB':
    loc = np.array([tae.tdoa_loc(R[:2,:], tdoa, c, x0=x0[:2])]).T
    loc = np.concatenate((loc, R[-1:,:1]))
else:
    loc = np.array([tae.tdoa_loc(R, tdoa, c, x0=x0)]).T

tdoa2 = la.norm(R - loc, axis=0) / c
tdoa2 -= tdoa2[0]

tdoa3 = la.norm(R - twitters[[spkr[0]]], axis=0) / c
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
