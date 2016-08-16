
import numpy as np
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

sys.path.append('Experiment/arrays')
sys.path.append('Experiment')

from point_cloud import PointCloud
from distances import *
from pyramic_tetrahydron import R

# FPGA array reference point offset
ref_pt_offset = 0.01  # meters

# Adjust the z-offset of Pyramic
R[2,:] += 0.01 - R[2,0]

# Localize microphones in new reference frame
R += twitters[['FPGA']]

# correct FPGA reference point to be at the center of the array
twitters.X[:,twitters.key2ind('FPGA')] = R.mean(axis=1)

# Now we can try to visualize the geometry

pyramic = PointCloud(X=R)

# Plot all the markers in the same figure to check all the locations are correct
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')

twitters.plot(axes=axes, c='k', marker='s')
pyramic.plot(axes=axes, show_labels=False, c='r', marker='.')

plt.show()
