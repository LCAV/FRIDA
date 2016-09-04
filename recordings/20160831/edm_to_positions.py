
import numpy as np
from scipy import linalg as la
import json

from experiment import PointCloud

# Open the experimental protocol
with open('recordings/20160831/protocol.json') as fd:
    protocol = json.load(fd) 

# Get the labels and distances (flattened upper triangular of distance matrix, row-wise)
labels = protocol['calibration']['labels']
flat_distances = protocol['calibration']['distances']
m = len(labels)

# fill in the EDM
EDM = np.zeros((m,m))
flat_counter = 0
for i in range(0,m-1):
    for j in range(i+1,m):
        EDM[i,j] = flat_distances[flat_counter]**2
        EDM[j,i] = EDM[i,j]
        flat_counter += 1

# Create the marker objects
markers = PointCloud(EDM=EDM, labels=labels)

# Here we know all speakers should be in a plane
markers.flatten(labels[:-2])

# Let the pyramic ref point be the center
markers.center('pyramic')

# And align x-axis onto speaker 7
markers.align(labels[0],'z')

# The speakers have some rotations around z-axis
# i.e. the baffles point to different directions
rotations = protocol['calibration']['rotation']
def rotz(v, deg):
    ''' rotation around z axis by some degrees '''
    c1, s1 = np.cos(deg/180.*np.pi), np.sin(deg/180.*np.pi)
    rot = np.array([[c1, -s1, 0.], [s1, c1, 0.], [0., 0., 1.]])
    return np.dot(rot, v)

# Now there a few correction vectors to apply between the measurement points
# and the center of the baffles
vec_twit = np.array([-0.01, 0., -0.05])  # the correction vector
corr_twitter = {}
for lbl, rot in zip(labels[:-2], rotations[:-2]):
    corr_twitter[lbl] = rotz(vec_twit, rot[2])

vec_woof = np.array([-0.02, 0., -0.155])
corr_woofer = {}
for lbl, rot in zip(labels[:-2], rotations[:-2]):
    corr_woofer[lbl] = rotz(vec_woof, rot[2])

# Now make two sets of markers for twitters and woofers
twitters = markers.copy()
woofers = markers.copy()

# Apply the correction vectors
twitters.correct(corr_twitter)
woofers.correct(corr_woofer)

if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Plot all the markers in the same figure to check all the locations are correct
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    twitters.plot(axes=axes, c='b', marker='s')
    woofers.plot(axes=axes, c='r', marker='<')
    markers.plot(axes=axes, c='k', marker='.')

    print 'DoA of Speaker 5 to FPGA:', twitters.doa('pyramic','5')/np.pi*180.,'degrees'

    plt.show()

