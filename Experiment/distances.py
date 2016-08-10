
import numpy as np
from scipy import linalg as la

from point_cloud import PointCloud

# The collection of markers where distances were measured
labels = ['11', '7', '5', '3', '13', '8', '4', '14', 'FPGA', 'BBB']
# The Euclidean distance matrix. Unit is squared meters
EDM = np.array(
      [ [ 0,    0.79, 1.63, 2.42, 2.82, 3.55, 2.44, 2.87, 2.22, 1.46 ],
        [ 0.79, 0,    1.45, 2.32, 2.49, 3.67, 2.32, 2.54, 1.92, 1.35 ],
        [ 1.63, 1.45, 0,    1.92, 2.09, 4.02, 3.48, 3.66, 2.50, 1.68 ],
        [ 2.42, 2.32, 1.92, 0,    0.86, 2.43, 2.92, 3.14, 1.56, 1.14 ],
        [ 2.82, 2.49, 2.09, 0.86, 0,    3.15, 3.10, 3.07, 1.58, 1.56 ],
        [ 3.55, 3.76, 4.02, 2.43, 3.15, 0,    2.44, 2.88, 2.11, 2.45 ],
        [ 2.44, 2.32, 3.48, 2.92, 3.10, 2.44, 0,    0.85, 1.52, 2.00 ],
        [ 2.87, 2.54, 3.66, 3.14, 3.07, 2.88, 0.85, 0,    1.60, 2.31 ],
        [ 2.22, 1.92, 2.50, 1.56, 1.58, 2.11, 1.52, 1.60, 0,    0.97 ],
        [ 1.46, 1.35, 1.68, 1.14, 1.56, 2.45, 2.00, 2.31, 0.97, 0 ] ]
      )**2

# Create the marker objects
markers = PointCloud(EDM=EDM, labels=labels)

# We know that these markers should be roughly on a plane
markers.flatten(['7','5','3','4'])

# Let the FPGA ref point be the center
markers.center('FPGA')

# And align x-axis onto speaker 7
markers.align('7','z')

# Now there a few correction vectors to apply between the measurement points
# and the center of the baffles
corr_twitter = {
        '7'  : np.array([+0.01, 0,    -0.05]),
        '3'  : np.array([0.,   -0.01, -0.05]),
        '4'  : np.array([0.,   +0.01, -0.05]),
        '5'  : np.array([0.01*np.cos(np.pi/4.), -0.01*np.sin(np.pi/4.), -0.05]),
        '11' : np.array([+0.01, 0,    -0.05]),  # top row, this needs to be rotated +30 deg around y-axis
        '8'  : np.array([-0.01, 0,    -0.05]),  # top row, this needs to be rotated -30 deg around y-axis
        '13' : np.array([0.,   -0.01, -0.19]),  # bottom row, this needs to be rotated +30 deg around x-axis
        '14' : np.array([0.,   +0.01, -0.19]),  # bottom row, this needs to be rotated -30 deg around x-axis
        }
corr_woofer = {
        '7'  : np.array([+0.02, 0,    -0.155]),
        '3'  : np.array([0.,   -0.02, -0.155]),
        '4'  : np.array([0.,   +0.02, -0.155]),
        '5'  : np.array([0.02*np.cos(np.pi/4.), -0.01*np.sin(np.pi/4.), -0.155]),
        '11' : np.array([+0.02, 0,    -0.155]),  # top row, this needs to be rotated +30 deg around y-axis
        '8'  : np.array([-0.02, 0,    -0.155]),  # top row, this needs to be rotated -30 deg around y-axis
        '13' : np.array([0.,   -0.02, -0.090]),  # bottom row, this needs to be rotated +30 deg around x-axis
        '14' : np.array([0.,   +0.02, -0.090]),  # bottom row, this needs to be rotated -30 deg around x-axis
        }

# Build rotation matrices by 30 degrees
c,s = np.cos(np.pi/6.), np.sin(np.pi/6.)
R_30_y = np.array([[c, 0., -s], [0., 1., 0], [s, 0., c]])
R_30_x = np.array([[1, 0., 0.], [0., c, -s], [0., s, c]])

# Apply the rotations
corr_twitter['11'] = np.dot(R_30_y, corr_twitter['11'])
corr_twitter['8'] = np.dot(R_30_y.T, corr_twitter['8'])
corr_twitter['13'] = np.dot(R_30_x, corr_twitter['13'])
corr_twitter['14'] = np.dot(R_30_x.T, corr_twitter['14'])
corr_woofer['11'] = np.dot(R_30_y, corr_woofer['11'])
corr_woofer['8'] = np.dot(R_30_y.T, corr_woofer['8'])
corr_woofer['13'] = np.dot(R_30_x, corr_woofer['13'])
corr_woofer['14'] = np.dot(R_30_x.T, corr_woofer['14'])

# Now make two sets of markers for twitters and woofers
twitters = markers.copy()
twitters.correct(corr_twitter)
woofers = markers.copy()
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

    plt.show()

