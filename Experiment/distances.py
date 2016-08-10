
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy import linalg as la

import theaudioexperimentalist as tae

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
markers = tae.MarkerSet()
markers.fromEDM(EDM, labels=labels)

# We know that these markers should be roughly on a plane
s = [1,2,3,6]
print la.norm(markers.X[2,s]- markers.X[2,s].mean())

markers.flatten(['7','5','3','4'])
markers.center('FPGA')

markers.align('7','z')
print markers['7']

print la.norm(markers.X[2,s]- markers.X[2,s].mean())

# Set the origin at 'FPGA' and x-axis 

markers.plot()
