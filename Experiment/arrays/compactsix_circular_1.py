'''
This file describes the location of microphones on
a circular array of six microphones

Author: Robin Scheibler
Date: 2016 / 08 / 09
'''

import numpy as np

n_mics = 6        # number of microphones
radius = 67.5     # array radius in mm

# angle locations of microphones in fractions of 2 \pi
angles = np.array([0, 1./3., 2./3., 1./6., 3./6., 5./6.])

# Array of microphones location sin 3D
R = np.array([ 
  radius*np.cos(2.*np.pi*angles), 
  radius*np.sin(2.*np.pi*angles), 
  np.zeros(n_mics),
  ])

if __name__ == "__main__":

  import matplotlib.pyplot as plt

  plt.plot(R[0], R[1], 'o')
  for i in range(n_mics):
      plt.text(R[0,i]+1, R[1,i]+1, str(i+1))
  plt.show()
