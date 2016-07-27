
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def classical_mds(D, dim=3):

    # Apply MDS algorithm for denoising
    n = D.shape[0]
    J = np.eye(n) - np.ones((n,n))/float(n)
    G = -0.5*np.dot(J, np.dot(D, J))

    s, U = np.linalg.eig(G)

    # we need to sort the eigenvalues in decreasing order
    s = np.real(s)
    o = np.argsort(s)
    s = s[o[::-1]]
    U = U[:,o[::-1]]

    S = np.diag(s)[0:dim,:]
    return np.dot(np.sqrt(S),U.T)

labels = ['11', '7', '5', '3', '13', '8', '4', '14', 'FPGA', 'BBB']
EDM = np.array(
      [ [ 0, 0.79, 1.63, 2.42, 2.82, 3.55, 2.44, 2.87, 2.22, 1.46 ],
        [ 0.79, 0, 1.45, 2.32, 2.49, 3.67, 2.32, 2.54, 1.92, 1.35 ],
        [ 1.63, 1.45, 0, 1.92, 2.09, 4.02, 3.48, 3.66, 2.50, 1.68 ],
        [ 2.42, 2.32, 1.92, 0, 0.86, 2.43, 2.92, 3.14, 1.56, 1.14 ],
        [ 2.82, 2.49, 2.09, 0.86, 0, 3.15, 3.10, 3.07, 1.58, 1.56 ],
        [ 3.55, 3.76, 4.02, 2.43, 3.15, 0, 2.44, 2.88, 2.11, 2.45 ],
        [ 2.44, 2.32, 3.48, 2.92, 3.10, 2.44, 0, 0.85, 1.52, 2.00 ],
        [ 2.87, 2.54, 3.66, 3.14, 3.07, 2.88, 0.85, 0, 1.60, 2.31 ],
        [ 2.22, 1.92, 2.50, 1.56, 1.58, 2.11, 1.52, 1.60, 0, 0.97 ],
        [ 1.46, 1.35, 1.68, 1.14, 1.56, 2.45, 2.00, 2.31, 0.97, 0 ] ]
      )**2

X = classical_mds(EDM)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0,:], X[1,:], X[2,:])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
