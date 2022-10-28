import numpy as np
from numpy.linalg import eig


def triangulation (P1, P2, u1, u2):
    """
    P1: numpy array, shape  (3,4)
    P2: numpy array, shape (3,4)
    point: numpy array, shape (3,1)
    """

    u1 = u1/u1[2]
    u2 = u2/u2[2]
    A = np.zeros((4,4))
    A[0,:] = u1[0] * P1[2,:] - P1[0,:]
    A[1,:] = u1[1] * P1[2,:] - P1[1,:]
    A[2,:] = u2[0] * P2[2,:] - P2[0,:]
    A[3,:] = u2[1] * P2[2,:] - P2[1,:]

    A_final = np.matmul(np.transpose(A), A)
    e_vals,e_vecs = eig(A_final)
    e_vals = list(e_vals)
    idx = e_vals.index(min(e_vals))
    x = e_vecs[:,idx]/e_vecs[3,idx]
    return x


# example
P1 = np.array([[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]]);

P2 = np.array([[1.0, 0.0, 0.0, 3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]]); # camera matrix change this

x = 3
y = 5
z = 5
pt3d = np.array([[x],[y],[z], [1]])
u1 = np.matmul(P1,pt3d)
u2 = np.matmul(P2,pt3d)
recovered = triangulation(P1, P2, u1, u2)
print(recovered)


