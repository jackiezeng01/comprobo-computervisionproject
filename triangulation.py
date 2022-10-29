import numpy as np
from numpy.linalg import eig
import cv2


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
z = 2
pt3d = np.array([[x],[y],[z], [1]])
u1 = np.matmul(P1,pt3d)
u2 = np.matmul(P2,pt3d)
recovered = triangulation(P1, P2, u1, u2)
print(recovered)

epipolar_threshold = 0.006737946999085467
E, mask = cv2.findFundamentalMat(u1,u2,cv2.FM_RANSAC,epipolar_threshold)


'''
math for finding fundamental matrix
1. normalize the points so that the centroid of the reference points is at the origin of the coordinates and the RMS distance of the point from the origin is sqrt(2)
2. For each point, create a linear equation
3. Solve the linear equations (at least 8 eq) using svd
4. Check that our fundamental matrix is 
'''

def calc_F(pts1, pts2):
    A = np.zeros((len(pts1),9))
    # img1 x' y' x y im2
    for i in range(len(pts1)):
        A[i][0] = pts1[i][0]*pts2[i][0]
        A[i][1] = pts1[i][1]*pts2[i][0]
        A[i][2] = pts2[i][0]
        A[i][3] = pts1[i][0]*pts2[i][1]
        A[i][4] = pts1[i][1]*pts2[i][1]
        A[i][5] = pts2[i][1] 
        A[i][6] = pts1[i][0]
        A[i][7] = pts1[i][1]
        A[i][8] = 1.0  
    
    _,_,v = np.linalg.svd(A)
    # print("v", v)
    f_vec = v.transpose()[:,8]
    # print("f_vec = ", f_vec)
    f_hat = np.reshape(f_vec, (3,3))
    # print("Fmat = ", f_hat)

    # Enforce rank(F) = 2 
    s,v,d = np.linalg.svd(f_hat)
    f_hat = s @ np.diag([*v[:2], 0]) @ d

    return f_hat