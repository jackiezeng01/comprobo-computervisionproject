import numpy as np
import cv2
from helper_functions import calc_F, triangulation, get_keypoints, set_axes_equal

P1 = np.array([[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]]);

P2 = np.array([[1.0, 0.0, 0.0, 3],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0]]); # camera matrix change this

# pt3d_1 = (3,5,2,1)
# pt3d_2 = (3,4,2,1)
# pt3d_3 = (3,3,2,1)
# pt3d_4 = (3,2,2,1)
# pt3d_5 = (3,5,1,1)
# pt3d_6 = (3,5,4,1)
# pt3d_7 = (3,1,2,1)
# pt3d_8 = (2,1,2,1)
# pt3d_9 = (2,5,2,1)
# pt3d_10 = (4,5,2,1)
# pt3d_11 = (6,2,2,1)
# pt3d_12 = (1,5,2,1)
# pt3d_13 = (1,2,2,1)
# pt3d_14 = (7,2,2,1)
# pt3d_15 = (7,2,1,1)
# pt3d_16 = (7,1,2,1)
pt3d_1 = np.array([[3],[5],[2], [1]])
pt3d_2 = np.array([[3],[4],[2], [1]])
pt3d_3 = np.array([[4],[3],[2], [1]])
pt3d_4 = np.array([[3],[4],[2], [1]])
pt3d_5 = np.array([[6],[4],[1], [1]])
pt3d_6 = np.array([[7],[5],[1], [1]])
pt3d_7 = np.array([[1],[4],[2], [1]])
pt3d_8 = np.array([[1],[2],[2], [1]])
pt3d_9 = np.array([[1],[4],[3], [1]])
pt3d_10 = np.array([[4],[1],[2], [1]])
pt3d_11 = np.array([[4],[1],[3], [1]])
pt3d_12 = np.array([[4],[2],[4], [1]])
pt3d_13 = np.array([[4],[2],[5], [1]])
pt3d_14 = np.array([[4],[2],[1], [1]])
pt3d_15 = np.array([[2],[1],[2], [1]])
pt3d_16 = np.array([[2],[2],[3], [1]])
pts_3d = [pt3d_1, pt3d_2, pt3d_3, pt3d_4, pt3d_5, pt3d_6, pt3d_7, pt3d_8, pt3d_9, pt3d_10, pt3d_11, pt3d_12, pt3d_13, pt3d_14, pt3d_15, pt3d_16]
pts_cam1 = []
pts_cam2 = []

'''
# PAUL'S CODE
# Haven't gotten this to work yet

def test_epipolar(E,pt1,pt2):
    """ Test how well two points fit the epipolar constraint using
        the formula pt1'*E*pt2 """
    pt1_h = np.zeros((3,1))
    pt2_h = np.zeros((3,1))
    pt1_h[0:2,0] = pt1.T
    pt2_h[0:2,0] = pt2.T
    pt1_h[2] = 1.0
    pt2_h[2] = 1.0
    return pt2_h.T.dot(E).dot(pt1_h)

for i in range(len(pts_3d)):
    pts_cam1.append(np.matmul(P1,pts_3d[i]))
    pts_cam2.append(np.matmul(P2,pts_3d[i]))

    E, mask = cv2.findFundamentalMat(np.array(pts_cam1),np.array(pts_cam2),cv2.FM_RANSAC)

    im1_pts_ud_fixed, im2_pts_ud_fixed = cv2.correctMatches(E, pts_cam1, pts_cam2)
    use_corrected_matches = True
    if not(use_corrected_matches):
        im1_pts_ud_fixed = pts_cam1
        im2_pts_ud_fixed = pts_cam2

    epipolar_error = np.zeros((im1_pts_ud_fixed.shape[1],))
    for i in range(im1_pts_ud_fixed.shape[1]):
        epipolar_error[i] = test_epipolar(E,im1_pts_ud_fixed[0,i,:],im2_pts_ud_fixed[0,i,:])

    W = np.array([[0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]])
    K = np.array([[1013.109848, 0.000000, 493.049154],
                [0.000000, 1013.410857, 390.447766],
                [0.000000, 0.000000, 1.000000]
    ])
    # calculate F since we know K
    F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
    U, Sigma, V = np.linalg.svd(E)

    # these are the two possible rotation matrices
    R1 = U.dot(W).dot(V)
    R2 = U.dot(W.T).dot(V)

    if np.linalg.det(R1)+1.0 < 10**-8:
        # if we accidentally got a rotation matrix with a negative determinant,
        # flip sign of E and recompute everything
        E = -E
        F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
        U, Sigma, V = np.linalg.svd(E)

        R1 = U.dot(W).dot(V)
        R2 = U.dot(W.T).dot(V)

    # these are the two translations that are possible
    t1 = U[:,2]
    t2 = -U[:,2]

    P = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]]);
    P1_possibilities = []
    P1_possibilities.append(np.column_stack((R1,t1)))
    P1_possibilities.append(np.column_stack((R1,t2)))
    P1_possibilities.append(np.column_stack((R2,t1)))
    P1_possibilities.append(np.column_stack((R2,t2)))
    print(P1_possibilities)
'''

# OUR CODE
#
#
epipolar_threshold = 0.006737946999085467
F, mask = cv2.findFundamentalMat(np.array([pts_cam1]), np.array([pts_cam2]),cv2.FM_RANSAC,epipolar_threshold)

# writing out camera calibration matrix
# K = np.array([[1013.109848, 0.000000, 493.049154],
#               [0.000000, 1013.410857, 390.447766],
#               [0.000000, 0.000000, 1.000000]
# ])
K = np.array([[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]);
    
# special matrice W used for seperating essential matrix
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# finding essential matrix
E = np.matmul(np.transpose(K), F, K)

# seperating into rotational and translation matrice options
U, sigma, V = np.linalg.svd(E)

# # testing out default function
# [R, t, good, mask, triangulatedPoints] = cv2.recoverPose(E, pts_img1, pts_img2)
# options for R and T
# R1 = np.matmul(U, W, np.transpose(V))
# R2 = np.matmul(U, np.transpose(W), np.transpose(V))
R1 = U.dot(W).dot(V)
R2 = U.dot(W.T).dot(V)
t1 = U[:,[2]]
t2 = -U[:,[2]]

# four options for camera matrices
P2_1 = np.hstack([R1, t1])
P2_2 = np.hstack([R1, t2])
P2_3 = np.hstack([R2, t1])
P2_4 = np.hstack([R2, t2])
print(P2_1)
print(P2_2)
print(P2_3)
print(P2_4)


# TEST 3D pt -> 2D projections -> recovered 3D pt
recovered_pts = []
for i in range(len(pts_3d)):
    recovered_pts.append(triangulation(P1, P2, pts_cam1[i], pts_cam2[i]))

print(recovered_pts)

x = 2
y = 3
z = 9
pt3d = np.array([[x],[y],[z], [1]])
u1 = np.matmul(K @ P1,pt3d)
print(u1)
u2 = np.matmul(K @ P2,pt3d)
# divide by u2[2]
recovered = triangulation(P1, P2, u1, u2)
print(recovered)
