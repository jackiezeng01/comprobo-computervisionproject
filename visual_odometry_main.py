# find good features

# carry out optical flow

#  nearest neighbor search? dont really understand this part

# ratio test -  its nice to have a slider and visual by this point

from rospkg import RosPack
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from helper_functions import calc_F, triangulation, get_keypoints, set_axes_equal


pts_img1, pts_img2, img3 = get_keypoints("image_1.png", "image_2.png")
print('number of keypoints', len(pts_img1))
# printing out points
# print('points image 1', pts_img1)
# print('points image 2', pts_img2)

# F, mask = cv2.findFundamentalMat(pts_img1,pts_img2,cv2.FM_RANSAC)

# finding fundamental matrix
# F = calc_F(pts_img1, pts_img2)
epipolar_threshold = 0.006737946999085467
F, mask = cv2.findFundamentalMat(np.array([pts_img1]), np.array([pts_img2]),cv2.FM_RANSAC,epipolar_threshold)

# writing out camera calibration matrix
K = np.array([[1013.109848, 0.000000, 493.049154],
              [0.000000, 1013.410857, 390.447766],
              [0.000000, 0.000000, 1.000000]
])

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

print("p2-1: ", P2_1)
print("p2-4: ", P2_2)


all_P2 = [P2_1, P2_2, P2_3, P2_4]
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

# find correct P2 by checking to see which P matrix gives us the 
# highest number of correct points
P2_dict = { 0:[], 
            1:[], 
            2:[], 
            3:[]}
for i in range(len(all_P2)):
    # triangulate all the points with P2: get xyz coordinates in 3D space
    for j in range(len(pts_img1)):
        pt_3d = triangulation(P1, all_P2[i], pts_img1[j], pts_img2[j])
        # print('3d point', pt_3d)
        # checking to see that z is greater than 0
        if (pt_3d[2] >= 0):
            P2_dict[i].append(pt_3d)
    
# Determine the most frequent z direction:
correct_p2 = max(P2_dict, key=lambda k: len(P2_dict[k]))
distanceThresh = 1000
R = []
t = []
mask = 0
triangulatedPoints = []
# cv2.recoverPose(E, pts_img1, pts_img2, K, distanceThresh[, R[, t[, mask[, triangulatedPoints]]]])
# correct_pts_3d, R, t, mask = cv2.recoverPose(E, pts_img1, pts_img2)

correct_pts_3d = np.array(P2_dict[correct_p2])

print('all correct points', correct_pts_3d)
print('number of correct points',len(correct_pts_3d))
for i in range(0,4):
    print(len(P2_dict[i]))

# # plotting keypoints from img1 and img2
# plt.imshow(img3,),plt.show()

# # visualize 3d points resulting from triangulation:
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.set_aspect('equal')ulat
# # plt.axis('equal')
# ax.scatter3D(correct_pts_3d[:,0], correct_pts_3d[:,1], correct_pts_3d[:,2], 'blue')
# set_axes_equal(ax)
# plt.show()

# get distance translation
print(all_P2[correct_p2])


