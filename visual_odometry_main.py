"""
This file contains the code to perform visual odometry between two images. The output estimates the distance translation between image 1 and image 2. 
"""

from rospkg import RosPack
import cv2
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
import PyKDL
from helper_functions import calc_F, triangulation, get_keypoints, set_axes_equal

def test_opencv_recoverPose(pts_img1, pts_img2, K):
    """ Test out built out OpenCV build in 3D reconstruction function:
            cv2.recoverPose calcualgtes the relative camera rotation and translation 
            from an estimated essental matrix and the corresponsing points in two images. 
    """
    # dist_coeff = np.array([0.142705, -0.613601, 0.001541, 0.000374, 0.000000])
    dist_coeff = np.array([0.0,0.0,0.0,0.0,0.0])
    pts_img1 = np.array(pts_img1)
    pts_img2 = np.array(pts_img2)
    undistortPts1 = cv2.undistortPoints(np.expand_dims(pts_img1, axis=1), cameraMatrix=K, distCoeffs=dist_coeff)
    undistortPts2 = cv2.undistortPoints(np.expand_dims(pts_img2, axis=1), cameraMatrix=K, distCoeffs=dist_coeff)
    # print("undistortPts1: ", undistortPts1)

    # Find camera motion with cv2.recoverPose
    num_inliers, R, t, mask = cv2.recoverPose(E, undistortPts1, undistortPts2)
    print("num_inliers, R, t: ", num_inliers, R, t)
    # R_pykdl = PyKDL.Rotation(*list(R.flatten()))
    # print(R_pykdl.GetRotAngle())
    return num_inliers, R, t, mask

# Perform keypoint matching 
pts_img1, pts_img2, img3 = get_keypoints("image_1.png", "image_2.png")
print('number of keypoints', len(pts_img1))

# Find the fundamental matrix using the built in OpenCV function
epipolar_threshold = 0.006737946999085467
F, mask = cv2.findFundamentalMat(np.array([pts_img1]), np.array([pts_img2]),cv2.FM_RANSAC,epipolar_threshold)
# Alternative method of finding the fundamental matrix using our own funciton
# F = calc_F(pts_img1, pts_img2)

# Camera calibration matrix
K = np.array([[1013.109848, 0.000000, 493.049154],
              [0.000000, 1013.410857, 390.447766],
              [0.000000, 0.000000, 1.000000]])

# Special matrice W used for seperating essential matrix
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# Finding essential matrix
E = np.matmul(np.transpose(K), F, K)

# Separating the essential matrix into the rotational and translational elements using SVD
U, sigma, V = np.linalg.svd(E)

# Calcualte the two variations of the rotational and translational matrices
R1 = U.dot(W).dot(V)
R2 = U.dot(W.T).dot(V)
# print("R1, R2: ", R1, R2)
t1 = U[:,[2]]
t2 = -U[:,[2]]
# print("t1, t2: ", t1, t2)

# We can pair the above rotational and translational matrices in the following 4 ways.
# We set the location of the first camera (P1) to be at the origin (0,0,0) and the following
# matrices representt he location of the second camera (P2).
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2_1 = np.hstack([R1, t1])
P2_2 = np.hstack([R1, t2])
P2_3 = np.hstack([R2, t1])
P2_4 = np.hstack([R2, t2])
all_P2 = [P2_1, P2_2, P2_3, P2_4]

# To detemine which P2 matrix is the correct one, we use triangulation to check which 
# P matrix gives the highest number of correct keypoint matches. 
P2_dict = { 0:[], 
            1:[], 
            2:[], 
            3:[]}
for i in range(len(all_P2)):
    # triangulate all the points with P2: get xyz coordinates in 3D space
    for j in range(len(pts_img1)):
        pt_3d = triangulation(P1, all_P2[i], pts_img1[j], pts_img2[j])
        # print('3d point', pt_3d)
        # checking to see that z is greater than 0 because the point should be in front of the camera
        if (pt_3d[2] >= 0):
            P2_dict[i].append(pt_3d)
    
# The P2 matrix that yields the most frequent positive z direction is the most likely to be correct. 
correct_p2 = max(P2_dict, key=lambda k: len(P2_dict[k]))
correct_pts_3d = np.array(P2_dict[correct_p2])

# Get camera movement from our own method
print(all_P2[correct_p2])

# Get camera movement from built in openCV method: 
print(test_opencv_recoverPose(pts_img1, pts_img2, K))

def visualize():
    # plotting keypoints from img1 and img2
    plt.imshow(img3,),plt.show()

    # visualize 3d points resulting from triangulation:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_aspect('equal')ulat
    # plt.axis('equal')
    ax.scatter3D(correct_pts_3d[:,0], correct_pts_3d[:,1], correct_pts_3d[:,2], 'blue')
    set_axes_equal(ax)
    plt.show()

