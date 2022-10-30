# find good features

# carry out optical flow

#  nearest neighbor search? dont really understand this part

# ratio test -  its nice to have a slider and visual by this point

from rospkg import RosPack
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from triangulation import calc_F


def get_keypoints(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img2_path, cv2.COLOR_BGR2GRAY)
    # Create sift detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 2
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 7)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good_matches = []
    pts_img1 = []
    pts_img2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.35 * n.distance:
            pts_img2.append(kp2[m.trainIdx].pt)
            pts_img1.append(kp1[m.queryIdx].pt)
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    return pts_img1, pts_img2


pts_img1, pts_img2 = get_keypoints("image_1.png", "image_2.png")

# printing out points
# print('points image 1', pts_img1)
# print('points image 2', pts_img2)

# F, mask = cv2.findFundamentalMat(pts_img1,pts_img2,cv2.FM_RANSAC)

# finding fundamental matrix
F = calc_F(pts_img1, pts_img2)

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

# options for R and T
R1 = np.matmul(U, W, np.transpose(V))
R2 = np.matmul(U, np.transpose(W), np.transpose(V))
t1 = U[:,[2]]
t2 = -U[:,[2]]

# four options for camera matrices
P1_1 = np.hstack([R1, t1])
P1_2 = np.hstack([R1, -t1])
P1_3 = np.hstack([R2, t1])
P1_4 = np.hstack([R2, -t1])

print('P1', P1_1, 'P1', P1_2)