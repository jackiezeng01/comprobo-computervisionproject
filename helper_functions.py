import numpy as np
from numpy.linalg import eig
import cv2


def triangulation (P1, P2, u1, u2):
    """
    P1: camera 1 matrix (at origin), numpy array, shape  (3,4)
    P2: camera 2 matrix (4 possibilites), numpy array, shape (3,4)

    For a single keypoint pair:
        u1: numpy array, shape (3,1) - (x,y,1) coordinate for the first image, 2d projections
        u2: numpy array, shape (3,1) - (x,y,1) coordinate for the first image, 2d projections
    """
    u1 = np.array([u1[0], u1[1], 1])
    u2 = np.array([u2[0], u2[1], 1])

    # print(u1)
    u1 = u1/u1[2] # 3d point projected on the first camera's 2 view
    u2 = u2/u2[2] 
    A = np.zeros((4,4))
    A[0,:] = u1[0] * P1[2,:] - P1[0,:]
    A[1,:] = u1[1] * P1[2,:] - P1[1,:]
    A[2,:] = u2[0] * P2[2,:] - P2[0,:]
    A[3,:] = u2[1] * P2[2,:] - P2[1,:]

    A_final = np.matmul(np.transpose(A), A)
    e_vals, e_vecs = eig(A_final)
    e_vals = list(e_vals)
    idx = e_vals.index(min(e_vals))
    x = e_vecs[:,idx]/e_vecs[3,idx]
    return x

# example
# P1 = np.array([[1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0]]);

# P2 = np.array([[1.0, 0.0, 0.0, 3],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0]]); # camera matrix change this

# x = 3
# y = 5
# z = 2
# pt3d = np.array([[x],[y],[z], [1]])
# pt3d in 2d:
    # u1 = np.matmul(P1,pt3d)
    # u2 = np.matmul(P2,pt3d)
# recovered = triangulation(P1, P2, u1, u2)
# print(recovered)

# epipolar_threshold = 0.006737946999085467
# E, mask = cv2.findFundamentalMat(u1,u2,cv2.FM_RANSAC,epipolar_threshold)


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
        if m.distance < 0.38 * n.distance:
            pts_img2.append(kp2[m.trainIdx].pt)
            pts_img1.append(kp1[m.queryIdx].pt)
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    return pts_img1, pts_img2, img3

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
