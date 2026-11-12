import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *
from plotData_v2 import *
#from triangulation import triangulatePoints

IMG1 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image1.png"), cv2.COLOR_BGR2RGB)
IMG2 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image2.png"), cv2.COLOR_BGR2RGB)

def plotTransferredPointsHomography(img1, img2, H_21,colors=('r', 'g', 'b', 'c', 'm', 'y')):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(img1); ax1.set_title("Click points in image 1 (Enter to finish)")
    ax2.imshow(img2); ax2.set_title("Display the transferred point in image 2")

    points = []
    k = 0
    while True:
        plt.sca(ax1)
        pts = plt.ginput(1, timeout=0)
        if not pts:
            break
        u1, v1 = pts[0]
        c = colors[k % len(colors)]

        x1 = np.array([u1, v1, 1.0])
        
        x2 = H_21 @ x1

        x2 /= x2[2]

        u2, v2 = x2[0], x2[1] 

        ax1.plot(u1, v1, 'x', color=c, markersize=10)
        #plt.sca(ax2)
        ax2.plot(u2, v2, 'x', color=c, markersize=10)
        ax2.set_xlim(0, img2.shape[1]); ax2.set_ylim(img2.shape[0], 0)
        plt.draw()

        points.append(x2)
        k += 1
    plt.show(block=True)

    return np.array(points).T


def homographyFromGroundTruth(T_w_c1, T_w_c2, K_c1, K_c2, Pi_1):
    """
    Compute the homography H_21 induced by a plane, from the ground-truth poses.
    For points X on the plane Pi_1 (expressed in camera 1: n^T X + d = 0):
        x2 ~ H_21 @ x1,   H_21 = K_c2 @ (R_21 - t_21 @ n^T / d) @ K_c1^-1
    with (R_21, t_21) the relative pose that maps points from camera 1 to camera 2
    (T_21 = T_c2_c1 = inv(T_w_c2) @ T_w_c1).
     -input:
         T_w_c1: (4x4) pose of camera 1 in the world frame.
         T_w_c2: (4x4) pose of camera 2 in the world frame.
         K_c1: (3x3) intrinsic calibration matrix of camera 1.
         K_c2: (3x3) intrinsic calibration matrix of camera 2.
         Pi_1: (4,) plane [n, d] in camera 1 coordinates (n^T X + d = 0).
     -output:
         H_21: (3x3) homography (pixels image 1 -> pixels image 2), normalized so H_21[2,2] = 1.
    """
    # Plane: n^T X + d = 0
    n = Pi_1[0:3]
    d = Pi_1[3]

    # Relative pose: T_c2_w @ T_w_c1 = T_c2_c1 (maps points camera 1 -> camera 2)
    T_21 = np.linalg.inv(T_w_c2) @ T_w_c1
    R_21 = T_21[0:3, 0:3]
    t_21 = T_21[0:3, 3]

    # Homography induced by the plane
    H_21 = K_c2 @ (R_21 - np.outer(t_21, n) / d) @ np.linalg.inv(K_c1)

    # Normalize (H is defined up to scale)
    H_21 = H_21 / H_21[2, 2]
    return H_21


def toPixels(x):
    """
    Return the (2xN) pixel coordinates of 2xN points or 3xN homogeneous points.
    """
    return x[:2, :] / x[2, :] if x.shape[0] == 3 else x[:2, :]


def homographyFromMatches(x1, x2):
    """
    Estimate the homography H_21 (x2 ~ H_21 @ x1) with the linear DLT method.
    Each match gives two equations, linear in the 9 entries of H:
        [u1 v1 1  0  0  0  -u2*u1 -u2*v1 -u2] @ h = 0
        [0  0  0  u1 v1 1  -v2*u1 -v2*v1 -v2] @ h = 0
    Stacking N >= 4 matches gives A (2N x 9) @ h = 0, solved by SVD (last row of Vt).
     -input:
         x1: (2xN or 3xN) pixel points in image 1, N >= 4 (points on the plane).
         x2: (2xN or 3xN) corresponding pixel points in image 2.
     -output:
         H_21: (3x3) homography, normalized so H_21[2,2] = 1.
    """
    x1, x2 = toPixels(x1), toPixels(x2)
    u1, v1 = x1[0, :], x1[1, :]
    u2, v2 = x2[0, :], x2[1, :]

    ones = np.ones_like(u1)
    zeros = np.zeros_like(u1)

    # Two equations per match -> (N, 9) each
    A1 = np.stack([u1, v1, ones,
                   zeros, zeros, zeros,
                   -u2*u1, -u2*v1, -u2], axis=1)
    A2 = np.stack([zeros, zeros, zeros,
                   u1, v1, ones,
                   -v2*u1, -v2*v1, -v2], axis=1)
    A = np.vstack([A1, A2])                         # (2N, 9)

    # Solve A h = 0 -> last row of Vt (smallest singular value)
    _, _, Vt = np.linalg.svd(A)
    H_21 = Vt[-1, :].reshape(3, 3)

    # Normalize (H is defined up to scale)
    return H_21 / H_21[2, 2]


def homographyTransferError(H_21, x1, x2):
    """
    Transfer error of a homography: points of image 1 are mapped to image 2 with
    H_21 and compared with the measured points in image 2.
        e_i = || x2_i - pixels(H_21 @ x1_i) ||
     -input:
         H_21: (3x3) homography (image 1 -> image 2).
         x1: (2xN or 3xN) pixel points in image 1.
         x2: (2xN or 3xN) corresponding pixel points in image 2.
     -output:
         rmse: sqrt(mean(e_i^2)) in pixels.
    """
    x1, x2 = toPixels(x1), toPixels(x2)
    x1_h = np.vstack((x1, np.ones((1, x1.shape[1]))))
    x2_pred = toPixels(H_21 @ x1_h)
    errors = np.linalg.norm(x2 - x2_pred, axis=0)   # (N,) one error per point
    return np.sqrt(np.mean(errors**2))


if __name__ == '__main__':

    print("Homography...")

    #3.1 Homography definition
    T_w_c1      = np.loadtxt(DATA_LAB + 'T_w_c1.txt')
    T_w_c2      = np.loadtxt(DATA_LAB + 'T_w_c2.txt')
    Kc          = np.loadtxt(DATA_LAB + 'K_c.txt')
    Pi_1        = np.loadtxt(DATA_LAB + 'Pi_1.txt')
    X_w         = np.loadtxt(DATA_LAB + 'X_w.txt')
    x1FloorData = np.loadtxt(DATA_LAB + 'x1FloorData.txt')
    x2FloorData = np.loadtxt(DATA_LAB + 'x2FloorData.txt')

    H_21 = homographyFromGroundTruth(T_w_c1, T_w_c2, Kc, Kc, Pi_1)
    print("H_21:")
    print(H_21)

    #3.2 Point transfer visualization
    plotTransferredPointsHomography(IMG1,IMG2,H_21)

    #3.3 Homography linear estimation from matches
    H_est = homographyFromMatches(x1FloorData, x2FloorData)
    print("H_21 estimated from matches:")
    print(H_est)

    # Evaluate the accuracy: estimated vs ground-truth homography on the same matches
    rmse_est = homographyTransferError(H_est, x1FloorData, x2FloorData)
    rmse_gt  = homographyTransferError(H_21,  x1FloorData, x2FloorData)
    print(f"Transfer RMSE: estimated H {rmse_est:.4f} px, ground-truth H {rmse_gt:.4f} px")
    plotTransferredPointsHomography(IMG1,IMG2,H_est)