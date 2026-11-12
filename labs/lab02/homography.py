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
    u1, v1 = x1[0, :], x1[1, :]
    u2, v2 = x2[0, :], x2[1, :]

    ones = np.ones_like(u1)
    zeros = np.zeros_like(u1)

    #Construct A
    A1 = np.stack([
        u1, v1, ones,
        zeros, zeros, zeros,
        -u2*u1, -u2*v1, -u2
    ], axis=1)

    A2 = np.stack([
        zeros, zeros, zeros,
        u1, v1, ones,
        -v2*u1, -v2*v1, -v2
    ], axis=1)

    print(A1.shape)  # (N, 9)
    print(A2.shape)  # (N, 9)

    #combine the equations:
    A = np.vstack([A1, A2])

    print(A.shape) # (2*N, 9)

    #Estimate the homography using SVD
    #solve AH = 0

    _, _, Vt = np.linalg.svd(A)

    h = Vt[-1, :]
    H_est = h.reshape(3, 3)

    H_est = H_est / H_est[2, 2]

    #Evaluate the accuracy

    x2_pred_h = H_est @ x1

    x2_pred = x2_pred_h[:2, :] / x2_pred_h[2, :]

    errors = np.linalg.norm(x2[:2, :] - x2_pred, axis=0)

    rmse = np.sqrt(np.mean(errors**2))

    print("Homography RMSE (pixels):", rmse)
