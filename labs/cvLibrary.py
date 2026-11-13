import matplotlib.pyplot as plt
import scipy.linalg as scAlg
import numpy as np
import cv2

DATA_LAB_1 = 'data/'
DATA_LAB = 'data/'

#########################################################################
#--Lab 01--
#########################################################################
def assemble_transform(R:np.ndarray,t:np.ndarray) -> np.ndarray:
    """Construct a 4x4 homogeneous rigid transformation from R and t."""
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = np.array(t).reshape(3)
    return T

def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 homogeneous rigid transformation matrix (SE3).

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        T_inv: 4x4 inverse transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv

def  computeProjectionFromKRT(Kc:np.ndarray,R_wc:np.ndarray,t_wc:np.ndarray)-> np.ndarray:
    # Return 3x4 camera Matrix
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    t_cw = t_cw.reshape(3,1)
    P = Kc @ np.hstack((R_cw,t_cw))

    return P

def computeProjectionMatrix(K: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    """
    Compute a 3x4 camera projection matrix.

    Args:
        K: 3x3 camera intrinsic matrix.
        T_cw: 4x4 transformation from world to camera coordinates.

    Returns:
        P: 3x4 projection matrix.
    """
    I = np.hstack((np.eye(3), np.zeros((3, 1))))
    P = K @ I @ T_cw

    return P

def projectPoints(P:np.ndarray,X:np.ndarray) -> np.ndarray:
    """
    Project 3D Points into an image

    Args:
        P: 3x4
        X: 4xN
    Returns:
        x: 2xN Dehomogenize.
    """
    S = P @ X              # 3xN (s*u, s*v, s)
    uv = S[:2,:] / S[2:,:] # # divide each row by s, Dehomogenize
    return uv

def lineFromPoints(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute the homogeneous 2D line through two points.

    Args:
        p1: Homogeneous 2D point.
        p2: Homogeneous 2D point.

    Returns:
        l: Homogeneous line coefficients (a, b, c).
    """
    p1 = np.asarray(p1).reshape(3)
    p2 = np.asarray(p2).reshape(3)

    l = np.cross(p1, p2)

    return l

def fit_line_with_svd(X):  # X is 3xN homogeneous points (like x or xGT)
    A = X.T                 # N x 3
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    l = vh[-1, :]           # (a,b,c)
    return u, s, vh, l   
#########################################################################

#########################################################################
#--Lab 02--
#########################################################################
#***********************************************************************
#+++Fundamental
#***********************************************************************
def drawLine(l, color='r', lWidth=1):
    p_l_y = np.hstack((0, -l[2] / l[1]))
    p_l_x = np.hstack((-l[2] / l[0], 0))
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], color, linewidth=lWidth)

def triangulatePoints(P1, P2, x1, x2):
    """
    Triangulate 3D points from two views with the linear (DLT) method.
    For each correspondence, the projection x ~ P @ X gives two linear equations per view:
        u * P[2,:] - P[0,:]  ->  (u * p3 - p1) @ X = 0
        v * P[2,:] - P[1,:]  ->  (v * p3 - p2) @ X = 0
    Stacking both views gives A (4x4) @ X = 0, solved by SVD (last row of Vh).
    All N points are solved at once (batched SVD).
     -input:
         P1: (3x4) projection matrix of camera 1 (K @ [R|t], world -> image 1).
         P2: (3x4) projection matrix of camera 2 (K @ [R|t], world -> image 2).
         x1: (2xN or 3xN) pixel points in image 1 (only the first two rows are used).
         x2: (2xN or 3xN) corresponding pixel points in image 2.
     -output:
         X: (3xN) triangulated points in Euclidean world coordinates.
    """
    u1, v1 = x1[0, :], x1[1, :]
    u2, v2 = x2[0, :], x2[1, :]

    # Two equations per view, one (2x4) block per point -> (N, 2, 4)
    A1 = np.stack([u1[:, None] * P1[2, :] - P1[0, :],
                   v1[:, None] * P1[2, :] - P1[1, :]], axis=1)
    A2 = np.stack([u2[:, None] * P2[2, :] - P2[0, :],
                   v2[:, None] * P2[2, :] - P2[1, :]], axis=1)
    A = np.concatenate([A1, A2], axis=1)            # (N, 4, 4)

    # Solve A @ X = 0 for each point: last row of Vh -> (N, 4)
    _, _, vh = np.linalg.svd(A)
    X_h = vh[:, -1, :].T                            # (4, N) homogeneous

    # Homogeneous -> Euclidean
    return X_h[0:3, :] / X_h[3, :]


def plotEpipolarLinesInteractive(img1, img2, F_21, e1=None, e2=None, lineStyle='--',
                                 colors=('r', 'g', 'b', 'c', 'm', 'y'),title=None):
    """
    Click points in image 1 and draw their epipolar lines in image 2.
    Press Enter (or right-click) to finish.
     -input:
         img1, img2: images.
         F_21: (3x3) fundamental matrix mapping points in image 1 to lines in image 2 (l2 = F_21 @ x1).
         e1: (3,) epipole in image 1 (homogeneous, e1[2] = 1). If given, drawn in image 1.
         e2: (3,) epipole in image 2 (homogeneous, e2[2] = 1). If given, drawn in image 2;
             all epipolar lines in image 2 should pass through it.
         lineStyle: matplotlib line style of the epipolar lines: '-', '--', ':', '-.'.
         colors: colors cycled for each point/line pair.
     -output:
         x1: (3xN) clicked points in homogeneous coordinates.
         l2: (3xN) corresponding epipolar lines in image 2.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    if title is not None:
        fig.suptitle(title, fontsize=14)
    ax1.imshow(img1); ax1.set_title("Click points in image 1 (Enter to finish)")
    ax2.imshow(img2); ax2.set_title("Epipolar lines in image 2")

    # Draw the epipoles (skipped if at infinity, e[2] ~ 0)
    for ax, e, name in ((ax1, e1, 'e1'), (ax2, e2, 'e2')):
        if e is not None and abs(e[2]) > 1e-10:
            ax.plot(e[0] / e[2], e[1] / e[2], 'o', color='blue',
                    markeredgecolor='k', markersize=10)
            ax.text(e[0] / e[2] + 10, e[1] / e[2] - 10, name, color='blue')

    points, lines = [], []
    k = 0
    while True:
        plt.sca(ax1)
        pts = plt.ginput(1, timeout=0)
        if not pts:
            break
        u1, v1 = pts[0]
        c = colors[k % len(colors)]

        x1 = np.array([u1, v1, 1.0])
        l2 = F_21 @ x1

        ax1.plot(u1, v1, 'x', color=c, markersize=10)
        plt.sca(ax2)
        drawLine(l2, c + lineStyle)     # e.g. 'r--' (dashed) or 'r:' (dotted)
        ax2.set_xlim(0, img2.shape[1]); ax2.set_ylim(img2.shape[0], 0)
        plt.draw()

        points.append(x1)
        lines.append(l2)
        k += 1

    plt.show(block=True)
    return np.array(points).T, np.array(lines).T

def skew(t):
    """
    Build the skew-symmetric matrix [t]x of a 3D vector, such that
    [t]x @ v == np.cross(t, v) for any 3D vector v.
     -input:
         t: (3,) vector (tx, ty, tz).
     -output:
         (3x3) skew-symmetric matrix:
             [  0  -tz   ty ]
             [ tz    0  -tx ]
             [-ty   tx    0 ]
    """    
    tx, ty, tz = t
    return np.array([
        [0, -tz, ty],
        [tz, 0, -tx],
        [-ty, tx, 0]
    ])

def essentialFromGroundTruth(T_wc1,T_wc2):
    """
    Compute the essential matrix E_21 from the ground-truth poses of two cameras.
    E_21 satisfies the epipolar constraint x2^T @ E_21 @ x1 = 0, where x1 and x2
    are corresponding points in normalized (calibrated) coordinates of camera 1 and 2.
        E_21 = [t_21]x @ R_21
    with (R_21, t_21) the relative pose that maps points from camera 1 to camera 2
    (T_21 = T_c2_c1 = inv(T_w_c2) @ T_w_c1).
     -input:
         T_wc1: (4x4) pose of camera 1 in the world frame (T_w_c1).
         T_wc2: (4x4) pose of camera 2 in the world frame (T_w_c2).
     -output:
         E_21: (3x3) essential matrix, defined up to scale.
    """
    T_21 = np.linalg.inv(T_wc2) @ T_wc1  # T_c2_c1: maps points camera 1 -> camera 2
    #T_12 = np.linalg.inv(T_wc1) @ T_wc2 # T_c1_c2 (= T_12): maps points camera 2 -> camera 1

    R_21 = T_21[:3,:3]
    t_21  = T_21[:3,3]

    tx = skew(t_21)

    #Essesntial Matrix
    E_21 = tx @ R_21
    
    # Optional normalization (E is defined up to scale)
    E_21 /= np.linalg.norm(E_21)

    return E_21

def fundamentalMatrixFromEssential(E,K_c1,K_c2):
    """
    Compute the fundamental matrix F_21 from the Essential matrix E_21.
    F_21 satisfies the epipolar constraint x2^T @ F_21 @ x1 = 0, where x1 and x2
    are corresponding points in pixel (homogeneous) coordinates of image 1 and 2.
        F_21 = K_c2^-T @ E_21 @ K_c1^-1

     -input:
         E: (3x3) essential matrix, defined up to scale.
         K_c1: (3x3) Camera intrinsic parameter for camera 1.
         K_c2: (3x3) Camera intrinsic parameter for camera 2.
     -output:
         F: (3x3) Fundamental matrix, normalized to unit Frobenius norm.
    """
    F = np.linalg.inv(K_c2.T) @ E @ np.linalg.inv(K_c1)
    F /= np.linalg.norm(F)
    return F

def fundamentalFromEightPoints(x1, x2):
    """
    Estimate the fundamental matrix F_21 with the (linear) eight-point method.
    Each correspondence gives one equation x2^T @ F_21 @ x1 = 0, linear in the
    9 entries of F_21. Stacking N >= 8 of them gives A @ f = 0, solved by SVD.
    Rank 2 is then enforced (smallest singular value set to 0).
     -input:
         x1: (3xN) homogeneous pixel points in image 1, N >= 8.
         x2: (3xN) corresponding homogeneous pixel points in image 2.
     -output:
         F_21: (3x3) fundamental matrix of rank 2, normalized to unit Frobenius norm.
    """
    # 1. Build A (N x 9): one row per correspondence
    u1, v1 = x1[0], x1[1]
    u2, v2 = x2[0], x2[1]
    ones = np.ones_like(u1)
    A = np.stack([u2*u1, u2*v1, u2,
                  v2*u1, v2*v1, v2,
                  u1,    v1,    ones], axis=1)

    # 2. Solve A f = 0 -> last row of Vh (smallest singular value)
    _, _, vh = np.linalg.svd(A)
    F_21 = vh[-1, :].reshape(3, 3)

    # 3. Enforce rank(F) = 2
    U, S, Vt = np.linalg.svd(F_21)
    S[-1] = 0
    F_21 = U @ np.diag(S) @ Vt

    F_21 /= np.linalg.norm(F_21)
    return F_21

def essentialFromEightPoints(x1, x2, K_c1, K_c2):
    """
    Estimate the essential matrix E_21 with the (linear) eight-point method.
    Pixel points are first converted to normalized coordinates (x_n = K^-1 @ x),
    where the epipolar constraint reads x2n^T @ E_21 @ x1n = 0. The linear system
    is the same as for F, so fundamentalFromEightPoints is reused on normalized points.
    Then the essential-matrix constraint is enforced: singular values (1, 1, 0).
     -input:
         x1: (3xN) homogeneous pixel points in image 1, N >= 8.
         x2: (3xN) corresponding homogeneous pixel points in image 2.
         K_c1: (3x3) intrinsic calibration matrix of camera 1.
         K_c2: (3x3) intrinsic calibration matrix of camera 2.
     -output:
         E_21: (3x3) essential matrix, normalized to unit Frobenius norm.
    """
    #from sympy import symbols
    #u1, u2, v1, v2 = symbols('u1,u2,v1,v2')
    #u1, u2, v1, v2 = symbols('u1,u2,v1,v2')
    #e11,e12,e13,e21,e22,e23,e31,e32,e33=symbols('e11,e12,e13,e21,e22,e23,e31,e32,e33')
    #P1 = np.array([u1,v1,1])
    #P2 = np.array([u2,v2,1])
    #E = np.array([[e11,e12,e13],[e21,e22,e23],[e31,e32,e33]])
    #P2.T @ E @ P1 = 0
    #QxE =0
    #But we can define
    #E = np.ones((3,3))
    #Then:
    #Q =  x2.T @ E @ x1 

    # 1. Pixels -> normalized coordinates
    x1n = np.linalg.inv(K_c1) @ x1
    x2n = np.linalg.inv(K_c2) @ x2

    # 2. Same linear solve as for F (A f = 0 + rank 2)
    E_21 = fundamentalFromEightPoints(x1n, x2n)

    # 3. Enforce essential-matrix constraints: singular values (1, 1, 0)
    U, _, Vt = np.linalg.svd(E_21)
    E_21 = U @ np.diag([1, 1, 0]) @ Vt

    E_21 /= np.linalg.norm(E_21)
    return E_21

def computeEpipoles(F_21, eps=1e-10):
    """
    Compute the epipoles of both images from the fundamental matrix F_21.
        e1: right null vector of F_21  ->  F_21 @ e1 = 0     (last row of Vt)
        e2: left null vector of F_21   ->  e2^T @ F_21 = 0   (last column of U)
    e1 is the projection of camera 2's centre in image 1, and e2 the projection
    of camera 1's centre in image 2. All epipolar lines of an image pass through its epipole.
     -input:
         F_21: (3x3) fundamental matrix of rank 2 (x2^T @ F_21 @ x1 = 0).
         eps: threshold below which the third coordinate is considered 0.
     -output:
         e1: (3,) epipole in image 1, homogeneous with e1[2] = 1
             (left unnormalized if the epipole is at infinity, e1[2] ~ 0).
         e2: (3,) epipole in image 2, same convention.
    """
    U, _, Vt = np.linalg.svd(F_21)
    e1 = Vt[-1, :]
    e2 = U[:, -1]

    # Convert to pixel coordinates (only if the epipole is not at infinity)
    if abs(e1[2]) > eps:
        e1 = e1 / e1[2]
    if abs(e2[2]) > eps:
        e2 = e2 / e2[2]

    return e1, e2

def essentialFromFundamental(F,Kc1,Kc2):

    """
    Compute the essential matrix E_21 from the fundamental matrix F_21.
    Inverse of F_21 = Kc2^-T @ E_21 @ Kc1^-1, i.e.
        E_21 = Kc2^T @ F_21 @ Kc1
    E_21 satisfies x2n^T @ E_21 @ x1n = 0 with x1n, x2n in normalized
    coordinates (x_n = K^-1 @ x_pixels).
     -input:
         F: (3x3) fundamental matrix F_21 (x2^T @ F_21 @ x1 = 0 in pixels).
         Kc1: (3x3) intrinsic calibration matrix of camera 1.
         Kc2: (3x3) intrinsic calibration matrix of camera 2.
     -output:
         E: (3x3) essential matrix E_21 with singular values (sigma, sigma, 0),
            defined up to scale.
    """
    E = Kc2.T @ F @ Kc1
    # A valid essential matrix E = [t]x @ R has singular values (sigma, sigma, 0):
    # [t]x has singular values (|t|, |t|, 0) and the rotation R does not change them.
    # With noisy data (e.g. F from the eight-point method) the product above is only
    # approximately essential, e.g. (1.02, 0.97, 0.003). We project it onto the
    # closest valid essential matrix (in Frobenius norm): keep U and Vt, set the two
    # largest singular values to their mean and the smallest to 0. This is needed so
    # that the epipoles exist (rank 2) and E can be decomposed into R and t.
    U, S, Vt = np.linalg.svd(E)
    sigma = (S[0] + S[1]) / 2
    E = U @ np.diag([sigma, sigma, 0]) @ Vt
    return E

def decomposeEssential(E_21):
    """
    Decompose the essential matrix E_21 into the four possible relative poses.
    E_21 = U diag(1,1,0) Vt  ->  R = U W Vt or U W^T Vt,  t = +-U[:, 2]
     -input:
         E_21: (3x3) essential matrix.
     -output:
         T_list: list of four (4x4) candidate transformations T_c2_c1.
                 t has unit norm (scale cannot be recovered from E).
    """
    U, _, Vt = np.linalg.svd(E_21)

    # Ensure R = U W Vt is a rotation (det = +1), not a reflection
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R_list = [U @ W @ Vt, U @ W.T @ Vt]
    t = U[:, -1]
    t_list = [t, -t]

    T_list = []
    for R in R_list:
        for t in t_list:
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            T_list.append(T)
    return T_list


def recoverPoseFromEssential(E_21, K_c1, K_c2, x1, x2):
    """
    Recover the relative pose T_21 (camera 1 -> camera 2) from the essential matrix.
    Among the four candidates of decomposeEssential, keep the one that places the
    most triangulated points in front of both cameras (cheirality: depth Z > 0).
    Camera 1 is the reference frame: P1 = K_c1 [I | 0].
     -input:
         E_21: (3x3) essential matrix (x2n^T @ E_21 @ x1n = 0).
         K_c1: (3x3) intrinsic calibration matrix of camera 1.
         K_c2: (3x3) intrinsic calibration matrix of camera 2.
         x1: (2xN or 3xN) pixel points in image 1.
         x2: (2xN or 3xN) corresponding pixel points in image 2.
     -output:
         T_21: (4x4) transformation T_c2_c1 (points in camera 1 -> camera 2),
               translation up to scale (unit norm).
         X: (3xN) triangulated points in camera 1 coordinates for the chosen T_21.
    """
    P1 = K_c1 @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_score, T_21, X_best = -1, None, None
    for T in decomposeEssential(E_21):
        P2 = computeProjectionMatrix(K_c2, T)
        X = triangulatePoints(P1, P2, x1, x2)          # camera 1 frame
        X_c2 = T[:3, :3] @ X + T[:3, 3:4]              # camera 2 frame

        # Cheirality: in front of BOTH cameras
        score = np.sum((X[2, :] > 0) & (X_c2[2, :] > 0))
        if score > best_score:
            best_score, T_21, X_best = score, T, X

    return T_21, X_best


def alignToWorld(T_21, X_c1, T_w_c1, scale):
    """
    Express a two-view reconstruction (relative to camera 1, unknown scale) in the
    world frame of the ground truth, so both can be compared directly.
    The scale cannot be recovered from E (||t_21|| = 1), so it is taken from the
    ground truth (true baseline). Then everything is moved from camera 1 to world
    with the known pose of camera 1:
        X_w   = T_w_c1 @ (scale * X_c1)
        T_w_c2 = T_w_c1 @ inv(T_21 with t scaled)
     -input:
         T_21: (4x4) estimated T_c2_c1, translation with unit norm.
         X_c1: (3xN) triangulated points in camera 1 coordinates (same unit scale).
         T_w_c1: (4x4) ground-truth pose of camera 1 in the world frame.
         scale: true baseline length ||t_21_gt|| (world units).
     -output:
         T_w_c2_est: (4x4) estimated pose of camera 2 in the world frame.
         X_w_est: (3xN) estimated 3D points in the world frame.
    """
    # 1. Fix the scale: translation and points scale together
    T_21_s = T_21.copy()
    T_21_s[:3, 3] *= scale
    X_c1_s = scale * X_c1

    # 2. Camera 1 frame -> world frame
    T_w_c2_est = T_w_c1 @ np.linalg.inv(T_21_s)
    X_c1_h = np.vstack((X_c1_s, np.ones((1, X_c1_s.shape[1]))))
    X_w_est = (T_w_c1 @ X_c1_h)[:3, :]

    return T_w_c2_est, X_w_est


def rotationError(R_est, R_gt):
    """
    Angular error between two rotations (geodesic distance on SO(3)).
    R_err = R_est^T @ R_gt is the residual rotation that takes the estimate to the
    ground truth (identity if both are equal). Its rotation angle is
        theta = arccos((trace(R_err) - 1) / 2)
     -input:
         R_est: (3x3) estimated rotation matrix.
         R_gt: (3x3) ground-truth rotation matrix.
     -output:
         theta: rotation error in degrees (0 = perfect).
    """
    R_err = R_est.T @ R_gt
    # Clip: round-off can push the value slightly outside [-1, 1] -> arccos gives NaN
    cos_theta = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def translationDirectionError(t_est, t_gt):
    """
    Angular error between the estimated and ground-truth translation directions.
    From E only the direction of t can be recovered (its norm is arbitrary), so the
    error is measured as the angle between both vectors:
        theta = arccos( (t_est . t_gt) / (||t_est|| ||t_gt||) )
    No abs() is applied: the sign of t is fixed by the cheirality check, so a wrong
    sign must show up as an error close to 180 deg.
     -input:
         t_est: (3,) estimated translation (any norm).
         t_gt: (3,) ground-truth translation (any norm).
     -output:
         theta: direction error in degrees (0 = same direction, 180 = opposite).
    """
    cos_theta = np.dot(t_est, t_gt) / (np.linalg.norm(t_est) * np.linalg.norm(t_gt))
    # Clip: round-off can push the value slightly outside [-1, 1] -> arccos gives NaN
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def pointsError(X_est, X_gt):
    """
    Euclidean error between estimated and ground-truth 3D points.
    Both sets must be in the same frame and scale (e.g. after alignToWorld)
    and in the same order (column i of X_est corresponds to column i of X_gt).
        d_i = || X_est_i - X_gt_i ||
     -input:
         X_est: (3xN) estimated 3D points.
         X_gt: (3xN) ground-truth 3D points.
     -output:
         mean: mean of d_i (world units).
         median: median of d_i (robust to a few bad points).
         rmse: sqrt(mean(d_i^2)) (penalizes large errors more).
    """
    d = np.linalg.norm(X_est[:3, :] - X_gt[:3, :], axis=0)     # (N,) one distance per point
    return np.mean(d), np.median(d), np.sqrt(np.mean(d**2))


def reprojectionError(P, X, x):
    """
    Reprojection error of 3D points in one image (does not need ground truth).
    Each 3D point is projected with P and compared with the measured pixel:
        x_proj = P @ [X; 1]  ->  (u, v) = x_proj[:2] / x_proj[2]
        e_i = || x_i - (u_i, v_i) ||
     -input:
         P: (3x4) projection matrix of the camera (same frame as X).
         X: (3xN) 3D points.
         x: (2xN or 3xN) measured pixel points (only the first two rows are used).
     -output:
         rmse: sqrt(mean(e_i^2)) in pixels.
    """
    X_h = np.vstack((X[:3, :], np.ones((1, X.shape[1]))))
    x_proj = P @ X_h
    x_proj = x_proj[:2, :] / x_proj[2, :]                       # homogeneous -> pixels
    e = np.linalg.norm(x[:2, :] - x_proj, axis=0)               # (N,) one error per point
    return np.sqrt(np.mean(e**2))

#***********************************************************************
#+++Homography
#***********************************************************************

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




#########################################################################


#########################################################################
#--Lab 03--
#########################################################################



#########################################################################



#########################################################################
#--Lab 04--
#########################################################################



#########################################################################


#########################################################################
#--Lab 05--
#########################################################################



#########################################################################

if __name__ == '__main__':
    print("Testing Library")