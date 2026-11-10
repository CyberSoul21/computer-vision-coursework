import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *
from plotData_v2 import *

IMG1 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image1.png"), cv2.COLOR_BGR2RGB)
IMG2 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image2.png"), cv2.COLOR_BGR2RGB)

def plotEpipolarLinesInteractive(img1, img2, F_21, e1=None, e2=None, lineStyle='--',
                                 colors=('r', 'g', 'b', 'c', 'm', 'y')):
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
    T_21 = np.linalg.inv(T_wc1) @ T_wc2

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
         K_c1: (3x3) Camera intrisict parameter for camera 1.
         K_c2: (3x3) Camera intrisict parameter for camera 2.
     -output:
         F: (3x3) Fundamental matrix, dnormalized to unit Frobenius norm.
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

if __name__ == '__main__':
    
    print("Fundamental...")

    F_21_test = np.loadtxt(DATA_LAB + 'F_21_test.txt')
    Kc        = np.loadtxt(DATA_LAB + 'K_c.txt')
    T_w_c1    = np.loadtxt(DATA_LAB + 'T_w_c1.txt')
    T_w_c2    = np.loadtxt(DATA_LAB + 'T_w_c2.txt')
    X1        = np.loadtxt(DATA_LAB + 'x1Data.txt')
    X2        = np.loadtxt(DATA_LAB + 'x2Data.txt')
    x1 = np.vstack((X1,np.ones((1,X1.shape[1]))))
    x2 = np.vstack((X2,np.ones((1,X2.shape[1]))))

    # # Display image 1
    # plt.figure(1)
    # plt.imshow(IMG1)
    # plt.title("Click a point in image 1")
    # # Wait for one mouse click
    # points = plt.ginput(1, timeout=0)
    # #plt.close()
    # # Extract coordinates
    # u1, v1 = points[0]
    # # Convert to homogeneous coordinates
    # x1 = np.array([u1, v1, 1.0])
    # l2 = F_21_test @ x1
    # # Display image 2
    # plt.figure(2)
    # plt.imshow(IMG2)
    # plt.title("Epipolar line in Image 2")
    # drawLine(l2)

    #2.1 Epipolar lines visualization
    #plotEpipolarLinesInteractive(IMG1,IMG2,F_21_test)

    #2.2 Fundamental matrix definition, from ground truth poses 
    
    E_21 = essentialFromGroundTruth(T_w_c1,T_w_c2)
    F_21 = fundamentalMatrixFromEssential(E_21,Kc,Kc)

    print("F_21_test:")
    F_21_test /= np.linalg.norm(F_21_test) 
    print(F_21_test)
    print("F_21:")
    print(F_21)

    #2.3 Fundamental matrix linear estimation with eight point solution.


    #E_21 = essentialFromEightPoints(x1, x2, Kc, Kc)
    #print("E_21 (eight points):")
    #print(E_21)


    F_21 = fundamentalFromEightPoints(x1, x2)
    print("F_21 (eight points):")
    print(F_21)
    
    e1, e2 = computeEpipoles(F_21)
    print("Epipole image 1:", e1)
    print("Epipole image 2:", e2)

    plotEpipolarLinesInteractive(IMG1,IMG2,F_21,e2=e2)

    E_21 = essentialFromFundamental(F_21,Kc,Kc)
    print("E_21 from F_21:")
    print(E_21)


    
