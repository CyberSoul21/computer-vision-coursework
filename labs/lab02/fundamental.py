import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *
from plotData_v2 import *

IMG1 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image1.png"), cv2.COLOR_BGR2RGB)
IMG2 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image2.png"), cv2.COLOR_BGR2RGB)

def plotEpipolarLinesInteractive(img1, img2, F_21, colors=('r', 'g', 'b', 'c', 'm', 'y')):
    """
    Click points in image 1 and draw their epipolar lines in image 2.
    Press Enter (or right-click) to finish.
     -input:
         img1, img2: images.
         F_21: (3x3) fundamental matrix mapping points in image 1 to lines in image 2 (l2 = F_21 @ x1).
         colors: colors cycled for each point/line pair.
     -output:
         x1: (3xN) clicked points in homogeneous coordinates.
         l2: (3xN) corresponding epipolar lines in image 2.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(img1); ax1.set_title("Click points in image 1 (Enter to finish)")
    ax2.imshow(img2); ax2.set_title("Epipolar lines in image 2")

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
        drawLine(l2,c)
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

if __name__ == '__main__':
    
    print("Fundamental...")

    F_21_test = np.loadtxt(DATA_LAB + 'F_21_test.txt')
    Kc        = np.loadtxt(DATA_LAB + 'K_c.txt')
    T_w_c1    = np.loadtxt(DATA_LAB + 'T_w_c1.txt')
    T_w_c2    = np.loadtxt(DATA_LAB + 'T_w_c2.txt')

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
    



    