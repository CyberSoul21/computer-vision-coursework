import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *
from plotData_v2 import *


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


if __name__ == '__main__':

    print("Triangulation...")

    # Load Groud truth
    Kc     = np.loadtxt(DATA_LAB_1 + 'K_c.txt')
    T_w_c1 = np.loadtxt(DATA_LAB_1 + 'T_w_c1.txt')
    T_w_c2 = np.loadtxt(DATA_LAB_1 + 'T_w_c2.txt')
    #We need world relative to the camera
    T_c1_w = invert_transform(T_w_c1) 
    T_c2_w = invert_transform(T_w_c2)

    #Load Data:
    x1 = np.loadtxt(DATA_LAB_1 + 'x1Data.txt')
    x2 = np.loadtxt(DATA_LAB_1 + 'x2Data.txt')

    print("Compute Proyection Matrixes...")
    Pc1 = computeProjectionMatrix(Kc,T_c1_w)
    Pc2 = computeProjectionMatrix(Kc,T_c2_w)

    print("Triangulate points 3D...")
    X = triangulatePoints(Pc1, Pc2, x1, x2)
    print(X.shape)

    fig3D = plt.figure(3)
    ax = plt.axes(projection='3d', adjustable='box')    
    cameras = {'W': np.eye(4), 'C1': T_w_c1, 'C2': T_w_c2}

    plot3DScene(X,cameras,ax=ax,show=True)
    #plt.show(block=True)