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



#########################################################################

if __name__ == '__main__':
    print("Testing Library")