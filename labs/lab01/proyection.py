#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 1
#
# Title: 2D-3D geometry in homogeneous coordinates and camera projection
#
# Date: 5 September 2026
#
#####################################################################################
#
# Authors: Wilson Javier Almario Rodriguez.
#
# Version: 2.0
#
#####################################################################################

import matplotlib.pyplot as plt
import scipy.linalg as scAlg
import numpy as np
import cv2

from plotData_v15 import *

#DATA_LAB_1 = 'data/'

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

if __name__ == '__main__':

    # Load ground truth
    # Orientation of cameras relative to the world
    R_w_c1 = np.loadtxt(DATA_LAB_1 + 'R_w_c1.txt')
    R_w_c2 = np.loadtxt(DATA_LAB_1 + 'R_w_c2.txt')
    # Traslation vector of camera 1 relative to the world coordinate system
    t_w_c1 = np.loadtxt(DATA_LAB_1 + 't_w_c1.txt')
    t_w_c2 = np.loadtxt(DATA_LAB_1 + 't_w_c2.txt')
    K      = np.loadtxt(DATA_LAB_1 + 'K.txt')

    X_A =  np.array([3.44, 0.80, 0.82, 1])
    X_B =  np.array([4.20, 0.80, 0.82, 1])
    X_C =  np.array([4.20, 0.60, 0.82, 1])
    X_D =  np.array([3.55, 0.60, 0.82, 1])
    X_E =  np.array([-0.01, 2.6, 1.21, 1])

    X = np.array([X_A,X_B,X_C,X_D,X_E]).transpose()
    print(X)

    labels = ['a','b','c','d','e']


    T_w_c1 = assemble_transform(R_w_c1,t_w_c1)
    T_w_c2 = assemble_transform(R_w_c2,t_w_c2)
    #We need world relative to the camera
    T_c1_w = invert_transform(T_w_c1) 
    T_c2_w = invert_transform(T_w_c2) 

    #print(computeProjectionFromKRT(K,R_w_c1,t_w_c1))
    P1 = computeProjectionMatrix(K,T_c1_w)
    P2 = computeProjectionMatrix(K,T_c2_w)
    
    #**************************************************************************************
    # 1.2 Compute and plot all the projections of the 3D points A,B,C,D,E in both images.
    x1 = projectPoints(P1,X)
    x2 = projectPoints(P2,X)
    #**************************************************************************************
    #########################################################################

    print("*************************")
    print("x1: ")
    print(x1)
    print("x2: ")
    print(x2)

    print("*************************")
    #Homogenized Points
    X1 = np.vstack([x1,np.ones(x1.shape[1])])
    X2 = np.vstack([x2,np.ones(x2.shape[1])])
    
    #########################################################################
    #2D Lines and vanishing points
    ########################################################################
    #**********************************************************************
    #2.1 Compute and plot in each image the line l_ab defined by a, and b (projections on A and B).

    #get each point
    A_2d_im1 = X1[:,0:1]
    B_2d_im1 = X1[:,1:2]
    C_2d_im1 = X1[:,2:3]
    D_2d_im1 = X1[:,3:4]
    E_2d_im1 = X1[:,4:]

    A_2d_im2 = X2[:,0:1]
    B_2d_im2 = X2[:,1:2]
    C_2d_im2 = X2[:,2:3]
    D_2d_im2 = X2[:,3:4]
    E_2d_im2 = X2[:,4:]

    print("A_2d_im1: ")
    print(A_2d_im1)

    print("B_2d_im1: ")
    print(B_2d_im1)

    #print(lineFromPoints(A_2d_im1,B_2d_im1)) #(1,3)


    #line ab on image 1
    l_ab_im1 = lineFromPoints(A_2d_im1,B_2d_im1)
    #line cd on image 1
    l_cd_im1 = lineFromPoints(C_2d_im1,D_2d_im1)

    #line ab on image 2
    l_ab_im2 = lineFromPoints(A_2d_im2,B_2d_im2)
    #line cd on image 2
    l_cd_im2 = lineFromPoints(C_2d_im2,D_2d_im2)


    #**********************************************************************
    #2.3
    #Compute p_12 the intersection point of l_ab and l_cd

    #Image 1
    P_12_im1 = np.cross(l_ab_im1,l_cd_im1)
    P_12_im1 = P_12_im1/P_12_im1[2]
    P_12_im1 = np.array([P_12_im1[:2]]).transpose() 
    print("P_12_im1: ", P_12_im1)

    #Image 2
    P_12_im2 = np.cross(l_ab_im2,l_cd_im2)
    P_12_im2 = P_12_im2/P_12_im2[2]
    P_12_im2 = np.array([P_12_im2[:2]]).transpose() 
    print("P_12_im2: ", P_12_im2)
    #**********************************************************************


    #**********************************************************************
    #2.4
    #Compute the 3D infinite point corresponding to the 3D direction defined by points A and B, AB_inf.
    #Get the direction of AB
    D_AB = X_B - X_A
    print("D_AB: ",D_AB)
    for i, val in enumerate(D_AB):
        if val > 0:
            D_AB[i] = 1
        elif val < 0:
            D_AB[i] = -1
    AB_inf = np.array([D_AB]).transpose()
    print("AB_inf: ",AB_inf)
    #**********************************************************************

    #**********************************************************************
    #2.5
    #Project the point AB_inf with matrix P to obtain the corresponding vanishing point ab_inf.
    #Image 1
    AB_2d_im1 = projectPoints(P1,AB_inf)
    print("AB_2d_im1: ",AB_2d_im1)
    #Image 2
    AB_2d_im2 = projectPoints(P1,AB_inf)
    print("AB_2d_im2: ",AB_2d_im2)    


    # ---- Plot on image 1 ----
    plt.figure(figsize=(9,5))
    plt.imshow(IMG1)
    #plot points
    plt.plot(x1[0,:], x1[1,:], '+r', markersize=12)
    plotLabeledImagePoints(x1, labels, 'r', (18, -18))   # offset (dx,dy) in pixels
    #plot lines
    drawLine(l_ab_im1, 'r-', 2) 
    drawLine(l_cd_im1, 'g-', 2)
    #plot intersection l_ab amd l_cd
    plt.plot(P_12_im1[0,:], P_12_im1[1,:], '+b', markersize=12)
    plt.text(P_12_im1[0,:] + 5, P_12_im1[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
    #plot infinite point
    plt.plot(AB_2d_im1[0,:], AB_2d_im1[1,:], '*c', markersize=12)
    plt.text(P_12_im1[0,:] + 5, P_12_im1[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
    # or: plotNumberedImagePoints(x1, 'r', (18, 24))
    plt.title('Image 1: projections of A..E')
    #plt.axis('off')
    plt.tight_layout()

    # ---- Plot on image 2 ----
    plt.figure(figsize=(9,5))
    plt.imshow(IMG2)
    #plot points
    plt.plot(x2[0,:], x2[1,:], '+c', markersize=12)
    plotLabeledImagePoints(x2, labels, 'c', (18, -18))
    #plot lines
    drawLine(l_ab_im2, 'r-', 2) 
    drawLine(l_cd_im2, 'g-', 2)
    #plot intersection l_ab amd l_cd
    plt.plot(P_12_im2[0,:], P_12_im2[1,:], '+b', markersize=12)
    plt.text(P_12_im2[0,:] + 5, P_12_im2[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
    #plot infinite point
    plt.plot(AB_2d_im2[0,:], AB_2d_im2[1,:], '*c', markersize=12)
    plt.text(P_12_im2[0,:] + 5, P_12_im2[1,:] - 5, 'P12', fontsize=10, color='blue')  # Add label near point
    plt.title('Image 2: projections of A..E')
    #plt.axis('off')
    plt.tight_layout()
    plt.show()
    #**********************************************************************    