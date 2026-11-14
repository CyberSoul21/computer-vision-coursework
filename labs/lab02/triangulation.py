#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Point triangulation (Exercise 1)
#
#####################################################################################
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
# Date: 13 November 2026
#          
#####################################################################################

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *
from plotData_v2 import *


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