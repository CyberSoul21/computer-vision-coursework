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

    Pc1_1 = Pc1[0,:]
    Pc1_2 = Pc1[1,:]
    Pc1_3 = Pc1[2,:]
 
    Pc2_1 = Pc2[0,:]
    Pc2_2 = Pc2[1,:]
    Pc2_3 = Pc2[2,:]

    u1 = x1[0,:]
    v1 = x1[1,:] 

    # u1P13 = u1[0] * Pc1_3.T
    # v1P13 = v1[0] * Pc1_3.T

    u1P13 = u1[:, None] * Pc1_3[None, :]
    v1P13 = v1[:, None] * Pc1_3[None, :]    

    Ac1 = np.stack([u1P13 - Pc1_1,v1P13 - Pc1_2], axis=1)
    #Ac1 = np.vstack((u1P13 - Pc1_1.T , v1P13 - Pc1_2.T))
    print(Ac1)
    u2 = x2[0,:]
    v2 = x2[1,:] 

    u2P23 = u2[:, None] * Pc2_3[None, :]
    v2P23 = v2[:, None] * Pc2_3[None, :]  
    
    Ac2 = np.stack([u2P23 - Pc2_1,v2P23 - Pc2_2],axis=1)
    print(Ac2)
    #print(u1P13.reshape(4,1))
    
    A = np.concatenate([Ac1,Ac2],axis=1)

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    X_h = vh[:,-1, :].T 
    #Convert homogeneous coordinates to Euclidean coordinates
    X = X_h[0:3,:]
    W = X_h[3,:]
    X = X/W
    print("Triangulate points 3D...")
    print(X.shape)

    fig3D = plt.figure(3)
    ax = plt.axes(projection='3d', adjustable='box')    
    cameras = {'W': np.eye(4), 'C1': T_w_c1, 'C2': T_w_c2}

    plot3DScene(X,cameras,ax=ax,show=True)
    #plt.show(block=True)