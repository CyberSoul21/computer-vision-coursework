import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvLibrary import *
from plotData_v2 import *

IMG1 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image1.png"), cv2.COLOR_BGR2RGB)
IMG2 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image2.png"), cv2.COLOR_BGR2RGB)


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
    plotEpipolarLinesInteractive(IMG1,IMG2,F_21_test,title="Fundamental test")

    #2.2 Fundamental matrix definition, from ground truth poses 
    
    E_21 = essentialFromGroundTruth(T_w_c1,T_w_c2)
    F_21 = fundamentalMatrixFromEssential(E_21,Kc,Kc)

    print("F_21_test:")
    F_21_test /= np.linalg.norm(F_21_test) 
    print(F_21_test)
    print("F_21:")
    print(F_21)

    plotEpipolarLinesInteractive(IMG1,IMG2,F_21,title="Fundamental from ground truth")


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

    plotEpipolarLinesInteractive(IMG1,IMG2,F_21,e2=e2,title="Fundamental from 8 points")

    #2.4 Pose estimation from two views
    E_21 = essentialFromFundamental(F_21,Kc,Kc)
    print("E_21 from F_21:")
    print(E_21)

    T_21, X = recoverPoseFromEssential(E_21, Kc, Kc, x1, x2)
    print("T_21:\n", T_21)

    #2.5 Results presentation
    X_w = np.loadtxt(DATA_LAB + 'X_w.txt')
    if X_w.shape[0] == 4:                              # homogeneous -> Euclidean
        X_w = X_w[:3, :] / X_w[3, :]

    # True baseline (the only thing E cannot give us)
    T_21_gt = np.linalg.inv(T_w_c2) @ T_w_c1
    scale = np.linalg.norm(T_21_gt[:3, 3])

    T_w_c2_est, X_w_est = alignToWorld(T_21, X, T_w_c1, scale)

    # Metrics
    rot_err = rotationError(T_21[:3, :3], T_21_gt[:3, :3])
    print(f"Rotation error: {rot_err:.4f} deg")
    t_err = translationDirectionError(T_21[:3, 3], T_21_gt[:3, 3])
    print(f"Translation direction error: {t_err:.4f} deg")

    mean_3d, median_3d, rmse_3d = pointsError(X_w_est, X_w)
    print(f"3D points error: mean {mean_3d:.4f}, median {median_3d:.4f}, RMSE {rmse_3d:.4f} (world units)")

    # Reprojection: estimated cameras + estimated points (camera 1 frame, scale-free)
    P1_est = Kc @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_est = computeProjectionMatrix(Kc, T_21)
    rep1 = reprojectionError(P1_est, X, x1)
    rep2 = reprojectionError(P2_est, X, x2)
    print(f"Reprojection error: image 1 {rep1:.4f} px, image 2 {rep2:.4f} px")

    # Ground truth (blue, solid) vs estimation (red, dashed camera)
    cameras = {'W': np.eye(4), 'C1': T_w_c1, 'C2': T_w_c2}
    ax = plot3DScene(X_w, cameras, strColor='b', show=False)
    plot3DPoints(ax, X_w_est, strColor='r')
    drawRefSystem(ax, T_w_c2_est, '--', 'C2_est')
    ax.set_title('Ground truth (blue) vs estimation (red)')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show(block=True)

























    # #Recover the four possible camera transformations

    # U, S , Vt = np.linalg.svd(E_21)

    # #Ensure valid rotation matrices

    # # Ensure valid rotation matrices
    # if np.linalg.det(U @ Vt) < 0:
    #     U[:, -1] *= -1

    # W = np.array([
    #     [0, -1, 0],
    #     [1,  0, 0],
    #     [0,  0, 1]
    # ])

    # R1 = U @ W @ Vt
    # R2 = U @ W.T @ Vt

    # R_list = [R1,R2]
    
    # t = U[:, -1]

    # t_list = [t,-t]

    # T_list = []
    # for R in R_list:
    #     for t in t_list:
    #         T = np.eye(4)
    #         T[:3, :3] = R
    #         T[:3, 3] = t
    #         T_list.append(T)

    # #Triangulate points using each candidate
    # #For relative pose estimation, we can choose camera 1 as our reference coordinate system:

    # P1 = Kc @ np.hstack([
    #     np.eye(3),
    #     np.zeros((3, 1))
    # ])

    # #Triangulate points using each candidate

    # total_score = []
    # for T in T_list:
    #     P2 = computeProjectionMatrix(Kc,T)

    #     #Select the correct transformation (Cheirality)
    #     #The key geometric constraint is:
    #     #A physically visible 3D point must be in front of both cameras.
    #     #In the camera coordinate system, this means that its depth Z must be positive.
    #     X = triangulatePoints(P1,P2,x1,x2)


    #     X_c2 = T[:3, :3] @ X + T[:3, 3:4]           # same points in camera 2 frame

    #     valid = (X[2, :] > 0) & (X[2, :] > 0)
    #     score = np.sum(valid)

    #     total_score.append(score)

    # print(total_score)
    # best = np.argmax(total_score)
    # T_21 = T_list[best]
    


