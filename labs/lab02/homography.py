#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Homographies (Exercise 3)
#        H from ground truth, point transfer and linear estimation from matches
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

IMG1 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image1.png"), cv2.COLOR_BGR2RGB)
IMG2 = cv2.cvtColor(cv2.imread(DATA_LAB + "Image2.png"), cv2.COLOR_BGR2RGB)

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
    H_est = homographyFromMatches(x1FloorData, x2FloorData)
    print("H_21 estimated from matches:")
    print(H_est)

    # Evaluate the accuracy: estimated vs ground-truth homography on the same matches
    rmse_est = homographyTransferError(H_est, x1FloorData, x2FloorData)
    rmse_gt  = homographyTransferError(H_21,  x1FloorData, x2FloorData)
    print(f"Transfer RMSE: estimated H {rmse_est:.4f} px, ground-truth H {rmse_gt:.4f} px")
    plotTransferredPointsHomography(IMG1,IMG2,H_est)