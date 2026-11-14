#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: RANSAC homography estimation (Exercise 4)
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
from matching import *


#########################################################################
# Plots
#########################################################################
def plotRansacIterations(img1, img2, x1, x2, history, fileName):
    """For some RANSAC iterations: the minimal sample (red) and its inliers (green)."""
    n = len(history)
    fig, axs = plt.subplots(n, 1, figsize=(12, 4.2 * n))
    for ax, h in zip(np.atleast_1d(axs), history):
        kp1, kp2, dm = pointsToKeypoints(x1, x2)
        inl = [dm[i] for i in np.nonzero(h['inliers'])[0]]
        smp = [dm[i] for i in h['sample']]
        img = cv2.drawMatches(img1, kp1, img2, kp2, inl, None, matchColor=(0, 200, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img = cv2.drawMatches(img1, kp1, img2, kp2, smp, img, matchColor=(255, 0, 0),
                              flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG |
                              cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        ax.imshow(img); ax.axis('off')
        ax.set_title(f'Iteration {h["iteration"]}: minimal sample (red, {len(h["sample"])} '
                     f'matches) -> {h["nVotes"]} votes (green)')
    saveFig(fileName)


def plotPlaneTransfer(img1, img2, H_21, x1, x2, inliers, title, fileName):
    """
    Identify the plane: warp image 1 onto image 2 with H_21 and blend. Pixels on the
    plane that induces H overlap perfectly; the rest of the scene appears doubled.
    Also transfers a grid of points spanning the inlier region of image 1.
    """
    h, w = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H_21, (w, h))
    blend = (0.5 * warped + 0.5 * img2).astype(np.uint8)

    p1 = x1[:2, inliers]
    gu, gv = np.meshgrid(np.linspace(p1[0].min(), p1[0].max(), 7),
                         np.linspace(p1[1].min(), p1[1].max(), 5))
    g1 = np.vstack((gu.ravel(), gv.ravel(), np.ones(gu.size)))
    g2 = H_21 @ g1; g2 /= g2[2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
    axs[0].imshow(img1); axs[0].set_title('Image 1: inliers (green), grid points (x)')
    axs[0].plot(x1[0, inliers], x1[1, inliers], 'g.', ms=4)
    axs[0].scatter(g1[0], g1[1], c=np.arange(g1.shape[1]), cmap='jet', marker='x', s=50)
    axs[1].imshow(img2); axs[1].set_title('Image 2: grid transferred with H')
    axs[1].plot(x2[0, inliers], x2[1, inliers], 'g.', ms=4)
    axs[1].scatter(g2[0], g2[1], c=np.arange(g2.shape[1]), cmap='jet', marker='x', s=50)
    axs[1].set_xlim(0, w); axs[1].set_ylim(h, 0)
    axs[2].imshow(blend); axs[2].set_title('Image 1 warped with H blended on image 2')
    for ax in axs:
        ax.axis('off')
    fig.suptitle(title)
    saveFig(fileName)


def runHomography(name, img1, img2, x1, x2, threshold, nStore=0):
    print(f'\n=== Homography RANSAC with {name} ({x1.shape[1]} matches) ===')
    H, inliers, history = ransacHomography(x1, x2, threshold, nStore=nStore)
    print('  H_21 =\n', H)
    rmse = np.sqrt(np.mean(homographyResiduals(H, x1[:, inliers], x2[:, inliers])**2))
    print(f'  Inlier transfer RMSE: {rmse:.3f} px')
    n_c1, d, n_w = planeFromHomography(H, K_C, K_C, T_W_C1, T_W_C2)
    print(f'  Plane normal (camera 1) = {n_c1}, distance = {d:.3f} m')
    print(f'  Plane normal (world)    = {n_w}   (|n_w . z_w| = {abs(n_w[2]):.3f})')

    tag = name.split()[0].lower()
    showMatches(img1, img2, x1, x2, f'{name}: all {x1.shape[1]} matches',
                fileName=f'ex4_{tag}_all_matches.png')
    if history:
        plotRansacIterations(img1, img2, x1, x2, history, f'ex4_{tag}_iterations.png')
    # Most voted homography: inliers (top) and outliers (bottom) in the same figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    showMatches(img1, img2, x1[:, inliers], x2[:, inliers],
                f'{name}: most voted homography, {np.sum(inliers)} inliers (green)',
                color=(0, 200, 0), ax=axs[0])
    showMatches(img1, img2, x1[:, ~inliers], x2[:, ~inliers],
                f'{name}: {np.sum(~inliers)} outliers (red)',
                color=(255, 0, 0), ax=axs[1])
    saveFig(f'ex4_{tag}_inliers_outliers.png')
    plotPlaneTransfer(img1, img2, H, x1, x2, inliers, f'{name}: plane induced by H',
                      f'ex4_{tag}_plane.png')
    return H, inliers


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    img1, img2 = loadImages()

    # SIFT + NNDR matches
    kp1, desc1, kp2, desc2 = extractSIFT(img1, img2)
    m = matchWith2NNDR(desc1, desc2, distRatio=0.8, minDist=500)
    x1, x2 = matchesToPoints(kp1, kp2, m)
    runHomography('SIFT NNDR', img1, img2, x1, x2, threshold=2.0, nStore=3)

    # SuperGlue matches
    x1_sg, x2_sg, _ = loadSuperGlue()
    if x1_sg is not None:
        runHomography('SuperGlue', img1, img2, x1_sg, x2_sg, threshold=2.0, nStore=3)

    plt.show(block=True)   # block: keep windows open (also in the VS Code debugger)
