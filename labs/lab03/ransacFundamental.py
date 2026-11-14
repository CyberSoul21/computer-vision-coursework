#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: RANSAC fundamental matrix estimation (Exercise 5)
#        and guided matching with epipolar geometry (Exercise 6, optional)
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
from ransacHomography import *


#########################################################################
# Plots
#########################################################################
def drawEpipolarLine(ax, l, w, color):
    """Draw the line l = (a, b, c) across the full image width (a u + b v + c = 0)."""
    u = np.array([-2000.0, w + 2000.0])
    v = -(l[0] * u + l[2]) / l[1]
    ax.plot(u, v, '-', color=color, linewidth=1)


def plotEpipolarLines(img1, img2, F_21, x1, x2, e1, e2, e1_gt, e2_gt, title, fileName,
                      nLines=12, seed=1):
    """
    Epipolar lines of a random subset of inlier matches in both images.
    All lines must cross at the epipole; the ground-truth epipole is shown for comparison.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(x1.shape[1], size=min(nLines, x1.shape[1]), replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(idx)))
    h, w = img1.shape[:2]

    fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
    axs[0].imshow(img1); axs[1].imshow(img2)
    for c, i in zip(colors, idx):
        drawEpipolarLine(axs[0], F_21.T @ x2[:, i], w, c)   # line in image 1
        axs[0].plot(x1[0, i], x1[1, i], 'o', color=c, mec='k')
        drawEpipolarLine(axs[1], F_21 @ x1[:, i], w, c)     # line in image 2
        axs[1].plot(x2[0, i], x2[1, i], 'o', color=c, mec='k')
    for ax, e, egt, name in ((axs[0], e1, e1_gt, 'e1'), (axs[1], e2, e2_gt, 'e2')):
        ax.plot(e[0], e[1], 'r+', ms=18, mew=3, label=f'{name} estimated')
        ax.plot(egt[0], egt[1], 'bx', ms=14, mew=3, label=f'{name} ground truth')
        # Keep the epipole visible even if it falls outside the image
        ax.set_xlim(min(0, e[0], egt[0]) - 20, max(w, e[0], egt[0]) + 20)
        ax.set_ylim(max(h, e[1], egt[1]) + 20, min(0, e[1], egt[1]) - 20)
        ax.legend(loc='lower right')
    axs[0].set_title('Image 1: epipolar lines F^T x2')
    axs[1].set_title('Image 2: epipolar lines F x1')
    fig.suptitle(title)
    saveFig(fileName)


def runFundamental(name, img1, img2, x1, x2, threshold, nStore=0):
    print(f'\n=== Fundamental RANSAC with {name} ({x1.shape[1]} matches) ===')
    F, inliers, history = ransacFundamental(x1, x2, threshold, nStore=nStore)
    print('  F_21 =\n', F)

    F_gt = fundamentalFromGroundTruth(T_W_C1, T_W_C2, K_C, K_C)
    e1_gt, e2_gt = epipolesFromGroundTruth(T_W_C1, T_W_C2, K_C, K_C)
    e1, e2 = computeEpipoles(F)
    print(f'  e1 estimated = {e1[:2]}, ground truth = {e1_gt[:2]}, '
          f'error = {np.linalg.norm(e1[:2] - e1_gt[:2]):.1f} px')
    print(f'  e2 estimated = {e2[:2]}, ground truth = {e2_gt[:2]}, '
          f'error = {np.linalg.norm(e2[:2] - e2_gt[:2]):.1f} px')
    r_est = fundamentalResiduals(F, x1[:, inliers], x2[:, inliers])
    r_gt = fundamentalResiduals(F_gt, x1[:, inliers], x2[:, inliers])
    print(f'  Inliers epipolar distance: estimated F median {np.median(r_est):.3f} px, '
          f'ground-truth F median {np.median(r_gt):.3f} px')

    tag = name.split()[0].lower()
    if history:
        plotRansacIterations(img1, img2, x1, x2, history, f'ex5_{tag}_iterations.png')
    showMatches(img1, img2, x1[:, inliers], x2[:, inliers],
                f'{name}: most voted F, {np.sum(inliers)} inliers',
                fileName=f'ex5_{tag}_inliers.png', color=(0, 200, 0))
    plotEpipolarLines(img1, img2, F, x1[:, inliers], x2[:, inliers], e1, e2, e1_gt, e2_gt,
                      f'{name}: epipolar lines of inliers (estimated F)',
                      f'ex5_{tag}_epipolar.png')
    return F, inliers


#########################################################################
# Exercise 6: guided matching
#########################################################################
def guidedMatching(F_21, kp1, desc1, kp2, desc2, epiThreshold=2.0, distRatio=0.8,
                   minDist=500):
    """
    Matching restricted by the epipolar constraint.
    For each feature of image 1, the candidates in image 2 are only the features at
    less than epiThreshold pixels of its epipolar line F_21 x1 (and symmetric). The
    NNDR test is applied among those candidates only. Since the repeated structures
    elsewhere in the image are no longer candidates, d2 grows and many correct
    matches that were rejected by the global NNDR (false negatives) are recovered.
     -output:
         matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    p1 = np.vstack((np.array([k.pt for k in kp1]).T, np.ones(len(kp1))))
    p2 = np.vstack((np.array([k.pt for k in kp2]).T, np.ones(len(kp2))))
    D = descriptorDistances(desc1, desc2)

    # Epipolar distance between every pair (i in image 1, j in image 2)
    l2 = F_21 @ p1                                  # lines in image 2 (3 x n1)
    l1 = F_21.T @ p2                                # lines in image 1 (3 x n2)
    d_in2 = np.abs(l2.T @ p2) / np.sqrt(l2[0]**2 + l2[1]**2)[:, None]     # n1 x n2
    d_in1 = np.abs(p1.T @ l1) / np.sqrt(l1[0]**2 + l1[1]**2)[None, :]     # n1 x n2
    candidates = 0.5 * (d_in1 + d_in2) < epiThreshold

    Dc = np.where(candidates, D, np.inf)
    idx = np.argsort(Dc, axis=1)[:, :2]
    rows = np.arange(D.shape[0])
    dNN, d2NN = Dc[rows, idx[:, 0]], Dc[rows, idx[:, 1]]
    # Only one candidate on the line: d2NN = inf -> ratio test trivially passed
    keep = np.isfinite(dNN) & (dNN < minDist) & (dNN < distRatio * d2NN)
    i1 = np.nonzero(keep)[0]
    return mutualFilter(np.column_stack((i1, idx[keep, 0], dNN[keep])))


def runGuidedMatching(img1, img2, kp1, desc1, kp2, desc2, F, matchesNNDR):
    print('\n=== Exercise 6: guided matching ===')
    mg = guidedMatching(F, kp1, desc1, kp2, desc2, epiThreshold=2.0, distRatio=0.8)
    oldPairs = {(int(a), int(b)) for a, b, _ in matchesNNDR}
    isNew = np.array([(int(a), int(b)) not in oldPairs for a, b, _ in mg])
    print(f'  Guided matches: {len(mg)} (NNDR had {len(matchesNNDR)}), new: {np.sum(isNew)}')

    x1g, x2g = matchesToPoints(kp1, kp2, mg)

    showMatches(img1, img2, x1g[:, isNew], x2g[:, isNew],
                f'Guided matching: {np.sum(isNew)} new matches (not found by global NNDR)',
                fileName='ex6_guided_new_matches.png', color=(255, 140, 0))
    return mg


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    img1, img2 = loadImages()

    print('Ground truth F_21 =\n', fundamentalFromGroundTruth(T_W_C1, T_W_C2, K_C, K_C))

    # SIFT + NNDR matches
    kp1, desc1, kp2, desc2 = extractSIFT(img1, img2)
    m = matchWith2NNDR(desc1, desc2, distRatio=0.8, minDist=500)
    x1, x2 = matchesToPoints(kp1, kp2, m)
    F, inliers = runFundamental('SIFT NNDR', img1, img2, x1, x2, threshold=2.0, nStore=3)

    # SuperGlue matches
    x1_sg, x2_sg, _ = loadSuperGlue()
    if x1_sg is not None:
        runFundamental('SuperGlue', img1, img2, x1_sg, x2_sg, threshold=2.0, nStore=3)

    # Exercise 6 (optional): guided matching with the F estimated from SIFT
    #runGuidedMatching(img1, img2, kp1, desc1, kp2, desc2, F, m)

    plt.show(block=True)   # block: keep windows open (also in the VS Code debugger)
