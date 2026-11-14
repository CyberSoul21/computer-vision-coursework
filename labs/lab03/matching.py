#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: Feature matching (Exercises 1, 2 and 3)
#        1. Nearest Neighbour (NN) matching with a maximum descriptor distance.
#        2. Nearest Neighbour Distance Ratio (NNDR) matching.
#        3. Loading SuperGlue matches (.npz produced by match_pairs.py).
#        Helpers shared by ransacHomography.py and ransacFundamental.py.
#
#####################################################################################
#####################################################################################
#
# Authors: Wilson Javier Almario, 962449
# Date: 13 November 2026
#          
#####################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
SUPERGLUE_NPZ = os.path.join(OUTPUT_DIR, 'image1_image2_matches.npz')   # SuperGlue output

# Calibration and ground-truth poses (from the lab statement)
K_C = np.array([[458.654, 0.0,     367.215],
                [0.0,     457.296, 248.375],
                [0.0,     0.0,     1.0]])

T_W_C1 = np.array([[-0.0744, -0.3079,  0.9485, 0.8801],
                   [-0.997,   0.0428, -0.0643, 0.8695],
                   [-0.0208, -0.9505, -0.3102, 1.73],
                   [0.0,      0.0,     0.0,    1.0]])

T_W_C2 = np.array([[-0.512,  -0.2779,  0.8128, 0.5236],
                   [-0.8586,  0.194,  -0.4745, 1.9537],
                   [-0.0258, -0.9408, -0.338,  1.2876],
                   [0.0,      0.0,     0.0,    1.0]])


#########################################################################
# Image / feature I/O
#########################################################################
def loadImages():
    """Return the two lab images in RGB (for matplotlib)."""
    img1 = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, 'image1.png')), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, 'image2.png')), cv2.COLOR_BGR2RGB)
    return img1, img2


def extractSIFT(img1, img2):
    """SIFT keypoints and descriptors with the parameters given in SIFTmatching.py."""
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.02,
                           edgeThreshold=20, sigma=0.5)
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    return kp1, desc1, kp2, desc2


def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    return [cv2.DMatch(_queryIdx=int(r[0]), _trainIdx=int(r[1]), _distance=float(r[2]))
            for r in matchesList]


def matchesToPoints(kp1, kp2, matchesList):
    """
    Pixel coordinates of the matched keypoints.
     -output:
         x1, x2: (3xN) homogeneous points in image 1 and image 2.
    """
    matchesList = np.asarray(matchesList)
    if len(matchesList) == 0:
        return np.zeros((3, 0)), np.zeros((3, 0))
    #p1 = np.array([kp1[int(i)].pt for i in matchesList[:, 0]]).T
    points = []
    for i in matchesList[:, 0]:
        points.append(kp1[int(i)].pt)

    p1 = np.array(points).T

    #p2 = np.array([kp2[int(j)].pt for j in matchesList[:, 1]]).T
    points = []
    for i in matchesList[:, 1]:
        points.append(kp2[int(i)].pt)

    p2 = np.array(points).T

    ones = np.ones((1, p1.shape[1]))
    return np.vstack((p1, ones)), np.vstack((p2, ones))


def loadSuperGlue(path=SUPERGLUE_NPZ):
    """
    Load the SuperGlue output (Appendix A.3).
     -output:
         x1, x2: (3xN) homogeneous matched points, or (None, None) if the file is missing.
         npz: the loaded dictionary (keypoints0/1, matches, match_confidence, ...).
    """
    if not os.path.exists(path):
        print(f'SuperGlue file not found: {path}  (run match_pairs.py, see README)')
        return None, None, None
    npz = np.load(path)
    mask = npz['matches'] > -1
    idxs = npz['matches'][mask]
    p1 = npz['keypoints0'][mask].T
    p2 = npz['keypoints1'][idxs].T
    ones = np.ones((1, p1.shape[1]))
    return np.vstack((p1, ones)), np.vstack((p2, ones)), npz


def pointsToKeypoints(x1, x2):
    """
    Build keypoints + 1:1 DMatch list from matched points so cv2.drawMatches
    can display any (3xN) point set (SIFT inliers, SuperGlue, ...).
    """
    kp1 = [cv2.KeyPoint(float(u), float(v), 5) for u, v in x1[:2].T]
    kp2 = [cv2.KeyPoint(float(u), float(v), 5) for u, v in x2[:2].T]
    dm = [cv2.DMatch(i, i, 0.0) for i in range(len(kp1))]
    return kp1, kp2, dm


def showMatches(img1, img2, x1, x2, title, fileName=None, color=None, ax=None):
    """Display matches given as (3xN) homogeneous points with cv2.drawMatches."""
    kp1, kp2, dm = pointsToKeypoints(x1, x2)
    kw = {} if color is None else {'matchColor': color, 'singlePointColor': color}
    imgMatched = cv2.drawMatches(img1, kp1, img2, kp2, dm, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, **kw)
    if ax is None:
        plt.figure(figsize=(14, 5))
        ax = plt.gca()
    ax.imshow(imgMatched)
    ax.set_title(title)
    ax.axis('off')
    if fileName is not None:
        saveFig(fileName)
    return ax


def saveFig(fileName):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fileName), dpi=110, bbox_inches='tight')


#########################################################################
# Exercise 1 and 2: NN and NNDR matching
#########################################################################
def descriptorDistances(desc1, desc2):
    """
    L2 distance between every pair of descriptors.
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b   (vectorized, no python loop)
     -output:
         D: (nDesc1 x nDesc2) matrix with D[i, j] = ||desc1[i] - desc2[j]||
    """
    d1 = np.sum(desc1.astype(np.float64)**2, axis=1)[:, None]
    d2 = np.sum(desc2.astype(np.float64)**2, axis=1)[None, :]
    D2 = d1 + d2 - 2.0 * desc1.astype(np.float64) @ desc2.astype(np.float64).T
    return np.sqrt(np.maximum(D2, 0.0))   # clip round-off negatives


def matchWithNN(desc1, desc2, minDist):
    """
    Exercise 1: plain Nearest Neighbour matching (the provided code).
    A match is accepted if the distance to the nearest neighbour is < minDist.
     -output:
        matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    D = descriptorDistances(desc1, desc2)
    #D is the distance matrix computed on the line before. 
    #it has one row per feature of image 1 (2310) and 
    # one column per feature of image 2 (6520), 
    # and D[i, j] is the distance between feature i of image 1 and feature j of image 2.
    nn = np.argmin(D, axis=1) # scans each row and returns the column index of the smallest value,
    d1 = D[np.arange(D.shape[0]), nn]
    keep = d1 < minDist
    idx1 = np.nonzero(keep)[0]
    return np.column_stack((idx1, nn[keep], d1[keep]))


def matchWith2NNDR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    For every descriptor of image 1 the two closest descriptors of image 2 are found
    (distances d1 <= d2). The match (nearest neighbour) is accepted if
        d1 < distRatio * d2     (the NN is clearly better than the 2nd NN)   and
        d1 < minDist            (the NN is close enough in absolute terms).
    Repeated texture (chessboard aliasing) gives d1 ~ d2, so those matches are rejected.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio: maximum ratio d1/d2 (typ. 0.6 - 0.8)
        minDist: maximum descriptor distance d1
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """


    """
    for i in range(n1):  # each feature of image 1
    dists = distances from desc1[i] to all desc2
    j1 = index of the smallest, d1 = its distance
    d2 = second smallest distance
    if d1 < distRatio * d2 and d1 < minDist:
        accept match (i, j1, d1)
    """

    """
    matches = []
    for i in range(D.shape[0]):            # each feature of image 1
        row = D[i]                         # its distances to all features of image 2
        j_sorted = np.argsort(row)         # indices from closest to farthest
        j1, j2 = j_sorted[0], j_sorted[1]  # closest and second closest
        d1, d2 = row[j1], row[j2]
        if d1 < distRatio * d2 and d1 < minDist:
            matches.append([i, j1, d1])
    return np.array(matches)
    """

    #Which two features of image 2 are closest?
    D = descriptorDistances(desc1, desc2)
    # Two smallest distances per row (argpartition: O(n), no full sort needed)
    idx2 = np.argpartition(D, 1, axis=1)[:, :2] # each row, argpartition finds the positions (column numbers) of the two smallest values.
    #How far away are they?
    rows = np.arange(D.shape[0])[:, None]
    d12 = D[rows, idx2] # look up those two distances
    # put the smaller one first:
    order = np.argsort(d12, axis=1)              # make column 0 the nearest
    idx2 = np.take_along_axis(idx2, order, axis=1)
    d12 = np.take_along_axis(d12, order, axis=1)
    dNN, d2NN = d12[:, 0], d12[:, 1] # d1 = closest, d2 = second closest

    #Is the closest one clearly the best, and close enough?
    keep = (dNN < distRatio * d2NN) & (dNN < minDist)
    idx1 = np.nonzero(keep)[0]
    return np.column_stack((idx1, idx2[keep, 0], dNN[keep]))


def mutualFilter(matches):
    """Keep at most one match per feature of image 2 (the one with lowest distance)."""
    if len(matches) == 0:
        return matches
    matches = matches[np.argsort(matches[:, 2])]
    _, first = np.unique(matches[:, 1], return_index=True)
    return matches[np.sort(first)]


#########################################################################
# Main: exercises 1, 2, 3
#########################################################################
if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    img1, img2 = loadImages()
    kp1, desc1, kp2, desc2 = extractSIFT(img1, img2)
    print(f'SIFT features: image 1 = {len(kp1)}, image 2 = {len(kp2)}')

    # ---------------- Exercise 1: NN with different thresholds ----------------
    print('\n--- Exercise 1: NN matching, threshold sweep ---')
    thresholds = [100, 200, 300, 500]
    nShow = 40                               # matches drawn per panel (readable)
    rng = np.random.default_rng(0)           # fixed seed -> same figure every run
    fig, axs = plt.subplots(len(thresholds), 1, figsize=(12, 4.2 * len(thresholds)))
    for ax, minDist in zip(axs, thresholds):
        m = matchWithNN(desc1, desc2, minDist)
        x1, x2 = matchesToPoints(kp1, kp2, m)
        print(f'minDist = {minDist:4d}: {len(m):5d} matches')
        # Random sample of the accepted matches: representative of the whole set,
        # so the growth of false positives with minDist is visible
        # (the lowest-distance matches would be identical in every panel).
        idx = rng.choice(len(m), size=min(nShow, len(m)), replace=False)
        showMatches(img1, img2, x1[:, idx], x2[:, idx], f'NN, minDist = {minDist}: '
                    f'{len(m)} matches ({len(idx)} random shown)', ax=ax)
    saveFig('ex1_NN_thresholds.png')
    
    #Results:
    ########
    #---------------------------------------------------------------------------------
    # minDist  | Total matches |	                 Interpretation
    # 100	   |     200	   | Strict threshold; relatively few accepted matches
    # 200	   |     942	   | Considerably more matches, with some visibly questionable correspondences
    # 300	   |     1,798	   | Many additional matches, including potentially incorrect ones
    # 500	   |     2,310	   | Permissive threshold; accepts many weak correspondences
    #------------------------------------------------------------------------------------

    # minDist=100 does not eliminates all false positive, it reduces the number of accepted matches.

    # The reason is that a small descriptor distance does not guarantee a correct correspondence.

    # For example, two different chessboard corners may have very similar SIFT descriptors:
    # - Correct match: distance = 80 → accepted.
    # - Incorrect match: distance = 45 → also accepted.
    # - Correct match: distance = 150 → rejected (false negative).
    # Therefore, decreasing the threshold can remove some false positives, but it can also introduce more false negatives.

    #######


    # ---------------- Exercise 2: NNDR ----------------
    print('\n--- Exercise 2: NNDR matching ---')
    minDist = 500
    ratios = [0.95, 0.8, 0.7, 0.6]
    fig, axs = plt.subplots(len(ratios), 1, figsize=(12, 4.2 * len(ratios)))
    for ax, distRatio in zip(axs, ratios):
        m = matchWith2NNDR(desc1, desc2, distRatio, minDist)
        x1, x2 = matchesToPoints(kp1, kp2, m)
        print(f'distRatio = {distRatio:.2f}: {len(m):5d} matches')
        showMatches(img1, img2, x1, x2, f'NNDR, distRatio = {distRatio}, minDist = {minDist}: '
                    f'{len(m)} matches', ax=ax)
    saveFig('ex2_NNDR_ratios.png')

    # Selected configuration (used by the RANSAC scripts)
    m = matchWith2NNDR(desc1, desc2, 0.8, 500)
    x1, x2 = matchesToPoints(kp1, kp2, m)
    showMatches(img1, img2, x1, x2, f'NNDR (distRatio 0.8): {len(m)} matches',
                fileName='ex2_NNDR_selected.png')


    #Results:
    #   | distRatio	Matches	|             Interpretation
    #   |   0.95	1,256	| Permissive; many ambiguous matches survive
    #   |   0.80	386	    | More selective, but some incorrect matches remain
    #   |   0.70	208	    | Stricter; most displayed correspondences appear more consistent
    #   |   0.60	106	    | Very strict; fewer matches and greater risk of false negatives

    # from the images, the correspondence lines become more consistent as the ratio decreases.

    # At distRatio = 0.95, there are many crossing lines connecting potentially unrelated image regions.

    # At distRatio = 0.70 and 0.60, the displayed matches are more concentrated around corresponding scene structures.


    #The key idea: NNDR rejects ambiguous matches, even when their absolute 
    # descriptor distances are small.
    #This is especially useful for repetitive patterns such as the chessboard, 
    # where different corners can produce nearly identical SIFT descriptors.



    # ---------------- Exercise 3: SuperGlue ----------------
    print('\n--- Exercise 3: SuperGlue ---')
    x1_sg, x2_sg, npz = loadSuperGlue()
    if x1_sg is not None:
        print(f'SuperGlue: {npz["keypoints0"].shape[0]} / {npz["keypoints1"].shape[0]} '
              f'keypoints, {x1_sg.shape[1]} matches')
        showMatches(img1, img2, x1_sg, x2_sg, f'SuperGlue: {x1_sg.shape[1]} matches',
                    fileName='ex3_superglue.png')

    # ---------------- Comparison table ----------------
    # SIFT methods are scored by descriptor distance (lower = better), SuperGlue by
    # match confidence in [0, 1] (higher = better): different units, separate columns.
    mNN = matchWithNN(desc1, desc2, 200)
    mNNDR = matchWith2NNDR(desc1, desc2, 0.8, 500)
    rows = [('SIFT NN (minDist 200)', len(kp1), len(kp2), len(mNN),
             f'{np.mean(mNN[:, 2]):.1f}', '-'),
            ('SIFT NNDR (ratio 0.8)', len(kp1), len(kp2), len(mNNDR),
             f'{np.mean(mNNDR[:, 2]):.1f}', '-')]
    if x1_sg is not None:
        conf = npz['match_confidence'][npz['matches'] > -1]
        rows.append(('SuperGlue (indoor)', npz['keypoints0'].shape[0],
                     npz['keypoints1'].shape[0], x1_sg.shape[1], '-', f'{np.mean(conf):.3f}'))

    header = ('Method', 'Kpts img1', 'Kpts img2', 'Matches', '% img1 matched',
              'Mean dist', 'Mean conf')
    widths = (23, 10, 10, 8, 15, 10, 10)
    print('\n--- Comparison of matching methods ---')
    print(' | '.join(h.ljust(w) for h, w in zip(header, widths)))
    print('-+-'.join('-' * w for w in widths))
    for name, k1, k2, n, dist, cf in rows:
        cells = (name, str(k1), str(k2), str(n), f'{100 * n / k1:.1f} %', dist, cf)
        print(' | '.join(c.ljust(w) for c, w in zip(cells, widths)))

    plt.show(block=True)   # block: keep windows open


    """
    --- Comparison of matching methods ---
    Method                  | Kpts img1  | Kpts img2  | Matches  | % img1 matched  | Mean dist  | Mean conf
    ------------------------+------------+------------+----------+-----------------+------------+-----------
    SIFT NN (minDist 200)   | 2310       | 6520       | 942      | 40.8 %          | 137.7      | -
    SIFT NNDR (ratio 0.8)   | 2310       | 6520       | 386      | 16.7 %          | 151.7      | -
    SuperGlue (indoor)      | 377        | 702        | 244      | 64.7 %          | -          | 0.659
    """

    # Mean dist (SIFT): Average Euclidean distance between matched SIFT descriptors. 
    #  Smaller generally means more similar descriptors.
    # Mean conf (SuperGlue): Average confidence assigned to accepted correspondences. 
    #  Larger generally indicates stronger model confidence.