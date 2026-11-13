#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: RANSAC homography estimation (Exercise 4)
#
#####################################################################################
from matching import *


#########################################################################
# Generic RANSAC (based on lineRANSAC.py)
#########################################################################
def nRansacAttempts(inlierFrac, pMinSet, P=0.999):
    """
    Number of random samples so that, with probability P, at least one minimal
    sample is free of outliers:  m = log(1 - P) / log(1 - w^s)
    """
    pGood = np.clip(inlierFrac, 0.0, 1.0)**pMinSet   # P(sample without outliers)
    if pGood <= 1e-12:
        return np.iinfo(np.int32).max                 # caller caps it with maxAttempts
    if pGood >= 1.0:
        return 1
    return int(np.ceil(np.log(1 - P) / np.log1p(-pGood)))


def ransac(x1, x2, pMinSet, fitModel, residuals, threshold, P=0.999,
           maxAttempts=20000, seed=0, nStore=0):
    """
    Generic RANSAC for two-view models (homography, fundamental matrix).
    Each iteration: random minimal sample -> model hypothesis -> votes = number of
    matches whose residual is below the threshold. The most voted hypothesis wins.
    The number of attempts is updated adaptively with the best inlier ratio found.
     -input:
         x1, x2: (3xN) homogeneous matched points.
         pMinSet: size of the minimal sample (4 for H, 8 for F).
         fitModel: f(x1, x2) -> model (3x3) or None.
         residuals: f(model, x1, x2) -> (N,) error in pixels.
         threshold: inlier threshold in pixels.
         nStore: if > 0, keep the hypotheses that improved the best vote count
                 (sample, inliers) and return nStore of them (first, ..., last) for plotting.
     -output:
         model: most voted model.
         inliers: (N,) boolean mask.
         history: list of dicts {'sample', 'inliers', 'nVotes', 'iteration'}.
    """

    """
    Step 1 — Random sampling

    Randomly select 4 correspondences from your matches.

    Step 2 — Model estimation

    Compute a candidate homography H using those four correspondences.


    Step 3 — Evaluate all matches

    Use H to transfer each point from image 1 to image 2. Measure its distance to the corresponding observed point.


    Step 4 — Count inliers

    Count how many matches have an error below the chosen threshold. These are the votes for that hypothesis.

    Step 5 — Repeat and select

    Repeat with new random samples, retaining the homography with the largest inlier count.
    """
        
    rng = np.random.default_rng(seed)
    N = x1.shape[1]
    nAttempts = maxAttempts
    bestVotes, bestModel, bestInliers = -1, None, np.zeros(N, bool)
    history = []

    k = 0
    while k < nAttempts:
        sample = rng.choice(N, size=pMinSet, replace=False)
        model = fitModel(x1[:, sample], x2[:, sample])
        k += 1
        if model is None:
            continue
        inliers = residuals(model, x1, x2) < threshold
        nVotes = int(np.sum(inliers))

        if nVotes > bestVotes:
            if nStore > 0:
                history.append({'sample': sample, 'inliers': inliers,
                                'nVotes': nVotes, 'iteration': k})
            bestVotes, bestModel, bestInliers = nVotes, model, inliers
            # Adaptive number of attempts with the current inlier ratio
            nAttempts = min(maxAttempts, nRansacAttempts(nVotes / N, pMinSet, P))

    if len(history) > nStore:   # evenly spaced improvements, always first and last
        history = [history[i] for i in np.linspace(0, len(history) - 1, nStore).astype(int)]
    print(f'  RANSAC: {k} attempts, {bestVotes}/{N} inliers ({100 * bestVotes / N:.1f} %)')
    return bestModel, bestInliers, history


#########################################################################
# Homography model
#########################################################################
def normalizationMatrix(x):
    """
    Hartley normalization: translate the centroid to the origin and scale so that the
    mean distance to it is sqrt(2). Improves the conditioning of the DLT/8-point.
     -output:
         T: (3x3) so that x_norm = T @ x
    """
    p = x[:2] / x[2]
    c = np.mean(p, axis=1)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c[:, None], axis=0)), 1e-12)
    return np.array([[s, 0, -s * c[0]],
                     [0, s, -s * c[1]],
                     [0, 0, 1]])


def homographyFromMatches(x1, x2):
    """
    Normalized DLT: H_21 such that x2 ~ H_21 @ x1, from N >= 4 matches.
    Each match gives two rows of A (2N x 9), h = right singular vector of A
    with the smallest singular value. Coordinates are normalized first (Hartley).
     -input:
         x1, x2: (3xN) homogeneous points.
     -output:
         H_21: (3x3) with H_21[2,2] = 1, or None if degenerate.
    """

    T1, T2 = normalizationMatrix(x1), normalizationMatrix(x2)
    p1 = T1 @ (x1 / x1[2])
    p2 = T2 @ (x2 / x2[2])
    u1, v1, u2, v2 = p1[0], p1[1], p2[0], p2[1]
    one, zero = np.ones_like(u1), np.zeros_like(u1)
    A = np.vstack([np.stack([u1, v1, one, zero, zero, zero, -u2 * u1, -u2 * v1, -u2], 1),
                   np.stack([zero, zero, zero, u1, v1, one, -v2 * u1, -v2 * v1, -v2], 1)])
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1                  # undo the normalization
    if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
        return None
    return H / H[2, 2]


def homographyResiduals(H_21, x1, x2):
    """Transfer error in image 2: e_i = || x2_i - pixels(H_21 @ x1_i) || (pixels)."""
    x2p = H_21 @ x1
    with np.errstate(divide='ignore', invalid='ignore'):
        x2p = x2p[:2] / x2p[2]
    e = np.linalg.norm(x2[:2] / x2[2] - x2p, axis=0)
    return np.nan_to_num(e, nan=np.inf)


def ransacHomography(x1, x2, threshold=2.0, nStore=0, seed=0):
    """
    RANSAC with the 4-point minimal solution, then least-squares refit of H with all
    inliers (and inliers recomputed with the refined H).
    """
    H, inliers, history = ransac(x1, x2, 4, homographyFromMatches, homographyResiduals,
                                 threshold, nStore=nStore, seed=seed)
    H_ls = homographyFromMatches(x1[:, inliers], x2[:, inliers])
    inliers_ls = homographyResiduals(H_ls, x1, x2) < threshold
    print(f'  After LS refinement: {np.sum(inliers_ls)} inliers')
    return H_ls, inliers_ls, history


#########################################################################
# Plane identification
#########################################################################
def planeFromHomography(H_21, K1, K2, T_w_c1, T_w_c2):
    """
    Recover the scene plane that induces H_21, using the ground-truth relative pose.
        K2^-1 H_21 K1 = lambda * (R_21 - t_21 n^T / d)
    Linear in lambda and v = lambda * n / d  (9 equations, 4 unknowns -> least squares).
     -output:
         n_c1: (3,) unit normal of the plane in camera 1, d: distance (m) to camera 1,
         n_w: (3,) normal in the world frame.
    """
    T_21 = np.linalg.inv(T_w_c2) @ T_w_c1
    R, t = T_21[:3, :3], T_21[:3, 3]
    A = np.linalg.inv(K2) @ H_21 @ K1
    # vec(A) = lambda * vec(R) - vec(t v^T)  -> unknowns [lambda, v1, v2, v3]
    M = np.zeros((9, 4))
    M[:, 0] = R.reshape(-1)
    for j in range(3):
        E = np.zeros((3, 3)); E[:, j] = -t
        M[:, 1 + j] = E.reshape(-1)
    sol, *_ = np.linalg.lstsq(M, A.reshape(-1), rcond=None)
    lam, v = sol[0], sol[1:]
    m = v / lam                                   # n / d
    d = 1.0 / np.linalg.norm(m)
    n_c1 = m * d
    if n_c1[2] > 0:                               # normal pointing towards camera 1
        n_c1, d = -n_c1, d
    n_w = T_w_c1[:3, :3] @ n_c1
    return n_c1, d, n_w


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
