import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Rutas relativas a este fichero (funciona sea cual sea el directorio de trabajo)
LAB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LAB_DIR, "data")
OUTPUT_DIR = os.path.join(LAB_DIR, "output")

# ==========================================================
# Ground truth (enunciado de la práctica)
# ==========================================================
K = np.array([[458.654, 0.0, 367.215],
              [0.0, 457.296, 248.375],
              [0.0, 0.0, 1.0]])

T_w_c1 = np.array([[-0.0744, -0.3079,  0.9485, 0.8801],
                   [-0.997,   0.0428, -0.0643, 0.8695],
                   [-0.0208, -0.9505, -0.3102, 1.73],
                   [0.0,      0.0,     0.0,    1.0]])

T_w_c2 = np.array([[-0.512,  -0.2779,  0.8128, 0.5236],
                   [-0.8586,  0.194,  -0.4745, 1.9537],
                   [-0.0258, -0.9408, -0.338,  1.2876],
                   [0.0,      0.0,     0.0,    1.0]])


def epipolesGroundTruth(K, T_w_c1, T_w_c2):
    """
    Epipolos de referencia:
      e1 = proyección del centro óptico de la cámara 2 en la imagen 1
      e2 = proyección del centro óptico de la cámara 1 en la imagen 2
    El centro óptico de cada cámara en el mundo es la última columna de T_w_c.
    """
    def project(T_w_c, X_w):
        X_c = np.linalg.inv(T_w_c) @ X_w      # mundo -> cámara
        x = K @ X_c[:3]
        return x / x[2]
    e1 = project(T_w_c1, T_w_c2[:, 3])
    e2 = project(T_w_c2, T_w_c1[:, 3])
    return e1, e2


# ==========================================================
# Fundamental matrix (8-point algorithm)
# ==========================================================
def computeFundamental(pts1, pts2):
    """
    Algoritmo de 8 puntos normalizado: F tal que x2^T F x1 = 0.
     -input:  pts1, pts2: (n x 2) puntos en pixeles, n >= 8
     -output: F (3x3) de rango 2 y norma 1, o None si la muestra es degenerada
    """
    def normalize(xy):
        # Normalización de Hartley: centroide en el origen, distancia media sqrt(2)
        m = xy.mean(axis=0)
        d = np.sqrt(((xy - m)**2).sum(axis=1)).mean()
        s = np.sqrt(2)/(d+1e-12)
        T = np.array([[s,0,-s*m[0]],[0,s,-s*m[1]],[0,0,1]])
        xh = np.hstack([xy, np.ones((xy.shape[0],1))]).T
        xn = (T @ xh); xn = (xn[:2]/xn[2]).T
        return xn, T

    x1n, T1 = normalize(pts1)
    x2n, T2 = normalize(pts2)

    # Una ecuación por match: x2^T F x1 = 0, lineal en los 9 elementos de F
    x, y = x1n[:,0], x1n[:,1]          # imagen 1
    u, v = x2n[:,0], x2n[:,1]          # imagen 2
    A = np.vstack([u*x, u*y, u, v*x, v*y, v, x, y, np.ones_like(x)]).T

    _, S_A, Vt = np.linalg.svd(A)
    # Muestra degenerada: con 8 puntos A es 8x9; si su rango es < 8 la solución no es única
    if pts1.shape[0] == 8 and S_A[-1] < 1e-8 * S_A[0]:
        return None
    Fh = Vt[-1].reshape(3,3)

    # Restricción de rango 2
    U,S,Vt = np.linalg.svd(Fh)
    S[-1] = 0.0
    Fh = U @ np.diag(S) @ Vt

    # Deshacer la normalización
    F = T2.T @ Fh @ T1
    return F / (np.linalg.norm(F) + 1e-12)


def epipolarError(F, x1_h, x2_h):
    """
    Error geométrico simétrico (pixeles^2): d1^2 + d2^2, donde
      d1 = distancia de x2 a su línea epipolar F x1     (imagen 2)
      d2 = distancia de x1 a su línea epipolar F^T x2   (imagen 1)
    """
    Fx1 = (F @ x1_h.T).T      # Líneas en imagen 2
    Ftx2 = (F.T @ x2_h.T).T   # Líneas en imagen 1
    # Término común de la ecuación epipolar
    x2tFx1 = np.sum(x2_h * Fx1, axis=1)
    d1 = np.abs(x2tFx1) / np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2)
    d2 = np.abs(x2tFx1) / np.sqrt(Ftx2[:, 0]**2 + Ftx2[:, 1]**2)
    return d1**2 + d2**2


def ransac_iterations(inlier_ratio, P=0.999, s=8):
    """Iteraciones para obtener con probabilidad P una muestra de s puntos sin outliers."""
    p_clean = inlier_ratio ** s
    if p_clean <= 1e-12:
        return np.inf
    if p_clean >= 1.0:
        return 1
    return int(np.ceil(np.log(1 - P) / np.log(1 - p_clean)))


# ==========================================================
# RANSAC estimation of Fundamental matrix
# ==========================================================
def ransacFundamental(x1, x2, max_iterations, threshold, P=0.999):
    """
    RANSAC con la solución mínima de 8 puntos.
     -input:
         x1, x2: (n x 2) puntos emparejados en pixeles
         max_iterations: máximo de iteraciones (el número real se adapta con la
                         mejor proporción de inliers encontrada)
         threshold: umbral en PIXELES. Un match es inlier si d1^2 + d2^2 < threshold^2
     -output:
         F_final: F reestimada con todos los inliers
         best_inliers: (n,) máscara booleana de inliers
    """
    best_F, best_inliers = None, np.zeros(x1.shape[0], dtype=bool)
    max_votes = 0

    # Puntos homogéneos
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])

    iterations = max_iterations
    i = 0
    while i < iterations:
        i += 1
        # Seleccionar 8 puntos aleatorios
        idx = np.random.choice(x1.shape[0], 8, replace=False)
        F = computeFundamental(x1[idx], x2[idx])
        if F is None:               # muestra degenerada, se descarta
            continue

        # Inliers por umbral
        error = epipolarError(F, x1_h, x2_h)
        inliers = error < threshold**2
        votes = np.sum(inliers)

        # Actualizar el mejor modelo y el número de iteraciones necesario
        if votes > max_votes:
            max_votes = votes
            best_F = F
            best_inliers = inliers
            iterations = min(max_iterations, ransac_iterations(votes / x1.shape[0], P))

    print(f"RANSAC: {i} iteraciones")
    if best_F is None or max_votes < 8:
        raise RuntimeError("RANSAC no ha encontrado ninguna hipótesis válida")

    # Recalcular F con todos los inliers, y los inliers con la F final
    F_final = computeFundamental(x1[best_inliers], x2[best_inliers])
    best_inliers = epipolarError(F_final, x1_h, x2_h) < threshold**2
    return F_final, best_inliers


# ==========================================================
# Draw epipolar lines using inlier matches
# ==========================================================
def draw_line(ax, l, x_range, color):
    """Dibuja la recta l = (a, b, c), a*x + b*y + c = 0, en el rango de x dado."""
    a, b, c = l
    if abs(b) > 1e-12:
        xs = np.array(x_range, dtype=float)
        ys = -(a * xs + c) / b
    else:                        # recta vertical
        xs = np.array([-c / a, -c / a])
        ys = np.array([-1e5, 1e5])
    ax.plot(xs, ys, '-', color=color, linewidth=1)


def plot_epipolar_lines(F, pts1, pts2, img1, img2, e1, e2, e1_gt, e2_gt, title, n_lines=15):
    """
    Dibuja las líneas epipolares en AMBAS imágenes para n_lines matches inliers:
      - en img2 para los puntos de img1: l2 = F * x1
      - en img1 para los puntos de img2: l1 = F^T * x2
    Todas las líneas de una imagen deben cortarse en su epipolo. Los epipolos suelen
    caer fuera de la imagen, así que los ejes se amplían para que se vean, junto con
    los epipolos del ground truth.
    """
    n = min(n_lines, len(pts1))
    idx = np.random.choice(len(pts1), size=n, replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, n))

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    for ax, img, pts_self, pts_other, F_dir, e, e_gt, name in (
            (axs[0], img1, pts1, pts2, F.T, e1, e1_gt, "e1"),
            (axs[1], img2, pts2, pts1, F,   e2, e2_gt, "e2")):
        h, w = img.shape[:2]
        ax.imshow(img, cmap="gray")
        # Rango visible: la imagen y los dos epipolos (estimado y ground truth)
        xmin, xmax = min(0, e[0], e_gt[0]) - 30, max(w, e[0], e_gt[0]) + 30
        ymin, ymax = min(0, e[1], e_gt[1]) - 30, max(h, e[1], e_gt[1]) + 30
        for c, i in zip(colors, idx):
            l = F_dir @ np.array([pts_other[i, 0], pts_other[i, 1], 1.0])
            draw_line(ax, l, (xmin, xmax), c)
            ax.plot(pts_self[i, 0], pts_self[i, 1], 'o', color=c, markeredgecolor='k')
        ax.plot(e[0], e[1], 'r+', markersize=18, mew=3, label=f"{name} estimado")
        ax.plot(e_gt[0], e_gt[1], 'bx', markersize=14, mew=3, label=f"{name} ground truth")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)          # eje y hacia abajo, como en la imagen
        ax.legend(loc="lower right")
    axs[0].set_title("Imagen 1: líneas epipolares F^T x2")
    axs[1].set_title("Imagen 2: líneas epipolares F x1")
    fig.suptitle(title)
    plt.tight_layout()


def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        d_best = dist[indexSort[0]]
        d_second_sort = dist[indexSort[1]]
        ratio = d_best / d_second_sort
        if ((ratio < distRatio) and (dist[indexSort[0]] < minDist)):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])


    return matches

# ==========================================================
# Load matches (SIFT or SuperGlue)
# ==========================================================
def load_images():
    img1 = cv2.imread(os.path.join(DATA_DIR, "image1.png"), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(DATA_DIR, "image2.png"), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"No se encuentran image1.png / image2.png en {DATA_DIR}")
    return img1, img2


def load_matches(mode, img1, img2):
    if mode == "sift":
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
        kpts1, desc1 = sift.detectAndCompute(img1, None)
        kpts2, desc2 = sift.detectAndCompute(img2, None)

        distRatio = 0.7
        #Discutir lo de la distancia después
        minDist = 500
        matchesList = matchWith2NDRR(desc1, desc2, distRatio, minDist)
        dMatchesList = indexMatrixToMatchesList(matchesList)
        dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

        # Matched points in numpy from list of DMatches
        srcPts = np.float32([kpts1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        dstPts = np.float32([kpts2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
        print(f"[SIFT] Matches: {len(dMatchesList)}")
        return srcPts, dstPts

    elif mode == "superglue":
        npz = np.load(os.path.join(OUTPUT_DIR, "image1_image2_matches.npz"))
        mask = npz["matches"] > -1
        idxs = npz["matches"][mask]
        srcPts = npz["keypoints0"][mask]
        dstPts = npz["keypoints1"][idxs]
        print(f"[SuperGlue] Matches: {len(srcPts)}")
        return srcPts, dstPts

    raise ValueError(f"mode desconocido: {mode}")


def epipole(F, which='right'):
    """
    which='right' -> e2 (imagen 2): espacio nulo de F^T  (e2^T F = 0)
    which='left'  -> e1 (imagen 1): espacio nulo de F    (F e1 = 0)
    Devuelve el epipolo en coordenadas homogéneas con tercera componente 1 (pixeles).
    """
    if which == 'right':  # e2
        _,_,Vt = np.linalg.svd(F.T)
    else:                 # e1
        _,_,Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / e[-1]


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(0)                     # resultados reproducibles

    img1, img2 = load_images()
    e1_gt, e2_gt = epipolesGroundTruth(K, T_w_c1, T_w_c2)
    print("Epipolos ground truth: e1 =", e1_gt[:2], " e2 =", e2_gt[:2])

    # Umbral en pixeles: inlier si d1^2 + d2^2 < threshold^2
    # (3 px ~ el umbral original de 8 px^2; con 10 semillas dio el epipolo más estable)
    threshold = 3.0
    results = []
    for mode in ("sift", "superglue"):
        print(f"\n===== {mode.upper()} =====")
        srcPts, dstPts = load_matches(mode, img1, img2)
        F, inliers = ransacFundamental(srcPts, dstPts, max_iterations=5000, threshold=threshold)
        print("Fundamental Matrix:\n", F)
        print("Rank(F):", np.linalg.matrix_rank(F))
        print(f"Inliers: {np.sum(inliers)} / {len(inliers)}")

        e1 = epipole(F, 'left')   # image 1
        e2 = epipole(F, 'right')  # image 2
        err1 = np.linalg.norm(e1[:2] - e1_gt[:2])
        err2 = np.linalg.norm(e2[:2] - e2_gt[:2])
        print(f"e1: {e1[:2]}  (ground truth {e1_gt[:2]}, error {err1:.1f} px)")
        print(f"e2: {e2[:2]}  (ground truth {e2_gt[:2]}, error {err2:.1f} px)")
        results.append((mode.upper(), len(inliers), np.sum(inliers), err1, err2))

        plot_epipolar_lines(F, srcPts[inliers], dstPts[inliers], img1, img2,
                            e1, e2, e1_gt, e2_gt,
                            f"Líneas epipolares de los inliers ({mode.upper()})")

    print("\n===== Comparación =====")
    print(f"{'Método':<10} {'Matches':>8} {'Inliers':>8} {'Error e1 (px)':>14} {'Error e2 (px)':>14}")
    for name, n, n_inl, err1, err2 in results:
        print(f"{name:<10} {n:>8} {n_inl:>8} {err1:>14.1f} {err2:>14.1f}")

    # block=True: las ventanas no se cierran solas (también en el depurador de VS Code)
    plt.show(block=True)


"""
Conclusión (umbral 3 px; error de e2 respecto al ground truth en 10 ejecuciones):
  SIFT+NNDR: mediana ~600 px   SuperGlue: mediana ~270 px (mínimo ~70 px)
En ambos casos las líneas epipolares se cortan en un único epipolo y los inliers quedan a
menos de 3 px de sus líneas, pero el epipolo varía mucho entre ejecuciones:
  - Con SIFT+NNDR casi todos los inliers están en el plano de la pared, configuración casi
    degenerada para F (los puntos de un plano no determinan F).
  - SuperGlue da matches repartidos en varios planos y profundidades (tablero, pared,
    suelo), lo que fija mejor el epipolo, aunque no del todo.
  - El epipolo cae muy fuera de la imagen (e2 ~ (883, -258)): un error angular pequeño
    en las líneas lo desplaza cientos de pixeles.
  - Las imágenes tienen distorsión radial sin corregir y las poses del ground truth son
    aproximadas (la F del ground truth tiene varios pixeles de error en matches correctos).
"""
