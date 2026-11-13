import matplotlib.pyplot as plt
import numpy as np
import cv2

print("This code is outdate do not use!!!")
#base_path = "D:/Dev_Space/python/Lab3/"
base_path = "data/"
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

def normalization_matrix(x):
    """
    Normalización de Hartley: lleva el centroide de los puntos al origen y escala
    para que la distancia media al origen sea sqrt(2).
    Sin ella, la matriz A mezcla valores ~1 con valores ~500000 (pixeles al cuadrado)
    y la SVD queda mal condicionada.
     -input:  x: (n x 2) puntos en pixeles
     -output: T: (3 x 3) tal que x_norm = T @ [x, 1]
    """
    c = np.mean(x, axis=0)
    mean_dist = np.mean(np.linalg.norm(x - c, axis=1))
    s = np.sqrt(2) / max(mean_dist, 1e-12)
    return np.array([[s, 0, -s * c[0]],
                     [0, s, -s * c[1]],
                     [0, 0, 1]])


def compute_homography(x1, x2):
    """
    Estima H (x2 ~ H @ x1) a partir de 4+ puntos usando DLT normalizado + SVD.
    Devuelve None si la muestra es degenerada (p.ej. 3 puntos colineales).
     -input:  x1, x2: (n x 2) puntos en pixeles, n >= 4
     -output: H: (3 x 3) con H[2,2] = 1, o None
    """
    # 1. Normaliza los puntos de cada imagen
    T1, T2 = normalization_matrix(x1), normalization_matrix(x2)
    x1n = (T1 @ np.hstack([x1, np.ones((x1.shape[0], 1))]).T).T
    x2n = (T2 @ np.hstack([x2, np.ones((x2.shape[0], 1))]).T).T

    # 2. DLT con los puntos normalizados (dos ecuaciones por punto)
    n = x1.shape[0]
    A = []
    for i in range(n):
        X, Y = x1n[i, 0], x1n[i, 1]
        x, y = x2n[i, 0], x2n[i, 1]
        A.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])
    A = np.asarray(A)
    _, S, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    # 3. Muestra degenerada (p.ej. 3 puntos colineales): con 4 puntos A es 8x9 y
    #    tiene rango < 8, su último valor singular es ~0 y la solución no es única
    if n == 4 and S[-1] < 1e-8 * S[0]:
        return None

    # 4. Deshace la normalización: H = T2^-1 @ Hn @ T1
    H = np.linalg.inv(T2) @ Hn @ T1
    if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
        return None
    return H / H[2, 2]


def ransac_iterations(inlier_ratio, P=0.999, s=4):
    """
    Número de iteraciones para que, con probabilidad P, al menos una muestra
    de s puntos no contenga outliers (como en lineRANSAC.py):
        m = log(1 - P) / log(1 - w^s)
    """
    p_clean = inlier_ratio ** s
    if p_clean <= 1e-12:
        return np.inf
    if p_clean >= 1.0:
        return 1
    return int(np.ceil(np.log(1 - P) / np.log(1 - p_clean)))


def homograhy4Ransac(x1, x2, max_iterations, ransac_threshold, P=0.999):
    """
    RANSAC manual para estimar la homografía H.
    El número de iteraciones se adapta con la mejor proporción de inliers encontrada
    hasta el momento (como máximo max_iterations).
     -input:
         x1, x2: (n x 2) puntos emparejados en pixeles
         max_iterations: tope de iteraciones
         ransac_threshold: error de transferencia máximo (pixeles) para ser inlier
         P: probabilidad deseada de obtener al menos una muestra sin outliers
     -output:
         H_final: (3 x 3) homografía reestimada con todos los inliers
         best_inliers: (n,) máscara booleana de inliers
    """
    best_H = None
    best_inliers = np.zeros(x1.shape[0], dtype=bool)
    max_votes = 0

    n_points = x1.shape[0]
    x1_h = np.hstack([x1, np.ones((n_points, 1))])
    iterations = max_iterations
    i = 0
    while i < iterations:
        i += 1
        #1 Selecciona 4 puntos aleatorios
        idx = np.random.choice(n_points, 4, replace=False)
        H = compute_homography(x1[idx], x2[idx])
        if H is None:                      # muestra degenerada, se descarta
            continue

        #2 Proyecta todos los puntos de x1
        projected = (H @ x1_h.T).T
        #por numpy, transpone y normaliza
        with np.errstate(divide='ignore', invalid='ignore'):
            projected = projected[:, :2] / projected[:, [2]]

        #3 Calcula error (transferencia en la imagen 2)
        errors = np.linalg.norm(x2 - projected, axis=1)
        errors = np.nan_to_num(errors, nan=np.inf)

        #4 Inliers
        inliers = errors < ransac_threshold
        votes = np.sum(inliers)

        #5 Guarda mejor H y actualiza el número de iteraciones necesario
        if votes > max_votes:
            max_votes = votes
            best_inliers = inliers
            best_H = H
            iterations = min(max_iterations, ransac_iterations(votes / n_points, P))

    print(f"RANSAC: {i} iteraciones")
    if best_H is None or max_votes < 4:
        raise RuntimeError("RANSAC no ha encontrado ninguna hipótesis válida")

    #Recalcula H final con los inliers y los vuelve a calcular con ella
    H_final = compute_homography(x1[best_inliers], x2[best_inliers])
    projected = (H_final @ x1_h.T).T
    projected = projected[:, :2] / projected[:, [2]]
    best_inliers = np.linalg.norm(x2 - projected, axis=1) < ransac_threshold
    return H_final, best_inliers


def points_to_matches(x1, x2):
    """
    Crea KeyPoints y DMatch 1 a 1 a partir de puntos emparejados (n x 2), para poder
    dibujar con cv2.drawMatches cualquier conjunto de matches (SIFT o SuperGlue).
    """
    kp1 = [cv2.KeyPoint(float(u), float(v), 5) for u, v in x1]
    kp2 = [cv2.KeyPoint(float(u), float(v), 5) for u, v in x2]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(len(kp1))]
    return kp1, kp2, matches


def show_results(name, image_1, image_2, x1, x2, H_final, best_inliers):
    """
    Muestra los resultados de RANSAC para un conjunto de matches (x1, x2: n x 2):
      1. Identificación del plano: la imagen 1 transformada con H superpuesta a la
         imagen 2. Solo el plano que induce H coincide; el resto sale duplicado.
      2. Inliers (verde) y outliers (rojo).
    """
    print(f"\n===== {name} =====")
    print("Homografía estimada:\n", H_final)
    print(f"Inliers encontrados: {np.sum(best_inliers)} / {len(best_inliers)}")
    print(f"Total de outliers: {np.sum(~best_inliers)}")

    img1_rgb = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

    # 1. Plano: transfiere la imagen 1 a la imagen 2 con H y mezcla ambas
    h2, w2 = image_2.shape[:2]
    warped = cv2.warpPerspective(img1_rgb, H_final, (w2, h2))
    blend = cv2.addWeighted(warped, 0.5, img2_rgb, 0.5, 0)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].imshow(img1_rgb)
    axs[0].plot(x1[best_inliers, 0], x1[best_inliers, 1], 'g.', ms=5)
    axs[0].set_title("Imagen 1: inliers (verde)")
    axs[1].imshow(blend)
    axs[1].set_title("Imagen 1 transformada con H sobre la imagen 2\n"
                     "(el plano de la homografía coincide, el resto sale doble)")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Plano estimado por la homografía con {name} (RANSAC)")
    plt.tight_layout(rect=(0, 0, 1, 0.92))   # deja espacio para el suptitle

    # 2. Inliers y outliers, dibujados con los puntos de ESTE conjunto de matches
    kp1, kp2, matches = points_to_matches(x1, x2)
    inlier_matches = [matches[i] for i in range(len(matches)) if best_inliers[i]]
    outlier_matches = [matches[i] for i in range(len(matches)) if not best_inliers[i]]

    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    img_combined = cv2.drawMatches(
        img1_rgb, kp1, img2_rgb, kp2, inlier_matches, None, matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axs[0].imshow(img_combined)
    axs[0].set_title(f"{name}: {len(inlier_matches)} inliers (verde) detectados por RANSAC")
    img_combined = cv2.drawMatches(
        img1_rgb, kp1, img2_rgb, kp2, outlier_matches, None, matchColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axs[1].imshow(img_combined)
    axs[1].set_title(f"{name}: {len(outlier_matches)} outliers (rojo) detectados por RANSAC")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    path_image_1 = base_path + 'image1.png'
    path_image_2 = base_path + 'image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)

    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    distRatio = 0.7
    #Discutir lo de la distancia después
    minDist = 500
    matchesList = matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # ----------------- RANSAC con matches SIFT (NNDR) -----------------
    # Umbral de 5 px. Iteraciones adaptativas: 5000 es solo el máximo.
    H_sift, inliers_sift = homograhy4Ransac(srcPts, dstPts, 5000, 5.0)
    show_results("SIFT", image_pers_1, image_pers_2, srcPts, dstPts, H_sift, inliers_sift)

    # ----------------- RANSAC con matches SuperGlue -----------------
    #npz = np.load("D:/Dev_Space/python/Lab3/output/image1_image2_matches.npz")
    npz = np.load("output/image1_image2_matches.npz")

    mask = npz['matches'] > -1
    idxs = npz['matches'][mask]

    x1_sp = npz['keypoints0'][mask]   # puntos en imagen 1
    x2_sp = npz['keypoints1'][idxs]   # puntos en imagen 2

    H_sg, inliers_sg = homograhy4Ransac(x1_sp, x2_sp, 5000, 5.0)
    show_results("SuperGlue", image_pers_1, image_pers_2, x1_sp, x2_sp, H_sg, inliers_sg)

    # block=True: las ventanas no se cierran solas (también en el depurador de VS Code)
    plt.show(block=True)
