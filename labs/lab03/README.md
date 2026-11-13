# Lab 3 – Robust homography and fundamental matrix estimation

| Exercise | Script | Figures (`figures/`) |
|---|---|---|
| 1. NN matching, threshold sweep | `matching.py` | `ex1_NN_thresholds.png` |
| 2. NNDR matching (`matchWith2NNDR`) | `matching.py` | `ex2_NNDR_ratios.png`, `ex2_NNDR_selected.png` |
| 3. SuperGlue matches | `matching.py` (loads `data/image1_image2_matches.npz`) | `ex3_superglue.png` |
| 4. RANSAC homography | `ransacHomography.py` | `ex4_*` |
| 5. RANSAC fundamental matrix | `ransacFundamental.py` | `ex5_*` |
| 6. Guided matching (optional) | `ransacFundamental.py` | `ex6_guided_new_matches.png` |

Run from this folder with the `computerVision` venv, e.g.
`~/venvs/computerVision/bin/python ransacHomography.py`.
The SuperGlue parts run automatically once the `.npz` file exists.

## Generating the SuperGlue matches (Appendix A)

The repository is cloned at `~/SuperGluePretrainedNetwork`.

1. (A.4, optional) In `match_pairs.py` around line 281, also save the descriptors:
   ```python
   out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                  'matches': matches, 'match_confidence': conf,
                  'descriptors0': pred['descriptors0'], 'descriptors1': pred['descriptors1']}
   ```
2. Run (the pairs file `data/image_pairs.txt` is already created):
   ```bash
   cd ~/SuperGluePretrainedNetwork
   L=~/Documents/CV_2026/computer-vision-coursework/labs/lab03/data
   ~/venvs/computerVision/bin/python match_pairs.py --resize 752 --superglue indoor \
       --max_keypoints 2048 --nms_radius 3 --resize_float \
       --input_dir $L --input_pairs $L/image_pairs.txt --output_dir $L --viz
   ```
   This writes `data/image1_image2_matches.npz` (and a visualization `.png`).

## Results and discussion (SIFT)

**Ex 1 – NN.** 2310 / 6520 SIFT features. Every feature of image 1 gets a nearest neighbour,
so the threshold `minDist` only trades matches for errors: 200 matches at 100, 942 at 200,
2310 at 500. Even at the strictest threshold many matches are false positives (crossing lines),
and many correct ones are lost (false negatives). On the chessboard, all corners look almost
the same, so a corner's nearest neighbour is often a *different* corner of the board with an
equally small distance. An absolute distance threshold cannot tell them apart: the wrong
match is as similar as the right one (aliasing / repeated texture).

**Ex 2 – NNDR.** Requiring `d1 < distRatio * d2` rejects ambiguous matches where the best and
second-best are similar (chessboard, repeated mats). 1256 matches at 0.95, 386 at 0.8,
208 at 0.7, 106 at 0.6. Used 0.8 (Lowe) for the rest of the lab: mostly consistent matches.

**Ex 4 – Homography RANSAC** (4-point normalized DLT, transfer error, 2 px threshold, adaptive
number of iterations with P = 0.999, LS refit on inliers): 165 / 386 inliers, 0.86 px RMSE,
about 240 iterations. The warped image 1 aligns exactly with image 2 on the **back wall (padded
mats + chessboard)**, while the floor and the side panel are doubled. Using the ground-truth pose
to recover the plane from H gives a normal of about (-1, 0.05, 0.02) in world coordinates (a
vertical wall) at 2.6 m from camera 1.

**Ex 5 – Fundamental RANSAC** (normalized 8-point + rank 2, symmetric point-to-epipolar-line
distance, 2 px threshold): about 230 / 386 inliers with 0.5 px median epipolar distance, and all
epipolar lines meet at one epipole. **But the epipole is not well determined with SIFT matches**:
e2 ≈ (728, 261) vs ground truth (883, -258), and it changes notably with the RANSAC seed.
Reasons:
- *Planar degeneracy*: about 165 of the inliers lie on the wall plane. Points on a plane only
  constrain F up to a 2-parameter family, so few, clustered off-plane matches decide the epipole.
- The images are raw EuRoC images with strong radial distortion that the pinhole model ignores.
  The given poses are also only approximate: the ground-truth F has about a 7 px residual even on
  certain wall matches.
- Sanity check: on synthetic matches projected with the ground-truth poses (noise + 50 % outliers)
  the same code recovers e2 = (867, -249) vs GT (883, -258). So the implementation is correct and
  the problem is the match distribution. SuperGlue should give many more matches spread over the
  floor and side panels; compare its epipole.

**Ex 6 – Guided matching.** With F, candidates for a feature are only those within 2 px of its
epipolar line, and NNDR is applied among them. The distractors elsewhere in the image disappear,
so many matches rejected by global NNDR are recovered (303 new). Its quality depends directly on
F: a biased F (as with SIFT here) guides the search to the wrong line.
