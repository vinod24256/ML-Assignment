# Verification Report (CHECKS.md)

This document records the exact verification checks performed on all models, datasets, metrics, and visual artifacts for the IIIT Sri City Machine Learning assignment.

---

## 1. Q1 Classification — Cardiotocography (CTG)
- **Data Source & Preprocessing**:
  - UCI ID: 193 (Cardiotocography). 2,126 fetal heart rate recordings, 21 numeric features.
  - Class distribution of `NSP`: Normal (1) = 1,655 (77.8%), Suspect (2) = 295 (13.9%), Pathologic (3) = 176 (8.3%).
  - Target is `NSP`; `CLASS` (morphologic code) was explicitly removed to avoid label leakage.
  - Split: Stratified 80/20 train/test (`train_test_split(..., stratify=y, test_size=0.20, random_state=42)`). Train = 1,700 rows, Test = 426 rows. Proportions strictly preserved across train and test.
  - Scaler (`StandardScaler`) was placed inside cross-validation pipelines (no leakage into validation/test folds).
- **Metric Verification (`results/q1_metrics_test.csv`)**:
  - All four models evaluated on the held-out test set with 5 headline metrics:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | OvR Macro ROC-AUC |
|---|---|---|---|---|---|
| **KNN** (k=4, manhattan, distance) | 0.9108 | 0.8328 | 0.8186 | 0.8255 | 0.9531 |
| **Decision Tree** (depth=8, entropy, leaf=1) | 0.9155 | 0.8583 | 0.8113 | 0.8321 | 0.9171 |
| **SVM (RBF)** (C=100, $\gamma$=0.03) | **0.9178** | **0.8649** | 0.8293 | **0.8450** | **0.9696** |
| **1D CNN (bonus)** (6 conv + 2 dense) | 0.8826 | 0.7538 | **0.8677** | 0.7981 | 0.9564 |

- **ROC-AUC Bootstrap & Statistical Significance (`results/q1_auc_bootstrap_ci.csv`)**:
  - 500 bootstrap iterations on the test set:
    - SVM (RBF): AUC = 0.9696, 95% CI [0.9546, 0.9814]
    - 1D CNN: AUC = 0.9564, 95% CI [0.9356, 0.9731], $\Delta$AUC vs SVM = -0.0130, 95% CI [-0.0293, +0.0009] (paired CI contains 0; indistinguishable from SVM)
    - KNN: AUC = 0.9531, 95% CI [0.9289, 0.9744], $\Delta$AUC vs SVM = -0.0166, 95% CI [-0.0348, +0.0007] (paired CI contains 0; indistinguishable from SVM)
    - Decision Tree: AUC = 0.9171, 95% CI [0.8747, 0.9501], $\Delta$AUC vs SVM = -0.0536, 95% CI [-0.0906, -0.0220] (statistically significantly lower than SVM)
- **Artifacts Present**:
  - Confusion matrices (counts & normalized): `figures/q1/08_confusion_counts.png`, `figures/q1/09_confusion_normalised.png`
  - ROC curves (per-class and macro): `figures/q1/11_roc_curves.png`
  - Overfitting & complexity curves: `figures/q1/03_knn_k_curve.png`, `figures/q1/04_dt_depth_curve.png`, `figures/q1/06_svm_C_gamma_heatmaps.png`, `figures/q1/07_cnn_training_curves.png`, `figures/q1/12_learning_curves.png`.

---

## 2. Q2 Regression — Parkinson's Telemonitoring
- **Data Source & Splitting**:
  - UCI ID: 189 (Parkinson's Telemonitoring). 5,875 voice recordings from 42 subjects. Target: `total_UPDRS`.
  - Primary split is **subject-wise** via `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`.
  - Train: 4,640 records from 33 patients; Test: 1,235 records from 9 patients. No patient appears in both train and test (`assert set(gtr).isdisjoint(gte)` passed).
  - Hyper-parameter tuning performed strictly on training patients via `GroupKFold(n_splits=5)`.
- **Gradient Descent vs Closed-Form OLS**:
  - Linear regression via batch gradient descent implemented from scratch (`lr=0.1, n_iter=50000`).
  - GD final train MSE = 93.9449, closed-form OLS train MSE = 93.8967 (difference < 0.05).
  - Maximum absolute prediction difference between GD and OLS on the test set is 0.0072. Agreement verified.
- **Metric Verification (`results/q2_metrics_test.csv`)**:

| Model | Test MSE | Test RMSE | Test $R^2$ | Test Pearson R |
|---|---|---|---|---|
| **Baseline (mean)** | 89.6240 | 9.4670 | -0.0816 | NaN |
| **Linear regression (GD)** | 130.7405 | 11.4342 | -0.5778 | 0.1195 |
| **Linear regression (closed form)** | 130.9058 | 11.4414 | -0.5798 | 0.1187 |
| **Polynomial (deg 1, top-5 feats)** | 130.4986 | 11.4236 | -0.5749 | 0.1044 |
| **Polynomial (deg 8, top-5 feats) [overfit]** | $1.1576 \times 10^{11}$ | $3.4023 \times 10^5$ | $-1.3970 \times 10^9$ | 0.0495 |
| **Ridge (CV $\alpha=2682.7$)** | 99.9557 | 9.9978 | -0.2063 | 0.1359 |
| **Lasso (CV $\alpha=1.2068$)** | 106.0178 | 10.2965 | -0.2795 | 0.1005 |
| **Poly deg 2 (all feats) OLS** | 1232.7901 | 35.1111 | -13.8777 | 0.0361 |
| **Poly deg 2 + Ridge ($\alpha=100000$)** | **87.2657** | **9.3416** | **-0.0532** | **0.1560** |
| **Poly deg 2 + Lasso ($\alpha=2.2603$)** | 95.6243 | 9.7788 | -0.1540 | 0.0876 |

- **Multicollinearity & Regularization**:
  - Exact multiples verified: `Jitter:DDP = 3 * Jitter:RAP`, `Shimmer:DDA = 3 * Shimmer:APQ3` (VIF $= 1.0 \times 10^6$).
  - Lasso at CV $\alpha$ zeroes out 17 of 19 features (keeps only `age` and `PPE`).
- **Leakage Experiment Verification (`results/q2_leakage_experiment.csv`)**:
  - Random Forest on subject-wise split: Test $R^2 = -1.1656$, RMSE = 13.3956.
  - Random Forest on random-row split (leaky): Test $R^2 = 0.9774$, RMSE = 1.5812.
  - Demonstrates that random-row splits merely memorize patient identities rather than generalizing.
- **Artifacts Present**:
  - `figures/q2/03_gd_learning_rates_and_scaling.png`, `figures/q2/04_gd_final_loss.png`
  - `figures/q2/06_poly_degree_curves.png`, `figures/q2/07_ridge_lasso_cv_curves.png`
  - `figures/q2/08_coefficient_paths.png`, `figures/q2/09_coefficient_comparison.png`
  - `figures/q2/10_poly_regularisation_cv.png`, `figures/q2/11_model_comparison.png`, `figures/q2/13_leakage_experiment.png`.

---

## 3. Q3 Clustering and PCA — Netflix Titles
- **Data & Feature Engineering**:
  - 8,807 titles $\times$ 12 original columns.
  - Engineered into a 66-dimensional matrix (4 continuous variables z-scored: `release_year`, `added_year`, `movie_minutes`, `tv_seasons`; 62 unscaled 0/1 binary indicators for rating groups, top 40 genres with $\ge 50$ titles, and top 15 countries + unknown).
- **K-Means from Scratch Verification (`results/q3_wcss_required.csv`)**:
  - Required K range ($K=2..6$):
    - $K=2$: WCSS = 50,994.77, Drop = 23.61%, Silhouette = 0.2572
    - $K=3$: WCSS = 45,671.92, Drop = 10.44%, Silhouette = 0.2538
    - $K=4$: WCSS = 41,917.89, Drop = 8.22%, Silhouette = 0.1485 (elbow detector choice)
    - $K=5$: WCSS = 38,700.20, Drop = 7.68%, Silhouette = 0.1596
    - $K=6$: WCSS = 36,900.66, Drop = 4.65%, Silhouette = 0.1431
  - Validation against `sklearn.cluster.KMeans` (`results/q3_kmeans_validation_vs_sklearn.csv`):
    - $K=2$: WCSS match relative diff = 0.0000%, ARI = 1.0000
    - $K=3$: WCSS match relative diff = -0.0002%, ARI = 0.9998
    - $K=4$: WCSS match relative diff = 0.0000%, ARI = 1.0000
    - $K=5$: WCSS match relative diff = -0.0002%, ARI = 0.9985
    - $K=6$: WCSS match relative diff = +0.0000%, ARI = 0.9997
    - Lloyd's algorithm implementation confirmed mathematically equivalent to scikit-learn.
- **PCA from Scratch Verification**:
  - Max $|\Sigma v - \lambda v| = 6.66 \times 10^{-16}$ (exact eigenpair satisfaction).
  - Max $|V^T V - I| = 4.44 \times 10^{-16}$ (orthonormality satisfied).
  - Sum of eigenvalues ($7.580479$) equals trace of covariance matrix ($7.580479$).
  - Explained-variance ratios match `sklearn.decomposition.PCA` with `allclose=True` (atol $= 10^{-8}$).
  - Variance retained:
    - 2 components: 42.36% variance retained (reconstruction MSE = 0.0662)
    - 3 components: 54.10% variance retained (reconstruction MSE = 0.0527)
    - PCs needed for 80% variance: 12; for 90%: 23; for 95%: 34.
- **Artifacts Present**:
  - `figures/q3/01_init_sensitivity.png`, `figures/q3/02_kmeans_convergence.png`
  - `figures/q3/03_elbow_curve.png`, `figures/q3/03b_elbow_required_only.png`
  - `figures/q3/04_cluster_genre_heatmap.png`, `figures/q3/05_pca_scree_and_cumulative.png`
  - `figures/q3/06_pca_2d_scatter.png`, `figures/q3/07_pca_3d_scatter.png`, `figures/q3/08_pca_loadings.png`.

---

## 4. Visual Inspection of Key Figures
The following figures were programmatically inspected for resolution, content, and formatting:
1. `figures/q1/11_roc_curves.png`: Clean 4-panel figure (Normal vs rest, Suspect vs rest, Pathologic vs rest, Macro-average). Curves clearly separated with labeled AUCs and diagonal reference lines.
2. `figures/q1/08_confusion_counts.png`: Clean 4-panel count matrix across all 4 models, labels formatted properly.
3. `figures/q2/06_poly_degree_curves.png`: Log-scale plot of train, CV, and test RMSE showing divergence starting at degree 3, with vertical dashed line for CV-selected degree 1.
4. `figures/q2/07_ridge_lasso_cv_curves.png`: Two-panel log-scale alpha curves with shaded 1-SE bands and vertical markers for optimal $\alpha$ and 1-SE $\alpha$.
5. `figures/q2/11_model_comparison.png`: Dual-panel comparison showing train vs test RMSE bars and negative test $R^2$ values.
6. `figures/q3/03_elbow_curve.png`: Three-panel plot showing required K=2..6, extended K=1..10, and silhouette curves with clear numeric annotations.
7. `figures/q3/06_pca_2d_scatter.png`: Three-panel 2-D scatter showing projections coloured by Type (Movie vs TV Show), K-Means Cluster, and Rating Group, with centroid markers.
8. `figures/q3/05_pca_scree_and_cumulative.png`: Two-panel scree plot and cumulative variance curve with 80%, 90%, 95% threshold lines and 2-PC / 3-PC markers.

All figures are intact, properly labelled, and ready for publication in the final PDF report.

