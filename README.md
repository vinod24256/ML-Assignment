# ML Assignment — IIIT Sri City

| Notebook | Question | Dataset | Models |
|---|---|---|---|
| `notebooks/q1_classification.ipynb` | Q1 (+ bonus) | UCI Cardiotocography (id 193), target `NSP` | KNN+CV, Decision Tree, SVM-RBF, 1D-CNN (PyTorch) |
| `notebooks/q2_regression.ipynb` | Q2 | UCI Parkinsons Telemonitoring (id 189), target `total_UPDRS` | Linear regression with GD (scratch), Polynomial, Ridge, Lasso |
| `notebooks/q3_kmeans_pca.ipynb` | Q3a + Q3b | Kaggle Netflix Titles | K-Means (scratch), PCA (scratch) |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Tip (no GPU): install CPU-only PyTorch first, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then the rest.

## Data
* **Q1, Q2** — downloaded automatically via `ucimlrepo` (needs internet). Offline fallback: put `CTG.xls` and `parkinsons_updrs.data` (from the UCI zip files) into `data/`.
* **Q3** — download `netflix_titles.csv` from https://www.kaggle.com/datasets/shivamb/netflix-shows and put it in `data/`. (If the file is absent the notebook tries `kagglehub`.)

## Run
Open each notebook and **Kernel → Restart & Run All**, or from the `notebooks/` folder:
```bash
jupyter nbconvert --to notebook --execute --inplace q1_classification.ipynb
jupyter nbconvert --to notebook --execute --inplace q2_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace q3_kmeans_pca.ipynb
```
Approximate run time on a laptop CPU: Q1 ≈ 4–8 min (grid searches + 5 CNN seeds), Q2 ≈ 2–3 min, Q3 ≈ 2 min.

## Outputs
* `figures/q1|q2|q3/*.png` — every figure used in the report (numbered in reading order)
* `results/*.csv|json` — every number used in the report (metrics, best hyper-parameters, WCSS, PCA variance, cluster profiles…)

All randomness is seeded (`SEED = 42`), so re-running reproduces the same numbers (the CNN may differ in the last decimals on GPU).
