# Changes Log

No scientific, algorithmic, hyper-parameter, split, or seed changes were made to the notebooks.
All algorithms, hyper-parameter grids, evaluation metrics, and split strategies are preserved exactly as provided.

| Timestamp | Notebook / File | Cell / Location | Change Description | Justification |
|---|---|---|---|---|
| 2026-11-02 14:51 | `notebooks/q2_regression.ipynb` | Cell 15 (Section 6.1: One-feature visual demo) | Added `ax.set_ylim(ytr.min()-5, ytr.max()+5)` for each panel in Figure 05 | Presentation-only fix: prevents Degree 15 polynomial edge oscillations from blowing up the y-axis scale, preserving visual inspection of scatter points and fits |

