# Scientific evidence and execution scope

The corrected July analyses comprise five scenarios (full519, reduced519, palb2_526, expert_panel, leave_one_gene_out), three model families and five folds: 75 fitted models. Superseded Loop03 surrogate and Loop04 model results are excluded. The retained original fold CSV is provenance only.

October 19 verification reloaded 15 primary models and recomputed their predictions and held-out SHAP values in the original separate environments. Maximum SHAP discrepancies were 1.18e-7 (VARITY), approximately 1e-16 (MutScore), and 5e-15 (MVP). All 75 training-median records and 25 MVP scaler/background sets were checked against training rows.

The 19-feature panel is predefined and reviewer-motivated, not prospectively registered. Recipes and feature selection were not repeated within outer folds. The implementations are study-specific adaptations, not unmodified published predictors. VUS scores are descriptive, not clinical validation. Model outputs have different SHAP units.

The November repository migration changes paths and documentation, not model recipes or scientific results. Tests performed during this migration are recorded in audit/repository_validation.json. It is not a new full training or annotation run.
