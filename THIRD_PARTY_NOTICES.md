# Third-party source notices

The study evaluates adaptations of published approaches, not their native released predictors.

- MVP network source: ShenLab/missense, https://github.com/ShenLab/missense, local upstream commit 0dda09cf841374e6a1fdff7c1c170daf95ebaeaa. The archived study-specific models.py is retained to reproduce saved model weights. Its source hash is included in SHA256SUMS.csv. The repository MIT license does not relicense upstream material.
- VARITY-derived implementation: XGBoost recipe and positive-example weighting traced to the original VARITY framework. No full copy of that framework is distributed here. Historical parameter provenance is retained in model metadata within the evidence archive.
- MutScore-derived implementation: study-specific random forest using the recorded scikit-learn recipe, not a redistribution of the published pretrained model.
- XGBoost, scikit-learn, TensorFlow, Keras, SHAP, pROC and all other dependencies retain their own licenses. Installed dependencies are not bundled.

Historical notebooks retain their source content and citations under legacy/2025_holdout_analysis/. They are not validated manuscript reproduction instructions.
