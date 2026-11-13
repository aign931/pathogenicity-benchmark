# Pathogenicity Benchmark

Code for the study of pathogenicity prediction and SHAP explanations in breast cancer-associated genes.

The analyses compare study-specific implementations derived from VARITY, MutScore and MVP. They cover the main cohort of 519 variants, a reduced feature panel, pooled PALB2 sensitivity analysis, an expert-panel subset and leave-one-gene-out evaluation. VUS scores are reported separately as descriptive results.

## Project structure

| Directory | Contents |
| --- | --- |
| src/ | Data preparation, model fitting, SHAP analysis, statistics and figures |
| data/folds/ | Fixed fold assignments used in the manuscript |
| config/ | Feature order and VEP annotation fields |
| environments/ | Separate VARITY, MutScore and MVP package versions, plus R information |
| legacy/2025_holdout_analysis/ | Earlier notebooks and scripts, not used to reproduce the submitted manuscript |

Processed annotation tables and generated results are stored locally in data/processed/ and results/. They are not committed to this repository.

## Running the analyses

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for three starting points:

1. Recalculate summaries and figures from supplied model outputs.
2. Fit the models using processed inputs and the fixed folds.
3. Rebuild annotations using external ClinVar, VEP and dbNSFP resources.

The three model environments must remain separate. The examples assume Linux or WSL.

## Data and citation

[DATA_SOURCES.md](DATA_SOURCES.md) describes source versions and licensing. [package_guide.md](package_guide.md) describes the supporting material held outside Git.

Citation details are in CITATION.cff. A version-specific Zenodo DOI will be added after the manuscript release is approved.

## Interpretation

These are adaptations for the study's benchmark, not the original published pretrained tools. Model recipes and the feature panel were fixed before the shared-fold analyses; this is not nested model-development cross-validation. SHAP output units differ between model families, and the VUS analysis is not clinical validation.
