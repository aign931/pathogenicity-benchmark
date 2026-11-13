# Reproducing the analyses

Run commands from the repository root. Python scripts read local inputs from data/ and write results to results/. To use other locations, set PB_DATA_DIR and PB_RESULTS_DIR. Fold assignments always come from data/folds/.

## Environments

| Model | Python | Main packages |
| --- | --- | --- |
| VARITY | 3.7.2 | XGBoost 0.90, scikit-learn 0.20.2, SHAP 0.34.0 |
| MutScore | 3.12.3 | scikit-learn 1.6.1, SHAP 0.47.2 |
| MVP | 3.7.16 | TensorFlow 1.15.0, Keras 2.2.4, SHAP 0.42.1 |

Package versions are listed separately in environments/varity/, environments/mutscore/ and environments/mvp/. Install each requirements.txt in its matching Python environment; do not combine them. Historical packages may require a compatible Linux distribution. Installation on a fresh machine has not been tested. Statistical comparisons use pROC 1.18.5; the R session information is in environments/statistics/.

In the examples below, VARITY_PY, MUTSCORE_PY and MVP_PY refer to the respective Python executables:

    export VARITY_PY=/path/to/varity/bin/python
    export MUTSCORE_PY=/path/to/mutscore/bin/python
    export MVP_PY=/path/to/mvp/bin/python

Data preparation and summaries use the MutScore environment.

## 1. Start from supplied outputs

Work with a copy of the supporting results. Each scenario directory contains models/, preprocess/, predictions/, shap/ and tables/. Place the scenario directories directly under results/, as described in package_guide.md. Processed feature tables are also needed to recalculate correlations and to colour the SHAP plots.

Recalculate the summaries for a scenario:

    "$MUTSCORE_PY" src/summarize_models.py --scenario full519
    "$MUTSCORE_PY" src/plot_supplement.py

These commands use the saved held-out predictions and SHAP values; they do not fit models. They compare row identities and feature orders before combining outputs. Generated summaries and figures replace the corresponding files in the working copy.

Repeat summarize_models.py for the other scenarios listed below. After all five summaries are present, calculate the statistical comparisons:

    Rscript src/compare_models.R results 5000

## 2. Start from processed inputs

Obtain the processed annotation tables under the source terms described in DATA_SOURCES.md and place them as follows:

| Local file | Contents |
| --- | --- |
| data/processed/canonical519.csv | Main cohort, before median imputation |
| data/processed/pooled526.csv | Main cohort plus seven PALB2 variants |
| data/processed/expert_panel.csv | Expert-panel subset, 279 variants |
| data/processed/canonical519_aligned.csv | Main cohort with metadata aligned for gene-held-out analysis |
| data/processed/vus.csv | Aligned VUS features, without binary labels |

The first four tables are required for the five labelled scenarios. The VUS table is needed only for the descriptive VUS analysis. Keep missing feature values: training scripts fit medians separately within each outer-training fold.

| Scenario | Data | Fold file |
| --- | --- | --- |
| full519 | canonical519.csv | main.csv |
| reduced519 | canonical519.csv; 19-feature panel | main.csv |
| palb2_526 | pooled526.csv | palb2.csv |
| expert_panel | expert_panel.csv | expert_panel.csv |
| leave_one_gene_out | canonical519_aligned.csv | leave_one_gene_out.csv |

Fit the three model families and summarise each scenario:

    for scenario in full519 reduced519 palb2_526 expert_panel leave_one_gene_out; do
      "$VARITY_PY" src/train_varity.py --scenario "$scenario"
      "$MUTSCORE_PY" src/train_mutscore.py --scenario "$scenario"
      "$MVP_PY" src/train_mvp.py --scenario "$scenario"
      "$MUTSCORE_PY" src/summarize_models.py --scenario "$scenario"
    done
    Rscript src/compare_models.R results 5000
    "$MUTSCORE_PY" src/plot_supplement.py

This fits 75 models with the study's fixed recipes. It does not repeat feature selection or hyperparameter tuning.

For VUS, use the five saved full519 models from each family:

    "$VARITY_PY" src/predict_vus.py --model VARITY
    "$MUTSCORE_PY" src/predict_vus.py --model MutScore
    "$MVP_PY" src/predict_vus.py --model MVP
    "$MUTSCORE_PY" src/summarize_vus.py

Once the five labelled scenarios, statistical comparisons and VUS summaries are available:

    "$MUTSCORE_PY" src/collect_results.py

The earlier 104-variant holdout comparison can be recalculated separately with src/compare_holdout.R, using data/processed/holdout_predictions.csv.

## 3. Start from source annotations

Full re-annotation requires the ClinVar release of 23 March 2025, VEP 113 with the GRCh38 cache, and dbNSFP 5.1a GRCh38 with its matching index. Obtain them from their providers under the applicable terms. These resources are not included in this repository.

Place the ClinVar source in data/raw/clinvar_20250323.vcf.gz. The historical five-gene candidate selection is documented in the archived dataset-construction notebook; that notebook is not a complete current reproduction workflow. The PALB2 and VUS candidate scripts are:

    "$MUTSCORE_PY" src/prepare_palb2.py
    "$MUTSCORE_PY" src/prepare_vus.py

Annotate each candidate VCF with the recorded field list in config/vep_fields.json. Supply a Docker image containing VEP 113 and the external resource directory:

    python src/annotate.py --input /path/to/candidates.vcf --output /path/to/annotated.txt --resources /path/to/vep_data --image YOUR_VEP_113_IMAGE

The resource directory must contain homo_sapiens/113_GRCh38/ and dbnsfp/dbNSFP5.1a_grch38.gz with its matching index.

Use the following names under data/raw/ for the main and PALB2 inputs:
- clinvar_bc_variants_for_vep.vcf
- clinvar_bc_annotated_canonical.txt
- clinvar_palb2_binary_breast_snv_for_vep.vcf
- clinvar_palb2_annotated_canonical.txt

Then prepare the labelled cohorts:

    "$MUTSCORE_PY" src/process_annotations.py
    "$MUTSCORE_PY" src/prepare_cohorts.py

The fixed fold files in data/folds/ remain the manuscript assignments. Additional fold and cohort tables written by prepare_cohorts.py are intermediate outputs.

For VUS, use data/raw/clinvar_5gene_vus_breast_snv_for_vep.vcf and data/raw/clinvar_5gene_vus_annotated_canonical.txt:

    "$MUTSCORE_PY" src/process_vus.py
    "$MUTSCORE_PY" src/align_vus.py

Expected cohort sizes are 519 main variants, 526 pooled variants, 279 expert-panel variants and 3,228 eligible VUS. Source-version changes may change eligibility and annotations; they would constitute a new analysis.

## Manuscript release

The corresponding author will create the approved release and arrange Zenodo archiving. Confirm the citation metadata and the actual deposited files before inserting the version-specific DOI in the manuscript. If code and supporting results are deposited separately, cite both records.
