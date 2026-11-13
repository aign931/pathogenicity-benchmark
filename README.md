# Pathogenicity Benchmark

Corrected analysis code supporting the manuscript on interpreting pathogenicity models for breast cancer-associated genes, prepared for submission to BMC Medical Genomics.

The current analysis uses study-specific VARITY-derived, MutScore-derived and MVP-derived implementations. It includes the canonical 519-variant cohort, reduced-feature, pooled PALB2, expert-panel and leave-one-gene-out analyses. Historical notebooks are under legacy/2025_holdout_analysis/ and DO NOT reproduce the submitted manuscript.

## Choose the reproducibility level

| Level | What it checks or runs | Requirements |
| --- | --- | --- |
| Verify supplied outputs | Hash integrity; optional saved-model prediction/SHAP checks and preprocessing audits | Repository for its own hashes; separate evidence archive for outputs; matching model environment for numerical checks |
| Rerun from processed inputs | Refit all three model families within the frozen folds, then recompute SHAP, statistics and figures | Processed annotation tables obtained under their source terms, supplied fold/feature files and three separate model environments |
| Full re-annotation | Rebuild annotations from the recorded ClinVar release with VEP 113, GRCh38 cache and dbNSFP 5.1a | External resources and their applicable permissions; substantially greater storage and runtime |

Start with:

    python verify_package.py

Then follow REPRODUCIBILITY.md. Passing hashes confirms file identity, not scientific validity or a fresh end-to-end rerun.

## Contents

- revision/04_analysis/: corrected source code, fold assignments, feature schema and compact audit records.
- thesis/models/missense/code/models.py: archived MVP network definition needed to load the saved weights.
- environments/VARITY/, environments/MutScore/, environments/MVP/: separate historical freezes and version records.
- environments/statistics/: R session information.
- provenance/: source and processed-input hashes, historical evidence inventory and migration record.
- audit/: scientific scope and repository validation.
- tools/: workspace preparation and repository maintenance.
- CITATION.cff: citation metadata; release version and DOI will be added when issued.
- DATA_PROVENANCE_AND_LICENSING.md: data provenance and distinct source terms.
- package_guide.md: repository versus separate archive contents.

## Repository and archive boundary

Ordinary Git contains code, small configuration/identity files, environment snapshots and audit summaries. Persisted models, preprocessing objects, bulk predictions/SHAP, feature-value matrices, figures, source databases and the complete handoff ZIP are not committed here.

The full local handoff inventory is preserved in provenance/handoff_20261020_SHA256SUMS.csv. It describes an existing author-held package, not a published downloadable deposit. Its path names and hashes do not grant redistribution rights. Data-bearing parts require source-specific licensing review before a public deposit.

## Release status

This is repository preparation for review. No manuscript release tag or Zenodo DOI has been issued by this update. The corresponding author will create the approved tagged release. Follow RELEASE_CHECKLIST.md to ensure that the DOI cited in the paper actually covers the stated files, including any separately deposited reproducibility archive.

## Scientific limits

The fixed recipes and feature panel were not selected within nested cross-validation. The original five-gene selection has historical limitations; PALB2 adds seven eligible variants. VUS results are descriptive only. SHAP output scales differ between model families. See audit/SCIENTIFIC_AUDIT.md.

## Citation and licensing

Use CITATION.cff and, once released, the version-specific DOI. The existing MIT LICENSE applies to repository-authored code; it does not override third-party code or annotation-data terms. See DATA_PROVENANCE_AND_LICENSING.md and THIRD_PARTY_NOTICES.md.
