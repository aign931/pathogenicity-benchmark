# Reproducibility

## 1. Verify supplied files and outputs

Repository-only integrity verification requires Python 3.7 or newer and the standard library:

    python verify_package.py

To verify an extracted complete handoff using its existing manifest:

    python verify_package.py --root /path/to/handoff_package

To check that same package against the inventory tracked here:

    python verify_package.py --root /path/to/handoff_package --manifest provenance/handoff_20261020_SHA256SUMS.csv

Hash checks alone do not reload models, recalculate statistics or validate conclusions. The complete handoff is currently author-held; no public archive URL is claimed.

Create a NEW workspace to run numerical checks without changing supplied evidence (workspace preparation requires Python 3.8+):

    python tools/prepare_workspace.py --destination /path/to/check-work --archive-root /path/to/handoff_package --include-outputs
    cd /path/to/check-work
    export PB_WORKSPACE="$PWD"

Set VARITY_PY, MUTSCORE_PY and MVP_PY to the Python executables of the three separately reconstructed environments. The examples below are Bash commands for Linux/WSL. For example, export VARITY_PY=/path/to/varity/bin/python. In PowerShell set $env:PB_WORKSPACE to the full workspace path.

    "$VARITY_PY" revision/04_analysis/round2_20261019/verify_runtime.py VARITY
    "$MUTSCORE_PY" revision/04_analysis/round2_20261019/verify_runtime.py MutScore
    "$MVP_PY" revision/04_analysis/round2_20261019/verify_runtime.py MVP
    "$MUTSCORE_PY" revision/04_analysis/round2_20261019/audit_folds.py

The first three commands recompute predictions/SHAP for the 15 primary models. The last audits all 75 model-fold preprocessing records and existing reload reports; it does not retrain or freshly reload all 75 models. Runtime audits overwrite their JSON reports only in the working copy.

Additional per-scenario saved-model validation is provided by loop05_validate_varity.py, loop05_validate_mutscore.py and loop05_validate_mvp.py. Check their --help for arguments.

## 2. Rerun from processed inputs

Use the exact processed tables listed in provenance/processed_inputs.csv. These are deliberately not redistributed in ordinary Git because they contain third-party annotation scores. Obtain them under applicable terms or reconstruct them using the external annotation workflow. The expected file hashes are recorded; the preparation utility rejects missing or altered inputs.

    python tools/prepare_workspace.py --destination /path/to/rerun-work --archive-root /path/to/handoff_package
    cd /path/to/rerun-work
    export PB_WORKSPACE="$PWD"

Keep the three environments separate. The environment records are forensic snapshots, not proof that every historical package remains installable on a new platform. Review local-file references in freezes and use a compatible Linux platform. Fresh installation of all three environments has not been validated.

| Environment | Python | Critical packages |
| --- | --- | --- |
| VARITY | 3.7.2 | numpy 1.16.0; pandas 0.24.0; XGBoost 0.90; scikit-learn 0.20.2; SHAP 0.34.0 |
| MutScore | 3.12.3 | numpy 2.2.4; pandas 2.2.3; scikit-learn 1.6.1; SHAP 0.47.2 |
| MVP | 3.7.16 | TensorFlow 1.15.0; Keras 2.2.4; numpy 1.21.6; scikit-learn 1.0.2; SHAP 0.42.1 |

The full pip freezes are in the corresponding environments directories. Do not combine them into one requirements file. R uses the recorded pROC 1.18.5 environment.

Run all scenarios with frozen fold assignments:

    for scenario in full519 reduced519 palb2_526 expert_panel leave_one_gene_out; do
      "$VARITY_PY" revision/04_analysis/loop05_run_varity.py --scenario "$scenario"
      "$MUTSCORE_PY" revision/04_analysis/loop05_run_mutscore.py --scenario "$scenario"
      "$MVP_PY" revision/04_analysis/loop05_run_mvp.py --scenario "$scenario"
      "$MUTSCORE_PY" revision/04_analysis/loop05_aggregate_corrected.py --scenario "$scenario"
    done
    Rscript revision/04_analysis/loop05_delong_bootstrap.R "$PB_WORKSPACE/revision/04_analysis/loop05_corrected_common5fold" 5000

This fits 75 models. It does not repeat historical hyperparameter selection or register a new feature panel. Numerical differences across platforms or library builds should be investigated rather than silently replacing archived values.

For the two additional supplementary figures, use the MutScore environment (the recorded plotting SHAP version is 0.47.2). Plotting consumes saved explanations and does not change their explainer provenance:

    "$MUTSCORE_PY" revision/04_analysis/round2_20261019/make_figures.py

The VUS scripts under revision/04_analysis/loop06_vus/ consume separate processed VUS inputs and the primary fitted models. Use --include-outputs for the existing VUS evidence, or reconstruct it under level 3. The four-input minimal training workspace does not include VUS annotations or the original holdout predictions. Once VUS summaries are also available (as in a workspace prepared with --include-outputs), consolidate the complete evidence with:

    "$MUTSCORE_PY" revision/04_analysis/loop07_consolidate_evidence.py

## 3. Full re-annotation

This requires external ClinVar, VEP and dbNSFP resources. The repository alone is insufficient. Historical source versions are:

- ClinVar release 2025-03-23, GRCh38, with recorded SHA256.
- VEP 113 and homo_sapiens/113_GRCh38 cache.
- dbNSFP 5.1a GRCh38 and its matching tabix index.
- Pinned container digest in provenance/vep_environment.txt; do not substitute today's latest tag.

Download resources from their official providers under the applicable terms. Place them in a separate workspace at the historical relative locations described in provenance/external_source_provenance.csv. Set PB_WORKSPACE to that workspace. Populate vep_data with the cache and dbNSFP resources; installed environment binaries are never part of the archive.

The retained historical dataset-construction notebook documents the original five-gene candidate selection, but is NOT a validated end-to-end reproduction entry point. The executable PALB2 and exact-VUS reconstruction scripts are current:

    python revision/04_analysis/build_palb2_candidate_vcf.py
    python revision/04_analysis/loop06_vus/01_build_candidate_vcfs.py

Use tools/run_vep.py for a supplied candidate VCF. It uses the recorded fields and pinned image:

    python tools/run_vep.py --input /absolute/path/candidates.vcf --output /absolute/path/annotated.txt --resources /absolute/path/vep_data

After annotation, run process_palb2_vep_output.py for the canonical/PALB2 processing audit, and 03_process_vus_vep.py then 05_validate_vus_outputs.py and 06_build_model_aligned_vus.py for VUS processing/validation. Expected filenames remain documented in the scripts. Then run build_loop05_inputs.py and loop05_make_logo_folds.py before level 2 training.

Check canonical 519, pooled PALB2 526, expert-panel 279 and eligible VUS 3,228 counts and row identities against the recorded audits. Do not replace historical source versions with current releases and describe the result as the same dataset.

A new full re-annotation and clean environment installation were not performed for this repository migration.

## Paths and provenance

Current Python scripts resolve historical D:/ and /mnt/d/ locations through workflow_paths.py and PB_WORKSPACE. The default workspace is the repository root. Use the preparation utility to keep generated material out of Git and protect the original archive. Path-only migration hashes are recorded in provenance/source_migration.json. Older path strings inside saved audit reports describe the historical machine and are not current installation requirements.
