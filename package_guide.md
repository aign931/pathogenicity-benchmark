# Repository and reproducibility archive guide

## Ordinary Git
This repository contains corrected Python/R code, the MVP architecture, input schemas, feature orders, fold assignments, compact audits, three separate environment freezes, citation metadata, provenance statements, this guide, SHA256SUMS.csv and verify_package.py.

## Separate author-held evidence archive
The October handoff contains 835 manifest-listed payload files, including 75 persisted models, preprocessing objects, predictions, SHAP matrices, statistics and manuscript sources. Its ZIP is 249,664,612 bytes (compressed); this is distinct from extracted size. Its ZIP SHA256 is cce98739343c8eea4e65d936829a799d1f38925b1337391c7845c11b2aac48bb.

The exact inventory is provenance/handoff_20261020_SHA256SUMS.csv. This inventory does not mean the archive has been published. Do not upload the entire ZIP to Git history. A public deposit must be assembled after reviewing data-bearing files against source permissions; the current handoff is not automatically cleared for unrestricted redistribution.

## Integrity
The root manifest covers repository files except itself. The historical handoff manifest covers the separate unchanged package. Run verify_package.py with the correct root and manifest. Missing external evidence is a failed check, never a silently skipped success.

## Release
Use RELEASE_CHECKLIST.md. The corresponding author retains the planned tag/release step. A source-only Zenodo DOI cannot substantiate a manuscript statement that the same record includes unuploaded models and SHAP outputs.
