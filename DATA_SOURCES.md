# Data sources and licensing

## Sources

| Resource | Version used | Role |
| --- | --- | --- |
| ClinVar | 23 March 2025, GRCh38 | Variant records and clinical classifications |
| Ensembl VEP | 113, GRCh38 cache | Consequence and transcript annotations |
| dbNSFP | 5.1a, GRCh38 | Functional prediction, conservation and population-frequency annotations |

The small files under data/folds/ contain variant identities, labels and study fold assignments. They do not contain the full annotation feature matrices.

Processed feature tables, database files and installed environments are not redistributed here. The original data providers and individual score sources retain their rights; the repository's software license does not apply to those data. Check the terms for the specific resource versions before redistributing annotation-bearing tables or supporting materials.

Sources:
- https://www.ncbi.nlm.nih.gov/clinvar/
- https://www.ncbi.nlm.nih.gov/home/about/policies/
- https://www.ensembl.org/info/docs/tools/vep/index.html
- https://www.dbnsfp.org/
- https://www.dbnsfp.org/license/

## Software

The existing MIT LICENSE covers repository-authored code. Dependencies and upstream source code retain their own licenses.

The MVP network in src/mvp_network.py is the study's archived adaptation of the network from https://github.com/ShenLab/missense. The VARITY-derived XGBoost and MutScore-derived random-forest scripts implement the fixed study recipes. They are not the published pretrained predictors.

## Use of the data

These are variant-level database records, not participant-level clinical records. The historical ClinVar classifications and descriptive VUS model scores should not be used as current clinical recommendations.
