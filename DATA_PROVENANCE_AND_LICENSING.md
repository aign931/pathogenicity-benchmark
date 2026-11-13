# Data provenance and licensing

## Scope
The repository MIT license covers repository-authored software. It is not a blanket license for third-party code, source databases, variant annotations or every file in a separately supplied archive.

## Data inventory
| Material | Recorded provenance | Distribution in this repository |
| --- | --- | --- |
| Variant identifiers, labels and fold assignments | ClinVar-derived historical cohort; release 2025-03-23 | Small identity/fold files included |
| Processed feature matrices | VEP 113, GRCh38, dbNSFP 5.1a and component scores | Not included; paths, sizes and hashes in provenance/processed_inputs.csv |
| ClinVar source VCF | NCBI ClinVar, release 2025-03-23 | Not included; historical hash documented |
| VEP cache and container | Ensembl VEP 113; recorded container digest | Not redistributed |
| dbNSFP data and tabix index | dbNSFP 5.1a GRCh38 | Not redistributed |
| Fitted models, preprocessors, predictions and SHAP arrays | Corrected July analyses; October numerical audits | Separate evidence archive, not Git history |

These are variant-level public database records, not participant-level clinical records. Historical classifications must not be treated as current clinical recommendations.

## Source terms
Consult the terms for the exact source versions and components used; attribution and a small file size do not remove source restrictions. The current dbNSFP license page describes academic/non-commercial restrictions and explicitly requires compliance with individual component terms. Do not infer that a derived feature table is covered by MIT or that every score may be redistributed.

- dbNSFP license: https://www.dbnsfp.org/license/
- dbNSFP downloads and version information: https://www.dbnsfp.org/download/
- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- NCBI policies: https://www.ncbi.nlm.nih.gov/home/about/policies/
- Ensembl VEP: https://www.ensembl.org/info/docs/tools/vep/index.html

The repository does not assert that the present website terms retrospectively resolve every component's 5.1a redistribution permission. Before publishing an evidence deposit, review the recorded version documentation and component terms, including any processed feature values, annotation-bearing plots and model-related restrictions. If a material cannot be redistributed, document its omission, expected checksum and acquisition/reconstruction route.

## Audit provenance
provenance/external_source_provenance.csv preserves the historical source paths, sizes and hashes. Some large-resource hashes were inherited from previously verified manifests rather than recomputed in November. provenance/handoff_20261020_SHA256SUMS.csv identifies the author-held package and does not grant rights in its contents.
