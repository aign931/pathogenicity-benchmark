# Pathogenicity Benchmark — VARITY, MutScore, MVP

This repository accompanies the study **"Interpreting Pathogenicity Prediction Models for Breast Cancer-Associated Genes"**.

## Structure
- `models/`: model training, evaluation, and SHAP analysis scripts.
- `data/`: The repository includes variant-level labels indicating genomic location, gene, and binary ClinVar-derived classification. Full feature annotations are excluded due to licensing restrictions.
- `notebooks/`: exploratory and correlation analyses.
- `env/`: environment specifications for reproducibility.

## Data
No real data are included.  

## Data Sources
- **ClinVar**: https://www.ncbi.nlm.nih.gov/clinvar/
- **dbNSFP v5.1a**: https://www.dbnsfp.org/

## Tools Used
- [Ensembl VEP](https://www.ensembl.org/info/docs/tools/vep/index.html) with Docker
- dbNSFP plugin for VEP
- Python (v3.12.3)
- PowerShell (on Windows)
- Ubuntu 24.04 via WSL2
- R (v4.3.3)
- Conda (Miniconda3)
- GitHub repositories of each model:
   - [VARITY](https://github.com/joewuca/varity).
   - [MutScore](https://github.com/mquinodo/MutScore).
   - [MVP](https://github.com/ShenLab/missense).


## Notes
1. Data is accessible via link in 'Data Scources'.
2. Please refer to the original models for virtual environment configuration.

## License
MIT License

---