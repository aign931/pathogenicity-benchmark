import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path(workspace_path('revision/04_analysis/loop05_corrected_common5fold/inputs'))
DATA_PATH = INPUT_DIR / "canonical519_cleaned_unimputed_aligned.csv"
OUTPUT_PATH = INPUT_DIR / "fold_assignments_leave_one_gene_out.csv"
SUMMARY_PATH = INPUT_DIR / "leave_one_gene_out_fold_label_counts.csv"


def main():
    df = pd.read_csv(str(DATA_PATH), dtype={"Uploaded_variation": str}).reset_index(drop=True)
    required = ["Uploaded_variation", "Gene", "source_gene", "Feature", "ClinicalSignificance", "label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing LOGO columns: {}".format(missing))
    genes = sorted(df["source_gene"].dropna().astype(str).unique().tolist())
    if genes != ["ATM", "BRCA1", "BRCA2", "CHEK2", "TP53"]:
        raise ValueError("Unexpected canonical519 gene set: {}".format(genes))
    gene_to_fold = {gene: index for index, gene in enumerate(genes)}
    folds = df[required].copy()
    folds.insert(0, "row_id", np.arange(len(df), dtype=int))
    folds["fold"] = folds["source_gene"].map(gene_to_fold).astype(int)
    if folds["Uploaded_variation"].duplicated().any():
        raise ValueError("LOGO Uploaded_variation IDs are not unique")
    folds.to_csv(str(OUTPUT_PATH), index=False)

    summary = folds.groupby(["fold", "source_gene", "label"]).size().rename("n").reset_index()
    summary.to_csv(str(SUMMARY_PATH), index=False)
    for fold, group in folds.groupby("fold"):
        if group["label"].nunique() != 2:
            raise ValueError("Held-out LOGO fold {} has one class".format(fold))
    print("saved {} rows across {} held-out genes".format(len(folds), len(genes)))


if __name__ == "__main__":
    main()
