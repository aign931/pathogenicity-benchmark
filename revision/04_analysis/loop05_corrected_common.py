from __future__ import print_function
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REVISION = Path(workspace_path('revision'))
THESIS = Path(workspace_path('thesis'))
BASE_DIR = REVISION / "04_analysis/loop05_corrected_common5fold"
INPUT_DIR = BASE_DIR / "inputs"

FULL_FEATURES = [
    "AllofUs_ALL_AF",
    "AlphaMissense_score",
    "BayesDel_addAF_score",
    "BayesDel_noAF_score",
    "CADD_raw",
    "ClinPred_score",
    "DANN_score",
    "DEOGEN2_score",
    "ESM1b_score",
    "Eigen-PC-raw_coding",
    "Eigen-raw_coding",
    "GERP++_RS",
    "GERP_91_mammals",
    "M-CAP_score",
    "MPC_score",
    "MVP_score",
    "MetaLR_score",
    "MetaRNN_score",
    "MetaSVM_score",
    "MutFormer_score",
    "MutScore_score",
    "MutationAssessor_score",
    "MutationTaster_score",
    "PROVEAN_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "PrimateAI_score",
    "REVEL_score",
    "RegeneronME_ALL_AF",
    "SIFT4G_score",
    "SIFT_score",
    "VEST4_score",
    "bStatistic",
    "dbNSFP_POPMAX_AF",
    "fathmm-XF_coding_score",
    "gMVP_score",
    "gnomAD4.1_joint_AF",
    "phastCons100way_vertebrate",
    "phastCons17way_primate",
    "phastCons470way_mammalian",
    "phyloP100way_vertebrate",
    "phyloP17way_primate",
    "phyloP470way_mammalian",
]

# Predefined, reviewer-motivated stress-test panel. It retains population-frequency annotations,
# conservation measures, and six long-standing single/base functional scores,
# while excluding high-level ensemble, meta-predictor, and deep-model outputs.
BASE_ANNOTATION_FEATURES = [
    "AllofUs_ALL_AF",
    "GERP++_RS",
    "GERP_91_mammals",
    "MutationAssessor_score",
    "PROVEAN_score",
    "Polyphen2_HDIV_score",
    "Polyphen2_HVAR_score",
    "RegeneronME_ALL_AF",
    "SIFT4G_score",
    "SIFT_score",
    "bStatistic",
    "dbNSFP_POPMAX_AF",
    "gnomAD4.1_joint_AF",
    "phastCons100way_vertebrate",
    "phastCons17way_primate",
    "phastCons470way_mammalian",
    "phyloP100way_vertebrate",
    "phyloP17way_primate",
    "phyloP470way_mammalian",
]

SCENARIOS = {
    "full519": {
        "data": THESIS / "data/benchmark_dataset_canonical_cleaned.csv",
        "folds": INPUT_DIR / "fold_assignments_519_original_copy.csv",
        "panel": "full",
        "description": "Canonical 519 variants; fixed 43-feature panel; fold-local median imputation.",
    },
    "reduced519": {
        "data": THESIS / "data/benchmark_dataset_canonical_cleaned.csv",
        "folds": INPUT_DIR / "fold_assignments_519_original_copy.csv",
        "panel": "base_annotation",
        "description": "Canonical 519 variants; predefined 19-feature base-annotation panel.",
    },
    "palb2_526": {
        "data": INPUT_DIR / "canonical526_cleaned_unimputed.csv",
        "folds": INPUT_DIR / "fold_assignments_526_with_palb2.csv",
        "panel": "full",
        "description": "Primary 519 plus seven PALB2 variants; original folds preserved for the 519 rows.",
    },
    "expert_panel": {
        "data": INPUT_DIR / "expert_panel_only_519_cleaned_unimputed.csv",
        "folds": INPUT_DIR / "expert_panel_only_5fold_assignments.csv",
        "panel": "full",
        "description": "ClinVar expert-panel-only label sensitivity cohort.",
    },
    "leave_one_gene_out": {
        "data": INPUT_DIR / "canonical519_cleaned_unimputed_aligned.csv",
        "folds": INPUT_DIR / "fold_assignments_leave_one_gene_out.csv",
        "panel": "full",
        "description": "Exploratory leave-one-gene-out generalisation analysis.",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_panel(name):
    if name == "full":
        return list(FULL_FEATURES)
    if name == "base_annotation":
        return list(BASE_ANNOTATION_FEATURES)
    raise ValueError("Unknown feature panel: {}".format(name))


def model_features(model_name, panel_name):
    features = feature_panel(panel_name)
    if panel_name == "full" and model_name == "MutScore":
        features = [f for f in features if f != "MutScore_score"]
    if panel_name == "full" and model_name == "MVP":
        features = [f for f in features if f != "MVP_score"]
    return features


def common_features(panel_name):
    models = ["VARITY", "MutScore", "MVP"]
    common = set(model_features(models[0], panel_name))
    for model in models[1:]:
        common &= set(model_features(model, panel_name))
    return [f for f in feature_panel(panel_name) if f in common]


def scenario_config(name):
    if name not in SCENARIOS:
        raise ValueError("Unknown scenario: {}".format(name))
    return dict(SCENARIOS[name])


def scenario_dir(name):
    return BASE_DIR / name


def ensure_scenario_dir(name):
    root = scenario_dir(name)
    for subdir in ["logs", "models", "predictions", "preprocess", "shap", "tables", "figures"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def read_scenario(name):
    config = scenario_config(name)
    data_path = config["data"]
    fold_path = config["folds"]
    if not data_path.exists():
        raise IOError("Scenario data missing: {}".format(data_path))
    if not fold_path.exists():
        raise IOError("Scenario fold file missing: {}".format(fold_path))

    df = pd.read_csv(str(data_path), dtype={"Uploaded_variation": str}).reset_index(drop=True)
    folds = pd.read_csv(str(fold_path), dtype={"Uploaded_variation": str})
    if "row_id" not in folds.columns and "expert_row_id" in folds.columns:
        folds = folds.rename(columns={"expert_row_id": "row_id"})
    folds = folds.sort_values("row_id").reset_index(drop=True)
    if len(df) != len(folds):
        raise ValueError("Data/fold row mismatch: {} vs {}".format(len(df), len(folds)))
    if not np.array_equal(folds["row_id"].astype(int).values, np.arange(len(df))):
        raise ValueError("Fold row_id is not an exact 0..N-1 sequence")
    if "Uploaded_variation" in folds.columns:
        data_ids = df["Uploaded_variation"].astype(str).values
        fold_ids = folds["Uploaded_variation"].astype(str).values
        if not np.array_equal(data_ids, fold_ids):
            raise ValueError("Fold IDs do not align with scenario data order")
    if "label" not in df.columns or "label" not in folds.columns:
        raise ValueError("Both data and fold files require label")
    if not np.array_equal(df["label"].astype(int).values, folds["label"].astype(int).values):
        raise ValueError("Fold labels do not align with scenario data")
    return df, folds, config


def numeric_frame(df, features):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError("Missing features: {}".format(missing))
    out = df.loc[:, features].copy()
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fold_local_median_impute(train_df, valid_df, features, audit_path):
    train = numeric_frame(train_df, features)
    valid = numeric_frame(valid_df, features)
    medians = train.median(axis=0, skipna=True)
    all_missing = medians[medians.isna()].index.tolist()
    if all_missing:
        raise ValueError("All-missing training features: {}".format(all_missing))

    audit = pd.DataFrame(
        {
            "feature": features,
            "train_median": [float(medians[f]) for f in features],
            "train_missing_n": [int(train[f].isna().sum()) for f in features],
            "valid_missing_n": [int(valid[f].isna().sum()) for f in features],
        }
    )
    audit.to_csv(str(audit_path), index=False)
    return train.fillna(medians), valid.fillna(medians), medians


def write_run_metadata(root, model_name, scenario_name, extra):
    config = scenario_config(scenario_name)
    payload = {
        "model": model_name,
        "scenario": scenario_name,
        "scenario_description": config["description"],
        "data_path": str(config["data"]),
        "data_sha256": sha256(config["data"]),
        "fold_path": str(config["folds"]),
        "fold_sha256": sha256(config["folds"]),
        "feature_panel": config["panel"],
        "model_features": model_features(model_name, config["panel"]),
        "common_features": common_features(config["panel"]),
        "preprocessing": "Median values fitted on each outer-training fold only.",
    }
    payload.update(extra)
    path = root / "logs/{}_run_metadata.json".format(model_name.lower())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path

