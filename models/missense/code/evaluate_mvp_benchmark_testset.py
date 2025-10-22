# evaluate_mvp_benchmark_testset.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from models import CNN_Model

# ------------CONFIG------------
TEST_DATA_PATH = "../data/test_set_scaled.csv"
MODEL_WEIGHTS_PATH = "../models/trained_mvp_model_benchmark_training_set.h5"
OUTPUT_PREDICTIONS_PATH = "../data/mvp_test_predictions.csv"
ROC_PLOT_PATH = "../results/mvp_test_set_roc_curve.png"

# ------------LOAD TEST DATA into model.X_pred & model.y------------
df_test = pd.read_csv(TEST_DATA_PATH)

exclude_cols = {
    'Uploaded_variation', 'level_1', 'Location', 'Allele', 'Gene',
    'Feature', 'Feature_type', 'Consequence',
    'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids',
    'Codons', 'Existing_variation', 'IMPACT', 'DISTANCE',
    'STRAND', 'FLAGS', 'ClinicalSignificance', 'target', 'MVP_score'
}

# ------------INITIALIZE MODEL FOR INFERENCE------------
model = CNN_Model(
    weights_path=MODEL_WEIGHTS_PATH,
    input_shape=(42, 1, 1),
    exclude_cols=exclude_cols,
    train_flag=False,
    verbose=0,
    name='res_HIS_benchmark_training_set',
    fname=TEST_DATA_PATH,
    f_out="../data/temp_output_eval.csv"  # not used in inference
)

# --- Load data and weights ---
model._load_data(sub_sample=False) 
model._init_model(verbose=False)

# ------------PREDICT------------
cnn_prob = model.model.predict(model.X_pred, batch_size=16)
y_true = model.y

# ------------SAVE PREDICTIONS------------
df_out = pd.DataFrame({
    "prob_pathogenic": cnn_prob[:, 0],
    "label": y_true
})
df_out.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
print("Predictions saved to:", OUTPUT_PREDICTIONS_PATH)

# ------------PLOT ROC------------
fpr, tpr, _ = roc_curve(y_true, cnn_prob[:, 0])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - MVP on Benchmark Test Set")
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(ROC_PLOT_PATH, dpi=300)
plt.close()
print("ROC curve saved to:", ROC_PLOT_PATH)
