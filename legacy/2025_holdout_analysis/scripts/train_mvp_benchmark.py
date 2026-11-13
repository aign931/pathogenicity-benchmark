# train_mvp_benchmark.py

import pandas as pd
import numpy as np
from models import CNN_Model

# ------------CONFIG------------
DATA_PATH = "../data/training_set_scaled.csv"
OUTPUT_MODEL_PATH = "../models/trained_mvp_model_benchmark_training_set.h5"

# ------------LOAD DATA------------
df = pd.read_csv(DATA_PATH)

exclude_cols = {
    'Uploaded_variation', 'level_1', 'Location', 'Allele', 'Gene',
    'Feature', 'Feature_type', 'Consequence',
    'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids',
    'Codons', 'Existing_variation', 'IMPACT', 'DISTANCE',
    'STRAND', 'FLAGS', 'ClinicalSignificance', 'target', 'MVP_score'
}

# ------------TRAIN MODEL------------
model = CNN_Model(
    weights_path=None,
    input_shape=(42, 1, 1),
    exclude_cols=exclude_cols,
    train_flag=True,
    verbose=2,
    nb_epoch=20,
    batch_size=16,
    name='res_HIS_benchmark_training_set',
    fname=DATA_PATH,
    f_out="../data/output_benchmark_mode5.csv"
)

model.train(sub_sample=3)

model.model.save_weights(OUTPUT_MODEL_PATH)
print("Model training complete. Weights saved to:", OUTPUT_MODEL_PATH)
