import pandas as pd
import numpy as np
from models import CNN_Model

# ------------CONFIG------------
DATA_PATH = "../data/mvp_input_data_cleaned.HIS.csv"
OUTPUT_MODEL_PATH = "../models/trained_mvp_model_original_data.h5"

# ------------LOAD DATA------------

df = pd.read_csv(DATA_PATH)

exclude_cols = {'var_id', 'aaref', 'aaalt', 'target', 'Ensembl_transcriptid',
                'ref', 'alt', 'category',
                'source', 'INFO', 'disease', 'genename',
                '#chr', 'pos(1-based)',  'hg19_chr', 'hg19_pos(1-based)',
                'CADD_phred', '1000Gp3_AF', 'ExAC_AF', 'gnomad',
                'RVIS', 'mis_badness', 'MPC', 'REVEL', 'domino'}

# ------------TRAIN MODEL------------
model = CNN_Model(
    weights_path=None,
    input_shape=(54, 1, 1),
    exclude_cols=exclude_cols,
    train_flag=True,
    verbose=2,
    nb_epoch=40,
    batch_size=64,
    name='res_HIS_54features',
    fname=DATA_PATH,
    f_out="../data/output_data_mode5.csv"
)

model.train(sub_sample=3)
model.model.save_weights(OUTPUT_MODEL_PATH)
print("Model training complete. Weights saved to:", OUTPUT_MODEL_PATH)