"""Locations of local data and generated analysis results."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('PB_DATA_DIR', str(PROJECT_DIR / 'data'))).expanduser().resolve()
RESULTS_DIR = Path(os.environ.get('PB_RESULTS_DIR', str(PROJECT_DIR / 'results'))).expanduser().resolve()
FOLDS_DIR = PROJECT_DIR / 'data' / 'folds'
CONFIG_DIR = PROJECT_DIR / 'config'
SOURCE_DIR = PROJECT_DIR / 'src'
