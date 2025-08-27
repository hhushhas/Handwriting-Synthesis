import pickle

import os
from pathlib import Path

# Auto-detect project base directory or allow override via env var
BASE_DIR = os.getenv("HW_BASE_DIR") or str(Path(__file__).resolve().parent.parent)
with open(f"{BASE_DIR}/ctoi.txt", "rb") as file:
    enc_dict = pickle.load(file)

# Dataset directory (override with HW_DATA_DIR)
SRC_DIR = os.getenv("HW_DATA_DIR") or f"{BASE_DIR}/IAM"

# Output and log directories
OUT_DIR = f"{BASE_DIR}/src/out"
RUNS_DIR = f"{BASE_DIR}/src/runs"

# Ensure directories exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/inference", exist_ok=True)

# Batch size (override with HW_BATCH_SIZE)
BATCH_SIZE = int(os.getenv("HW_BATCH_SIZE", "64"))
NUM_TOKENS = len(enc_dict)
EMBEDDING_SIZE = 128
NUM_LAYERS = 4  # Ideally 4
PADDING_IDX = 0
Z_LEN = 128  # Z_LEN should be equal to EMBEDDING_SIZE
CHUNKS = 8
CBN_MLP_DIM = 512
RELEVANCE_FACTOR = 1
LEARNING_RATE = 2e-4
BETAS = (0, 0.999)
