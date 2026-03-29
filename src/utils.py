import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Directories created: processed/, models/")


def load_csv(filename, folder=RAW_DIR):
    path = os.path.join(folder, filename)
    return pd.read_csv(path)


def save_csv(df, filename, folder=PROCESSED_DIR):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    df.to_csv(path, index=False)
    print(f"Saved file: {path}")
    return path