import os
import pandas as pd

def _read_data_with_candidates(candidates):
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    # fallback: try to find a matching file in data/ by keywords
    data_dir = "data"
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if "students" in fname.lower() and "performance" in fname.lower():
                return pd.read_csv(os.path.join(data_dir, fname))
    raise FileNotFoundError(f"Could not find any of: {candidates}")

def load_standard_data():
    candidates = [
        "data/Students_Performance_Dataset.csv",
        "data/Students Performance Dataset.csv",
        "data/Students-Performance-Dataset.csv",
    ]
    df = _read_data_with_candidates(candidates)
    allowed_columns = [
        'Attendance (%)',
        'Midterm_Score',
        'Final_Score',
        'Projects_Score',
        'Study_Hours_per_Week'
    ]
    return df[allowed_columns]

def load_biased_data():
    candidates = [
        "data/Students_Performance_Dataset.csv",
        "data/Students Performance Dataset.csv",
        "data/Students-Performance-Dataset.csv",
    ]
    df = _read_data_with_candidates(candidates)
    biased_allowed_columns = [
        'Attendance (%)',
        'Midterm_Score',
        'Final_Score',
        'Projects_Score',
        'Study_Hours_per_Week',
        'Gender',
        'Parent_Education',
        'Internet_Access_at_Home'
    ]
    return df[biased_allowed_columns]