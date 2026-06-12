"""
load_data.py
============
Load all PhysioNet 2019 Sepsis Challenge .psv files from the data/ directory
into a single pandas DataFrame with a patient_id column derived from filenames.
"""

import os
import pandas as pd
from tqdm import tqdm


def load_all_patients(data_dir: str = "data") -> pd.DataFrame:
    """
    Read every .psv file in `data_dir`, tag rows with patient_id, and
    concatenate into one DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing .psv files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with an added 'patient_id' column.
    """
    psv_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".psv")]
    )

    if len(psv_files) == 0:
        raise FileNotFoundError(
            f"No .psv files found in '{data_dir}/'. "
            "Please download the PhysioNet 2019 Sepsis Challenge data and "
            "place the .psv files in the data/ directory."
        )

    frames = []
    for fname in tqdm(psv_files, desc="Loading patient files"):
        filepath = os.path.join(data_dir, fname)
        patient_id = os.path.splitext(fname)[0]  # e.g. "p000001"
        df = pd.read_csv(filepath, sep="|")
        df["patient_id"] = patient_id
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # ---- summary statistics ----
    n_patients = combined["patient_id"].nunique()
    n_rows = len(combined)
    n_sepsis = int(combined["SepsisLabel"].sum())
    n_no_sepsis = n_rows - n_sepsis
    prevalence = n_sepsis / n_rows * 100

    print(f"\n{'='*55}")
    print(f"  Dataset loaded successfully")
    print(f"{'='*55}")
    print(f"  Patients        : {n_patients:,}")
    print(f"  Total rows      : {n_rows:,}")
    print(f"  SepsisLabel = 1 : {n_sepsis:,}  ({prevalence:.2f}%)")
    print(f"  SepsisLabel = 0 : {n_no_sepsis:,}  ({100 - prevalence:.2f}%)")
    print(f"  Columns         : {combined.shape[1]}")
    print(f"{'='*55}\n")

    return combined


if __name__ == "__main__":
    df = load_all_patients()
    print(df.head())
