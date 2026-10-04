"""
Dataset Ingestion and Lead-Field Matrix Loader for MEG Signals.
Handles compressed CSV streams, sparse CSR target reconstruction, and subject lead fields.
Author: Abdul Rehman Rattu
"""

import os
import glob
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_subject_lead_fields(data_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads all subject-specific lead field matrices and parcel index mappings.
    Returns:
        dict: Mapping from subject string ('subject_1', etc.) to {'lead_field': ..., 'parcel_indices': ...}
    """
    lead_fields: Dict[str, Dict[str, np.ndarray]] = {}
    pattern = os.path.join(data_dir, "subject_*_L.npz")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No subject lead field files found matching {pattern}")

    for file_path in files:
        base_name = os.path.basename(file_path).replace("_L.npz", "")
        npz = np.load(file_path)
        lead_fields[base_name] = {
            "lead_field": npz["lead_field"],
            "parcel_indices": npz["parcel_indices"],
        }

    return lead_fields


def _load_sparse_target(npz_path: str) -> np.ndarray:
    """Loads and reconstructs a compressed CSR target matrix into a dense binary array."""
    target_file = np.load(npz_path)
    y_sparse = csr_matrix(
        (target_file["data"], target_file["indices"], target_file["indptr"]),
        shape=target_file["shape"]
    )
    return y_sparse.toarray().astype(int)


def load_meg_dataset(
    data_dir: Optional[str] = None,
    use_benchmark: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Loads training and test MEG features along with ground truth parcel activation matrices.
    Args:
        data_dir: Path to directory containing train, test, and subject lead fields.
        use_benchmark: If True, uses stratified benchmark training slice (5,000 samples).
    Returns:
        X_train, y_train, X_test, y_test, lead_fields
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")

    # 1. Load lead fields
    lead_fields = load_subject_lead_fields(data_dir)

    # 2. Select training files
    train_dir = os.path.join(data_dir, "train")
    if use_benchmark and os.path.exists(os.path.join(train_dir, "X_benchmark.csv.gz")):
        train_csv = os.path.join(train_dir, "X_benchmark.csv.gz")
        train_target = os.path.join(train_dir, "target_benchmark.npz")
    elif os.path.exists(os.path.join(train_dir, "X.csv.gz")):
        train_csv = os.path.join(train_dir, "X.csv.gz")
        train_target = os.path.join(train_dir, "target.npz")
    elif os.path.exists(os.path.join(train_dir, "X.csv")):
        train_csv = os.path.join(train_dir, "X.csv")
        train_target = os.path.join(train_dir, "target.npz")
    else:
        raise FileNotFoundError(f"No valid training data found in {train_dir}")

    X_train = pd.read_csv(train_csv)
    y_train = _load_sparse_target(train_target)

    # 3. Select test files
    test_dir = os.path.join(data_dir, "test")
    test_csv = os.path.join(test_dir, "X.csv.gz" if os.path.exists(os.path.join(test_dir, "X.csv.gz")) else "X.csv")
    test_target = os.path.join(test_dir, "target.npz")

    X_test = pd.read_csv(test_csv)
    y_test = _load_sparse_target(test_target)

    # Reconstruct L_path metadata if needed
    X_train["L_path"] = X_train["subject"].apply(lambda s: f"data/{s}_L.npz")
    X_test["L_path"] = X_test["subject"].apply(lambda s: f"data/{s}_L.npz")

    return X_train, y_train, X_test, y_test, lead_fields
