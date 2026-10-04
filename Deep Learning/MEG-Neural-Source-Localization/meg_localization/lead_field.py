"""
Biophysical Lead-Field Matrix Inversion and Source Space Projection.
Maps 204 magnetic sensor channels into 450 cortical parcel feature representations
using Maxwell sensitivity matrices and subject-specific anatomy.
Author: Abdul Rehman Rattu
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def project_sensors_to_source_space(
    X: pd.DataFrame,
    lead_fields: Dict[str, Dict[str, np.ndarray]],
    scale_factor: float = 1e8,
    aggregation: str = "max"
) -> np.ndarray:
    """
    Projects raw 204-sensor MEG readings into 450 cortical parcel spaces.
    Args:
        X: DataFrame containing sensor columns ('e1' to 'e204') and 'subject'.
        lead_fields: Subject-specific lead field dictionaries.
        scale_factor: Multiplicative constant for numerical stability (default: 1e8).
        aggregation: Aggregation method across dipoles in each parcel ('max' or 'mean').
    Returns:
        X_features: (N, 450) array of normalized parcel activations.
    """
    n_samples = X.shape[0]
    n_parcels = 450
    X_features = np.zeros((n_samples, n_parcels), dtype=float)

    sensor_cols = [f"e{i}" for i in range(1, 205)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[sensor_cols])

    for subject_id in np.unique(X["subject"]):
        if subject_id not in lead_fields:
            raise KeyError(f"Subject '{subject_id}' not found in provided lead fields.")

        lead_data = lead_fields[subject_id]
        L = lead_data["lead_field"] * scale_factor
        parcel_indices = lead_data["parcel_indices"]

        # Subject mask
        mask = (X["subject"] == subject_id).values
        X_sub = X_scaled[mask]

        if X_sub.shape[0] == 0:
            continue

        # Column normalization of lead field
        norms = np.linalg.norm(L, axis=0)
        norms[norms == 0] = 1.0
        L_norm = L / norms[None, :]

        # Approximate source amplitudes
        z_est = np.abs(X_sub @ L_norm)

        # Aggregate dipoles belonging to the same cortical parcel
        df_dipoles = pd.DataFrame(z_est).T
        if aggregation == "mean":
            aggregated = df_dipoles.groupby(parcel_indices).mean().T.values
        else:
            aggregated = df_dipoles.groupby(parcel_indices).max().T.values

        # Normalize across parcel features
        z_parcels = scaler.fit_transform(aggregated)
        X_features[mask] = z_parcels

    return X_features
