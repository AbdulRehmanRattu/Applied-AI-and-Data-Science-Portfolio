"""
Lasso-Lars Sparse Regularization for Biophysical Source Inversion.
Inverts electromagnetic lead fields (x = Lz) using L1 penalization to identify sparse active parcels.
Author: Abdul Rehman Rattu
"""

from typing import Dict, Any, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.linear_model._least_angle")


class LassoLarsSourceLocalizer(BaseEstimator, ClassifierMixin):
    """
    Solves the ill-posed MEG inverse problem via sparse L1 regularized Least Angle Regression (Lasso-Lars).
    Leverages subject-specific physical lead fields to estimate sparse dipole moments.
    """

    def __init__(self, alpha: float = 0.5, max_sources: int = 3, max_iter: int = 200):
        self.alpha = alpha
        self.max_sources = max_sources
        self.max_iter = max_iter
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "LassoLarsSourceLocalizer":
        """Fits standard scaler on 204 sensor channels."""
        sensor_cols = [f"e{i}" for i in range(1, 205)]
        self.scaler.fit(X[sensor_cols])
        return self

    def _solve_sample_dipoles(self, L: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Solves L1 sparse regression for a single sample x against lead field L."""
        norms = np.linalg.norm(L, axis=0)
        norms[norms == 0] = 1.0
        L_norm = L / norms[None, :]

        # Scale alpha relative to maximum gradient
        grad_max = np.abs(L_norm.T.dot(x)).max() / len(L_norm)
        eff_alpha = self.alpha * grad_max

        lasso = LassoLars(alpha=eff_alpha, max_iter=self.max_iter, fit_intercept=False)
        lasso.fit(L_norm, x)

        coef = np.abs(lasso.coef_) / norms
        return coef

    def decision_function(
        self,
        X: pd.DataFrame,
        lead_fields: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Computes 450-dimensional cortical parcel activation intensities (betas).
        """
        sensor_cols = [f"e{i}" for i in range(1, 205)]
        X_scaled = self.scaler.transform(X[sensor_cols])

        n_samples = len(X)
        n_parcels = 450
        betas = np.zeros((n_samples, n_parcels), dtype=float)

        subjects = X["subject"].values

        for subj_id in np.unique(subjects):
            lead_info = lead_fields[subj_id]
            L = lead_info["lead_field"] * 1e8
            parcels = lead_info["parcel_indices"]

            subj_mask = np.where(subjects == subj_id)[0]
            for idx in subj_mask:
                x_vec = X_scaled[idx]
                dipole_coefs = self._solve_sample_dipoles(L, x_vec)

                # Aggregate dipole coefficients into 450 cortical parcels
                df_c = pd.Series(dipole_coefs).groupby(parcels).max()
                betas[idx, df_c.index] = df_c.values

        return betas

    def predict(
        self,
        X: pd.DataFrame,
        lead_fields: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Predicts binary parcel activations with dynamic sparsity (<= 3 sources).
        """
        betas = self.decision_function(X, lead_fields)
        n_samples = betas.shape[0]
        y_pred = np.zeros_like(betas, dtype=int)

        for i in range(n_samples):
            sorted_indices = np.argsort(betas[i])[::-1]
            non_zero_count = int(np.sum(betas[i] > 0.0))
            k = min(self.max_sources, max(1, non_zero_count))
            top_indices = sorted_indices[:k]
            y_pred[i, top_indices] = 1

        return y_pred
