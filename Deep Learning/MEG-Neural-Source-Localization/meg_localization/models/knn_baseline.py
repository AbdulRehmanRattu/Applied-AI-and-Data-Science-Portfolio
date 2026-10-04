"""
Multi-Output k-Nearest Neighbors Baseline for Neural Source Localization.
Projects sensor signals to source space and ranks parcel probabilities.
Author: Abdul Rehman Rattu
"""

from typing import Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier


class KNNSourceLocalizer:
    """
    Multi-output k-NN baseline for cortical parcel source localization.
    Enforces dynamic top-k source selection (up to 3 parcels).
    """

    def __init__(self, n_neighbors: int = 5, max_sources: int = 3):
        self.n_neighbors = n_neighbors
        self.max_sources = max_sources
        self.clf = MultiOutputClassifier(
            KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="brute"),
            n_jobs=1
        )

    def fit(self, X_features: np.ndarray, y: np.ndarray) -> "KNNSourceLocalizer":
        """Fits multi-output k-NN on 450-dimensional parcel features."""
        self.clf.fit(X_features, y)
        return self

    def predict(self, X_features: np.ndarray) -> np.ndarray:
        """
        Predicts active cortical parcels by extracting top-k parcel probabilities per sample.
        Returns:
            y_pred: (N, 450) binary matrix with at most max_sources ones per row.
        """
        # MultiOutputClassifier returns a list of (N, 2) probability arrays for each parcel
        probas_list = self.clf.predict_proba(X_features)
        prob_class1 = np.array([p[:, 1] if p.shape[1] > 1 else np.zeros(len(p)) for p in probas_list]).T

        n_samples = X_features.shape[0]
        n_parcels = prob_class1.shape[1]
        y_pred = np.zeros((n_samples, n_parcels), dtype=int)

        for i in range(n_samples):
            # Select top parcel indices
            top_indices = np.argpartition(prob_class1[i], -self.max_sources)[-self.max_sources:]
            # Filter by non-zero probability
            valid_indices = [idx for idx in top_indices if prob_class1[i, idx] > 0.0]
            if not valid_indices:
                valid_indices = top_indices.tolist()
            y_pred[i, valid_indices] = 1

        return y_pred
