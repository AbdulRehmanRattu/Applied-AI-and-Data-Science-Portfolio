"""
Evaluation Metrics and Sparsity Analysis for MEG Source Localization.
Computes multi-label Jaccard error, precision, recall, and active source distributions.
Author: Abdul Rehman Rattu
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score


def compute_jaccard_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes sample-averaged Jaccard error: 1 - Jaccard Score."""
    score = jaccard_score(y_true, y_pred, average="samples", zero_division=1.0)
    return float(1.0 - score)


def evaluate_source_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes a comprehensive performance suite for multi-label source localization.
    Returns:
        dict: Jaccard error, precision, recall, F1, and predicted source statistics.
    """
    jaccard_err = compute_jaccard_error(y_true, y_pred)
    prec = float(precision_score(y_true, y_pred, average="samples", zero_division=1.0))
    rec = float(recall_score(y_true, y_pred, average="samples", zero_division=1.0))
    f1 = float(f1_score(y_true, y_pred, average="samples", zero_division=1.0))

    predicted_counts = np.sum(y_pred, axis=1)
    true_counts = np.sum(y_true, axis=1)

    return {
        "jaccard_error": jaccard_err,
        "jaccard_score": 1.0 - jaccard_err,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "predicted_source_counts": np.unique(predicted_counts).tolist(),
        "mean_predicted_sources": float(np.mean(predicted_counts)),
        "mean_true_sources": float(np.mean(true_counts)),
    }
