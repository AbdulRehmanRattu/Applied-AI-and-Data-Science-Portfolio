"""
MEG Neural Source Localization Suite.
Computational neuroscience pipeline for biophysical electromagnetic source localization.
Author: Abdul Rehman Rattu
"""

from meg_localization.data_loader import load_meg_dataset, load_subject_lead_fields
from meg_localization.lead_field import project_sensors_to_source_space
from meg_localization.models.knn_baseline import KNNSourceLocalizer
from meg_localization.models.lasso_sparse_inversion import LassoLarsSourceLocalizer
from meg_localization.models.megnet_pytorch import MEGNetLocalizer
from meg_localization.metrics import compute_jaccard_error, evaluate_source_predictions

__all__ = [
    "load_meg_dataset",
    "load_subject_lead_fields",
    "project_sensors_to_source_space",
    "KNNSourceLocalizer",
    "LassoLarsSourceLocalizer",
    "MEGNetLocalizer",
    "compute_jaccard_error",
    "evaluate_source_predictions",
]

__version__ = "1.0.0"
__author__ = "Abdul Rehman Rattu"
