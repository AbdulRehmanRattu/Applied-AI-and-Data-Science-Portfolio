"""
Models suite for MEG Neural Source Localization.
"""

from meg_localization.models.knn_baseline import KNNSourceLocalizer
from meg_localization.models.lasso_sparse_inversion import LassoLarsSourceLocalizer
from meg_localization.models.megnet_pytorch import MEGNetLocalizer

__all__ = [
    "KNNSourceLocalizer",
    "LassoLarsSourceLocalizer",
    "MEGNetLocalizer",
]
