"""
Metacell AI: Deep Neural Network Surrogate & Inverse Design Framework
======================================================================
Physics-informed deep surrogate models for large-phase-shift metacell
optimization in computational electromagnetics and metasurface engineering.

Author: Abdul Rehman Rattu
License: MIT
"""

from .models import ForwardSurrogateNetwork, InverseSynthesisNetwork
from .data_loader import MetacellDataLoader
from .evaluator import MetacellEvaluator

__all__ = [
    "ForwardSurrogateNetwork",
    "InverseSynthesisNetwork",
    "MetacellDataLoader",
    "MetacellEvaluator"
]
