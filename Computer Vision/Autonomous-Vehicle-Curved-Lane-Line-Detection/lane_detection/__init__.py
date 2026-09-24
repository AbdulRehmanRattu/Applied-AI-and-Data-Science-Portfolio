"""
Autonomous Vehicle Lane Detection & Highway Road Telemetry
==========================================================
Advanced computer vision pipeline for real-time lane tracking,
camera intrinsic calibration, inverse perspective mapping,
and curvature estimation in autonomous driving.

Author: Abdul Rehman Rattu
License: MIT
"""

from .camera_calibration import CameraCalibration
from .perspective import PerspectiveTransformation
from .thresholding import Thresholding
from .lane_tracker import LaneTracker
from .pipeline import LaneDetectionPipeline

__all__ = [
    "CameraCalibration",
    "PerspectiveTransformation",
    "Thresholding",
    "LaneTracker",
    "LaneDetectionPipeline"
]
