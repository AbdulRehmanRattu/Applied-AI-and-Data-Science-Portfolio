"""
Perspective Transformation (Inverse Perspective Mapping)
=======================================================
Warps front-facing vehicle camera perspective into an orthogonal
bird's-eye top-down view of the road surface, and unwarps back.
"""

import cv2
import numpy as np


class PerspectiveTransformation:
    """Computes homography matrices and transforms between camera and bird's-eye views."""

    def __init__(self, img_size=(1280, 720)):
        """
        Parameters:
            img_size (tuple): (width, height) of the camera frame
        """
        self.img_size = img_size
        w, h = img_size

        # Trapezoidal region of interest on the road surface
        self.src = np.float32([
            (550, 460),    # Top-Left
            (150, 720),    # Bottom-Left
            (1200, 720),   # Bottom-Right
            (770, 460)     # Top-Right
        ])

        # Orthogonal bird's-eye destination coordinates
        self.dst = np.float32([
            (100, 0),      # Top-Left
            (100, 720),    # Bottom-Left
            (1100, 720),   # Bottom-Right
            (1100, 0)      # Top-Right
        ])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        """
        Transform from vehicle front perspective to top-down bird's-eye view.
        """
        return cv2.warpPerspective(img, self.M, self.img_size, flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        """
        Transform from top-down bird's-eye view back to vehicle front perspective.
        """
        return cv2.warpPerspective(img, self.M_inv, self.img_size, flags=flags)
