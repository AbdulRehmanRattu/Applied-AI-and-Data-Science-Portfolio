"""
Color and Gradient Thresholding Module
======================================
Combines HLS color space (S and L channels), HSV color space, and
directional Sobel gradient filtering to extract lane boundary pixels
under varying illumination, shadow gradients, and asphalt reflectance.
"""

import cv2
import numpy as np


class Thresholding:
    """Extracts lane boundary pixels using adaptive color and gradient thresholding."""

    def __init__(self, sobel_kernel=3):
        self.sobel_kernel = sobel_kernel

    @staticmethod
    def _threshold_rel(channel, lo_ratio, hi_ratio):
        vmin = np.min(channel)
        vmax = np.max(channel)
        if vmax == vmin:
            return np.zeros_like(channel, dtype=np.uint8)
        vlo = vmin + (vmax - vmin) * lo_ratio
        vhi = vmin + (vmax - vmin) * hi_ratio
        return np.uint8((channel >= vlo) & (channel <= vhi)) * 255

    @staticmethod
    def _threshold_abs(channel, lo_val, hi_val):
        return np.uint8((channel >= lo_val) & (channel <= hi_val)) * 255

    def forward(self, img_rgb):
        """
        Process an RGB top-down bird's-eye image into a binary thresholded mask.

        Parameters:
            img_rgb (np.ndarray): Warped RGB image (H, W, 3)

        Returns:
            binary_mask (np.ndarray): Binary lane pixel mask (H, W) with values {0, 255}
        """
        hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        v_channel = hsv[:, :, 2]

        # White lane isolation via lightness and saturation
        right_lane = self._threshold_rel(l_channel, 0.78, 1.0)
        right_lane[:, :650] = 0

        # Yellow lane isolation via hue and value
        yellow_h = self._threshold_abs(h_channel, 15, 35)
        yellow_v = self._threshold_rel(v_channel, 0.65, 1.0)
        left_lane = yellow_h & yellow_v
        left_lane[:, 600:] = 0

        # Sobel X gradient thresholding for sharp lane contrast
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
        grad_lane = self._threshold_abs(scaled_sobel, 30, 200)

        # S-channel saturation reinforcement
        s_binary = self._threshold_rel(s_channel, 0.6, 1.0)
        combined_grad = (grad_lane & s_binary)

        # Combined binary lane extraction
        combined_binary = (left_lane | right_lane | combined_grad)

        return combined_binary
