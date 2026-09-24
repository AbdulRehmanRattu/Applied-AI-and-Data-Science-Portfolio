"""
Lane Detection Pipeline Module
==============================
Unified orchestrator coordinating camera undistortion, perspective warping,
color/gradient thresholding, polynomial tracking, and HUD telemetry rendering.
"""

import os
import cv2
import numpy as np
import matplotlib.image as mpimg

from .camera_calibration import CameraCalibration
from .perspective import PerspectiveTransformation
from .thresholding import Thresholding
from .lane_tracker import LaneTracker


class LaneDetectionPipeline:
    """End-to-end highway lane detection and telemetry pipeline."""

    def __init__(self, cal_dir="camera_cal", icons_dir="assets/icons", img_size=(1280, 720)):
        self.img_size = img_size
        self.calibration = CameraCalibration(cal_dir=cal_dir)
        self.perspective = PerspectiveTransformation(img_size=img_size)
        self.thresholding = Thresholding()
        self.tracker = LaneTracker(icons_dir=icons_dir)

    def forward(self, img_rgb):
        """
        Processes a single RGB camera frame and returns annotated image with HUD.
        """
        # 1. Undistort camera lens
        undist = self.calibration.undistort(img_rgb)

        # 2. Inverse Perspective Mapping (Warp to bird's-eye view)
        warped = self.perspective.forward(undist)

        # 3. Adaptive Color & Gradient Thresholding
        binary_warped = self.thresholding.forward(warped)

        # 4. Polynomial Fit & Lane Tracking
        color_warp, diagnostics = self.tracker.fit_polynomial(binary_warped)

        # 5. Unwarp detected lane corridor back to original camera view
        unwarped_lane = self.perspective.backward(color_warp)

        # 6. Alpha-blend lane corridor onto undistorted camera frame
        blended = cv2.addWeighted(undist, 1.0, unwarped_lane, 0.45, 0)

        # 7. Render telemetry HUD
        final_hud = self.tracker.render_hud(blended, diagnostics)

        return final_hud

    def get_diagnostic_stages(self, img_rgb):
        """
        Extracts intermediate images across all 6 pipeline stages for research diagnostics:
        1. Raw Input Frame
        2. Undistorted Frame
        3. Bird's-Eye Perspective View
        4. Binary Threshold Mask
        5. Sliding Window Polynomial Fit
        6. Final HUD Telemetry Overlay
        """
        undist = self.calibration.undistort(img_rgb)
        warped = self.perspective.forward(undist)
        binary_warped = self.thresholding.forward(warped)
        color_warp, diagnostics = self.tracker.fit_polynomial(binary_warped)
        unwarped_lane = self.perspective.backward(color_warp)
        blended = cv2.addWeighted(undist, 1.0, unwarped_lane, 0.45, 0)
        final_hud = self.tracker.render_hud(blended.copy(), diagnostics)

        return {
            "raw": img_rgb,
            "undistorted": undist,
            "warped": warped,
            "binary_warped": binary_warped,
            "sliding_windows": diagnostics["vis_sliding_windows"],
            "final_hud": final_hud,
            "diagnostics": diagnostics
        }

    def process_image(self, input_path, output_path):
        """Processes a single image file and saves annotated output."""
        img = mpimg.imread(input_path)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.uint8(img * 255)

        annotated = self.forward(img)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mpimg.imsave(output_path, annotated)
        return output_path

    def process_video(self, input_path, output_path, max_frames=None, show_progress=True):
        """
        Processes a video file frame-by-frame via OpenCV VideoCapture and VideoWriter.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.img_size[0]
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.img_size[1]

        if max_frames:
            total_frames = min(total_frames, max_frames)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb = self.forward(frame_rgb)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            out.write(annotated_bgr)
            frame_count += 1

            if show_progress and (frame_count % 25 == 0 or frame_count == total_frames):
                pct = (frame_count / total_frames) * 100 if total_frames else 0
                print(f"  * Processed {frame_count}/{total_frames} frames ({pct:.1f}%)")

            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        out.release()
        return output_path
