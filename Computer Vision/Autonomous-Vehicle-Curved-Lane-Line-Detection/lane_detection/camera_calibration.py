"""
Camera Calibration Module
=========================
Calculates camera intrinsic matrix (K) and distortion coefficients (dist)
using OpenCV chessboard calibration patterns, providing distortion correction.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg


class CameraCalibration:
    """Computes and caches camera intrinsic parameters and applies undistortion."""

    def __init__(self, cal_dir="camera_cal", nx=9, ny=6, cache_file=None):
        """
        Parameters:
            cal_dir (str): Directory containing chessboard images
            nx (int): Internal corners along chessboard width
            ny (int): Internal corners along chessboard height
            cache_file (str, optional): Path to cached npz file
        """
        self.cal_dir = cal_dir
        self.nx = nx
        self.ny = ny
        if cache_file is None:
            self.cache_file = os.path.join(cal_dir, "calibration_cache.npz")
        else:
            self.cache_file = cache_file

        self.mtx = None
        self.dist = None
        self._calibrate()

    def _calibrate(self):
        # 1. Try loading cached parameters
        if os.path.exists(self.cache_file):
            try:
                data = np.load(self.cache_file)
                self.mtx = data["mtx"]
                self.dist = data["dist"]
                return
            except Exception:
                pass

        # 2. Compute from chessboard images
        fnames = sorted(glob.glob(os.path.join(self.cal_dir, "*.jpg")))
        if not fnames:
            raise FileNotFoundError(f"No calibration images found in directory: {self.cal_dir}")

        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        img_shape = None

        for fname in fnames:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_shape is None:
                img_shape = (gray.shape[1], gray.shape[0])

            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if not objpoints:
            raise RuntimeError("Failed to detect chessboard corners in calibration images.")

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        if not ret:
            raise RuntimeError("OpenCV camera calibration computation failed.")

        # Save cache
        try:
            np.savez(self.cache_file, mtx=self.mtx, dist=self.dist)
        except Exception:
            pass

    def undistort(self, img):
        """
        Apply lens distortion correction to input RGB/BGR image.
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
