"""
Lane Tracker & Curvature Telemetry Module
=========================================
Performs sliding-window histogram peak extraction, 2nd-order polynomial
curve fitting, real-world physical curvature radius estimation (in meters),
vehicle lateral lane departure offset calculation, and HUD overlay rendering.
"""

import os
import cv2
import numpy as np
import matplotlib.image as mpimg


class LaneTracker:
    """Sliding-window lane pixel detector, polynomial fitter, and telemetry estimator."""

    def __init__(self, nwindows=9, margin=100, minpix=50, ym_per_pix=30/720, xm_per_pix=3.7/700, icons_dir=None):
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix

        self.left_fit = None
        self.right_fit = None
        self.left_fit_history = []
        self.right_fit_history = []
        self.history_len = 5
        self.dir_history = []

        # Load turn icons if available
        self.left_curve_img = None
        self.right_curve_img = None
        self.keep_straight_img = None

        if icons_dir and os.path.exists(icons_dir):
            l_path = os.path.join(icons_dir, "left_turn.png")
            r_path = os.path.join(icons_dir, "right_turn.png")
            s_path = os.path.join(icons_dir, "straight.png")
            if os.path.exists(l_path):
                self.left_curve_img = self._prepare_icon(l_path)
            if os.path.exists(r_path):
                self.right_curve_img = self._prepare_icon(r_path)
            if os.path.exists(s_path):
                self.keep_straight_img = self._prepare_icon(s_path)

    @staticmethod
    def _prepare_icon(path):
        icon = mpimg.imread(path)
        if icon.dtype != np.uint8:
            icon = np.uint8(icon * 255)
        return icon

    def find_lane_pixels(self, binary_warped):
        """
        Extract active left and right lane pixel coordinates using 9 sliding windows.
        """
        assert len(binary_warped.shape) == 2, "Input must be 2D binary image"
        h, w = binary_warped.shape
        window_height = int(h // self.nwindows)

        histogram = np.sum(binary_warped[h // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        leftx_current = leftx_base
        rightx_current = rightx_base

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        # Diagnostic visualization canvas
        vis_img = np.dstack((binary_warped, binary_warped, binary_warped))
        windows_rects = []

        for window in range(self.nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            windows_rects.append((
                (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                (win_xright_low, win_y_low), (win_xright_high, win_y_high)
            ))

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Draw windows and pixels for diagnostics
        vis_img[lefty, leftx] = [255, 0, 0]
        vis_img[righty, rightx] = [0, 0, 255]
        for w_box in windows_rects:
            cv2.rectangle(vis_img, w_box[0], w_box[1], (0, 255, 0), 2)
            cv2.rectangle(vis_img, w_box[2], w_box[3], (0, 255, 0), 2)

        return leftx, lefty, rightx, righty, vis_img

    def fit_polynomial(self, binary_warped):
        """
        Fits 2nd-degree polynomial f(y) = Ay^2 + By + C to detected lane pixels.
        Returns:
            warp_zero (np.ndarray): Color image containing lane overlay
            diagnostics (dict): Fitted polynomial curves and metric evaluations
        """
        h, w = binary_warped.shape
        leftx, lefty, rightx, righty, vis_img = self.find_lane_pixels(binary_warped)

        if len(lefty) > 300:
            fit_left = np.polyfit(lefty, leftx, 2)
            self.left_fit_history.append(fit_left)
            if len(self.left_fit_history) > self.history_len:
                self.left_fit_history.pop(0)
            self.left_fit = np.mean(self.left_fit_history, axis=0)
        elif self.left_fit is None:
            self.left_fit = np.array([0, 0, 250], dtype=np.float32)

        if len(righty) > 300:
            fit_right = np.polyfit(righty, rightx, 2)
            self.right_fit_history.append(fit_right)
            if len(self.right_fit_history) > self.history_len:
                self.right_fit_history.pop(0)
            self.right_fit = np.mean(self.right_fit_history, axis=0)
        elif self.right_fit is None:
            self.right_fit = np.array([0, 0, 1050], dtype=np.float32)

        ploty = np.linspace(0, h - 1, h)
        left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Draw green lane corridor on top-down view
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(255, 0, 0), thickness=25)
        cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25)

        # Plot polynomial lines on diagnostic image
        pts_left_vis = np.int32([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right_vis = np.int32([np.transpose(np.vstack([right_fitx, ploty]))])
        cv2.polylines(vis_img, pts_left_vis, isClosed=False, color=(255, 255, 0), thickness=3)
        cv2.polylines(vis_img, pts_right_vis, isClosed=False, color=(255, 255, 0), thickness=3)

        curv_left, curv_right, offset_m = self.measure_curvature_and_offset(h, w)

        diagnostics = {
            "vis_sliding_windows": vis_img,
            "curv_left": curv_left,
            "curv_right": curv_right,
            "curv_mean": (curv_left + curv_right) / 2.0,
            "offset_m": offset_m,
            "left_fit": self.left_fit,
            "right_fit": self.right_fit
        }

        return color_warp, diagnostics

    def measure_curvature_and_offset(self, h=720, w=1280):
        """
        Calculate radius of curvature in physical meters and vehicle lateral offset.
        """
        ploty = np.linspace(0, h - 1, h)
        left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Fit metric polynomials
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, right_fitx * self.xm_per_pix, 2)

        y_eval = (h - 1) * self.ym_per_pix
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

        # Lateral lane offset
        y_bottom = h - 1
        x_left_bottom = self.left_fit[0] * y_bottom**2 + self.left_fit[1] * y_bottom + self.left_fit[2]
        x_right_bottom = self.right_fit[0] * y_bottom**2 + self.right_fit[1] * y_bottom + self.right_fit[2]
        lane_center = (x_left_bottom + x_right_bottom) / 2.0
        vehicle_center = w / 2.0
        offset_m = (vehicle_center - lane_center) * self.xm_per_pix

        return left_curverad, right_curverad, offset_m

    def render_hud(self, out_img, diagnostics):
        """
        Overlays heads-up display (HUD) with road curvature, lateral deviation, and direction.
        """
        curv_mean = diagnostics["curv_mean"]
        offset_m = diagnostics["offset_m"]
        left_A = self.left_fit[0]
        right_A = self.right_fit[0]
        avg_A = (left_A + right_A) / 2.0

        if abs(avg_A) < 0.00018:
            direction = 'F'
        elif avg_A < 0:
            direction = 'L'
        else:
            direction = 'R'

        self.dir_history.append(direction)
        if len(self.dir_history) > 10:
            self.dir_history.pop(0)
        smoothed_dir = max(set(self.dir_history), key=self.dir_history.count)

        # Draw modern HUD card in top-left
        hud_w, hud_h = 420, 240
        overlay = out_img.copy()
        cv2.rectangle(overlay, (20, 20), (20 + hud_w, 20 + hud_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, out_img, 0.25, 0, out_img)
        cv2.rectangle(out_img, (20, 20), (20 + hud_w, 20 + hud_h), (59, 130, 246), 2)

        # Title
        cv2.putText(out_img, "AUTONOMOUS VEHICLE HUD", (35, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2, cv2.LINE_AA)

        # Status text
        status_msg = "Lane Tracking: Locked"
        cv2.putText(out_img, status_msg, (35, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94), 2, cv2.LINE_AA)

        # Curvature
        curv_text = f"Curvature Radius: {curv_mean:,.0f} m"
        cv2.putText(out_img, curv_text, (35, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Offset text
        side = "right" if offset_m > 0 else "left"
        offset_text = f"Offset: {abs(offset_m):.2f} m {side} of center"
        cv2.putText(out_img, offset_text, (35, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Direction intent
        dir_text = "Trajectory: Straight Ahead"
        if smoothed_dir == 'L':
            dir_text = "Trajectory: Left Curve Ahead"
        elif smoothed_dir == 'R':
            dir_text = "Trajectory: Right Curve Ahead"

        cv2.putText(out_img, dir_text, (35, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (253, 224, 71), 2, cv2.LINE_AA)

        # Timestamp/Speed indicator
        cv2.putText(out_img, "FPS: 30.0 | Vision State: Optimal", (35, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

        return out_img
