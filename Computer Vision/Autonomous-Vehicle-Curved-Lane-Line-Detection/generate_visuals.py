#!/usr/bin/env python3
"""
Visual Documentation Asset Generator
====================================
Generates 300 DPI publication-grade technical figures:
1. lane_detection_pipeline_architecture.png (System block diagram)
2. lane_detection_stage_breakdown.png (6-panel empirical stage transformation)

Author: Abdul Rehman Rattu
License: MIT
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure package access
MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODULE_ROOT)

from lane_detection.pipeline import LaneDetectionPipeline


def generate_architecture_diagram(output_path="assets/docs/lane_detection_pipeline_architecture.png"):
    """Generates 300 DPI high-resolution system architecture flowchart."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    ax.text(50, 95, "AUTONOMOUS VEHICLE CURVED LANE DETECTION & TELEMETRY",
            fontsize=18, fontweight="bold", ha="center", color="#0F172A")
    ax.text(50, 91.5, "Self-Driving Perception Pipeline: Geometric Calibration, Bird's-Eye Warping, Polynomial Fitting, and Metric Curvature Telemetry",
            fontsize=11, ha="center", color="#475569")

    # Pipeline stages
    stages = [
        {
            "num": "01",
            "title": "Camera Intrinsic Calibration",
            "color": "#2563EB",
            "bg": "#EFF6FF",
            "x": 6, "y": 55, "w": 25, "h": 28,
            "bullets": [
                "20 Chessboard image captures",
                "Corner point detection (9x6)",
                "Intrinsic camera matrix K",
                "Radial/tangential distortion [k1, k2, p1, p2]"
            ]
        },
        {
            "num": "02",
            "title": "Inverse Perspective Mapping",
            "color": "#4F46E5",
            "bg": "#EEF2FF",
            "x": 37.5, "y": 55, "w": 25, "h": 28,
            "bullets": [
                "Trapezoidal road surface ROI",
                "Homography matrix M & M⁻¹",
                "Bird's-eye top-down unwarping",
                "Orthogonal parallel lane rectification"
            ]
        },
        {
            "num": "03",
            "title": "Multi-Channel Thresholding",
            "color": "#059669",
            "bg": "#ECFDF5",
            "x": 69, "y": 55, "w": 25, "h": 28,
            "bullets": [
                "HLS S-channel saturation filtering",
                "HSV Hue gating for yellow lines",
                "L-channel brightness for white lines",
                "Directional Sobel Sx edge gradients"
            ]
        },
        {
            "num": "04",
            "title": "Sliding-Window Polynomial Fit",
            "color": "#D97706",
            "bg": "#FFFBEB",
            "x": 6, "y": 14, "w": 25, "h": 28,
            "bullets": [
                "Vertical histogram peak base search",
                "9 Adaptive sliding search windows",
                "Quadratic fitting: x = Ay² + By + C",
                "Temporal buffer jitter smoothing"
            ]
        },
        {
            "num": "05",
            "title": "Physical Road Geometry & Telemetry",
            "color": "#7C3AED",
            "bg": "#F5F3FF",
            "x": 37.5, "y": 14, "w": 25, "h": 28,
            "bullets": [
                "Metric scale: ym=30/720m, xm=3.7/700m",
                "Curvature: R = (1+(2Ay+B)²)^1.5 / |2A|",
                "Lateral lane departure offset (m)",
                "Turn direction intent classification"
            ]
        },
        {
            "num": "06",
            "title": "HUD Projection & ADAS Actuation",
            "color": "#DC2626",
            "bg": "#FEF2F2",
            "x": 69, "y": 14, "w": 25, "h": 28,
            "bullets": [
                "Translucent green corridor back-projection",
                "Alpha blending with undistorted camera",
                "Real-time HUD telemetry card",
                "Steering angle guidance / Lane departure"
            ]
        }
    ]

    for s in stages:
        # Card outline
        rect = patches.FancyBboxPatch(
            (s["x"], s["y"]), s["w"], s["h"],
            boxstyle="round,pad=1.0,rounding_size=2.0",
            facecolor=s["bg"], edgecolor=s["color"], linewidth=2.0
        )
        ax.add_patch(rect)

        # Stage Number Badge
        badge = patches.FancyBboxPatch(
            (s["x"] + 1.2, s["y"] + s["h"] - 4.5), 5.5, 3.2,
            boxstyle="round,pad=0.2,rounding_size=0.8",
            facecolor=s["color"], edgecolor="none"
        )
        ax.add_patch(badge)
        ax.text(s["x"] + 3.95, s["y"] + s["h"] - 3.0, s["num"],
                fontsize=11, fontweight="bold", color="#FFFFFF", ha="center", va="center")

        # Stage Title
        ax.text(s["x"] + 7.2, s["y"] + s["h"] - 3.0, s["title"],
                fontsize=10.2, fontweight="bold", color=s["color"], va="center")

        # Bullets
        bullet_y = s["y"] + s["h"] - 8.0
        for b in s["bullets"]:
            ax.text(s["x"] + 1.8, bullet_y, "•", fontsize=11, color=s["color"], fontweight="bold")
            ax.text(s["x"] + 3.8, bullet_y, b, fontsize=9.2, color="#1E293B", va="center")
            bullet_y -= 4.4

    # Connectors (Arrows)
    # 01 -> 02
    ax.annotate("", xy=(37.0, 69.0), xytext=(31.5, 69.0),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#64748B", lw=2.2))
    # 02 -> 03
    ax.annotate("", xy=(68.5, 69.0), xytext=(63.0, 69.0),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#64748B", lw=2.2))
    # 03 -> 04 Clean orthogonal corridor routing (zero collision)
    ax.plot([81.5, 81.5], [54.5, 48.5], color="#64748B", lw=2.2)
    ax.plot([81.5, 18.5], [48.5, 48.5], color="#64748B", lw=2.2)
    ax.annotate("", xy=(18.5, 42.5), xytext=(18.5, 48.5),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#64748B", lw=2.2))
    # 04 -> 05
    ax.annotate("", xy=(37.0, 28.0), xytext=(31.5, 28.0),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#64748B", lw=2.2))
    # 05 -> 06
    ax.annotate("", xy=(68.5, 28.0), xytext=(63.0, 28.0),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#64748B", lw=2.2))

    # Footer
    ax.text(50, 3.5, "Applied AI Master Portfolio | Computer Vision Autonomous Driving Laboratory | Abdul Rehman Rattu",
            fontsize=10, ha="center", color="#64748B", style="italic")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    print(f"[+] Architecture diagram saved: {output_path}")


def generate_stage_breakdown(output_path="assets/docs/lane_detection_stage_breakdown.png"):
    """Processes a test image through all 6 stages and renders a 300 DPI composite grid."""
    test_img_path = os.path.join(MODULE_ROOT, "test_images", "test1.jpg")
    if not os.path.exists(test_img_path):
        test_img_path = os.path.join(MODULE_ROOT, "test_images", "straight_lines1.jpg")

    pipeline = LaneDetectionPipeline(
        cal_dir=os.path.join(MODULE_ROOT, "camera_cal"),
        icons_dir=os.path.join(MODULE_ROOT, "assets", "icons")
    )

    img_rgb = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)
    stages = pipeline.get_diagnostic_stages(img_rgb)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    panel_info = [
        (stages["raw"], "(a) Raw Camera Input Frame (1280x720 RGB)", "#1E293B"),
        (stages["undistorted"], "(b) Lens Distortion Corrected (Intrinsic Matrix K)", "#2563EB"),
        (stages["warped"], "(c) Inverse Perspective Mapping (Bird's-Eye View)", "#4F46E5"),
        (stages["binary_warped"], "(d) Adaptive Color & Gradient Threshold Mask", "#059669"),
        (stages["sliding_windows"], "(e) 9 Sliding Windows & 2nd-Order Polynomial Fits", "#D97706"),
        (stages["final_hud"], "(f) Back-Projected Lane Corridor & Metric HUD Telemetry", "#DC2626")
    ]

    for ax, (img_data, title, title_color) in zip(axes.flat, panel_info):
        ax.set_facecolor("#F8FAFC")
        if len(img_data.shape) == 2:
            ax.imshow(img_data, cmap="gray")
        else:
            ax.imshow(img_data)
        ax.set_title(title, fontsize=12, fontweight="bold", color=title_color, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(title_color)
            spine.set_linewidth(1.5)

    curv_m = stages["diagnostics"]["curv_mean"]
    offset_m = stages["diagnostics"]["offset_m"]
    side = "Right" if offset_m > 0 else "Left"

    fig.suptitle(
        f"Empirical Intermediate Stage Transformations across Autonomous Driving Perception Pipeline\n"
        f"[Physical Metrics: Road Curvature Radius = {curv_m:,.0f} m | Lateral Center Departure = {abs(offset_m):.2f} m {side}]",
        fontsize=14, fontweight="bold", color="#0F172A", y=0.98
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(output_path, dpi=300, facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    print(f"[+] Stage breakdown saved: {output_path}")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_stage_breakdown()
    print("[+] All visual documentation assets generated successfully.")
