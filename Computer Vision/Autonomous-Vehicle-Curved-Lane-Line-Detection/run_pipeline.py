#!/usr/bin/env python3
"""
Autonomous Vehicle Lane Detection CLI
=====================================
Unified command-line interface for camera calibration, single frame inference,
batch image testing, and real-time highway driving video telemetry rendering.

Author: Abdul Rehman Rattu
License: MIT
"""

import os
import sys
import glob
import argparse
import time

# Ensure local module access
MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODULE_ROOT)

from lane_detection.pipeline import LaneDetectionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Vehicle Curved Lane Line Detection & Curvature Telemetry CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["image", "batch", "video", "calibrate", "visuals", "all"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input image or video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output image or video file"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum video frames to process (optional)"
    )
    parser.add_argument(
        "--cal-dir",
        type=str,
        default=os.path.join(MODULE_ROOT, "camera_cal"),
        help="Directory containing chessboard calibration images"
    )
    parser.add_argument(
        "--icons-dir",
        type=str,
        default=os.path.join(MODULE_ROOT, "assets", "icons"),
        help="Directory containing HUD turn icons"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("   AUTONOMOUS VEHICLE CURVED LANE LINE DETECTION & TELEMETRY PIPELINE")
    print("   Self-Driving Perception | Intrinsic Calibration | Metric Road Geometry")
    print("=" * 80)

    # Initialize pipeline
    t0 = time.time()
    pipeline = LaneDetectionPipeline(
        cal_dir=args.cal_dir,
        icons_dir=args.icons_dir
    )
    init_time = time.time() - t0
    print(f"[+] Initialized Pipeline in {init_time:.2f}s (Camera Matrix & Homography Loaded)")
    print(f"    - Intrinsic Matrix K focal lengths: fx={pipeline.calibration.mtx[0,0]:.1f}, fy={pipeline.calibration.mtx[1,1]:.1f}")
    print(f"    - Distortion Coefficients: k1={pipeline.calibration.dist[0,0]:.4f}, k2={pipeline.calibration.dist[0,1]:.4f}\n")

    if args.mode in ["calibrate"]:
        print("[+] Camera calibration verified successfully. Parameters cached in camera_cal/calibration_cache.npz\n")
        return

    if args.mode in ["image", "all"]:
        inp = args.input or os.path.join(MODULE_ROOT, "test_images", "test1.jpg")
        out = args.output or os.path.join(MODULE_ROOT, "data", "test1_annotated.jpg")
        print(f"[+] Processing Single Road Image Frame: {os.path.basename(inp)}...")
        out_path = pipeline.process_image(inp, out)
        print(f"    - Annotated Frame with HUD saved to: {out_path}\n")

    if args.mode in ["batch", "all"]:
        test_images = sorted(glob.glob(os.path.join(MODULE_ROOT, "test_images", "test*.jpg")))[:4]
        out_dir = os.path.join(MODULE_ROOT, "data", "batch_output")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[+] Running Batch Processing across {len(test_images)} Test Frames...")
        for img_p in test_images:
            out_p = os.path.join(out_dir, os.path.basename(img_p))
            pipeline.process_image(img_p, out_p)
            print(f"    - Processed {os.path.basename(img_p)} -> {os.path.basename(out_p)}")
        print(f"[+] Batch outputs stored in: {out_dir}\n")

    if args.mode in ["video", "all"]:
        inp_v = args.input or os.path.join(MODULE_ROOT, "data", "demo_highway_drive.mp4")
        out_v = args.output or os.path.join(MODULE_ROOT, "data", "demo_highway_drive_annotated.mp4")
        if os.path.exists(inp_v):
            max_f = args.max_frames or (75 if args.mode == "all" else None)
            print(f"[+] Processing Highway Video Stream: {os.path.basename(inp_v)} (Frames: {max_f or 'All'})...")
            pipeline.process_video(inp_v, out_v, max_frames=max_f, show_progress=True)
            print(f"[+] Annotated Video with Dynamic HUD Telemetry saved to: {out_v}\n")
        else:
            print(f"[!] Video input not found at {inp_v}, skipping video test.\n")

    if args.mode in ["visuals"]:
        print("[+] Generating 300 DPI high-resolution visual documentation...")
        from generate_visuals import generate_architecture_diagram, generate_stage_breakdown
        generate_architecture_diagram()
        generate_stage_breakdown()
        print("[+] Visual documentation generated successfully in assets/docs/\n")

    print("=" * 80)
    print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
