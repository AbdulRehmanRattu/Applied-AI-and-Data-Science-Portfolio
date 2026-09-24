#!/usr/bin/env python3
"""
Autonomous Vehicle Lane Detection Desktop GUI
=============================================
Modern desktop GUI built with Tkinter and Pillow for interactive
video and single-frame lane detection and curvature telemetry.

Author: Abdul Rehman Rattu
License: MIT
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Ensure local module access
MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODULE_ROOT)

from lane_detection.pipeline import LaneDetectionPipeline


class LaneDetectionGUI:
    """Desktop GUI interface for autonomous vehicle lane detection."""

    def __init__(self, master):
        self.master = master
        master.title("Autonomous Vehicle Curved Lane Line Detection & Telemetry")
        master.geometry("780x560")
        master.resizable(False, False)

        # Style configuration
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.pipeline = None
        self._init_ui()

    def _init_ui(self):
        # Header banner
        header_frame = tk.Frame(self.master, bg="#0F172A", height=90)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_lbl = tk.Label(
            header_frame,
            text="AUTONOMOUS VEHICLE LANE DETECTION",
            font=("Helvetica", 16, "bold"),
            fg="#38BDF8",
            bg="#0F172A"
        )
        title_lbl.pack(pady=(18, 4))

        subtitle_lbl = tk.Label(
            header_frame,
            text="Advanced Driver Assistance System (ADAS) Computer Vision Pipeline",
            font=("Helvetica", 10),
            fg="#94A3B8",
            bg="#0F172A"
        )
        subtitle_lbl.pack()

        # Main content card
        content_frame = ttk.Frame(self.master, padding=25)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Input Selection
        input_lbl = ttk.Label(content_frame, text="Input Media (Video or Image):", font=("Helvetica", 10, "bold"))
        input_lbl.grid(row=0, column=0, sticky=tk.W, pady=(0, 6))

        self.input_entry = ttk.Entry(content_frame, width=62)
        default_video = os.path.join(MODULE_ROOT, "data", "demo_highway_drive.mp4")
        if os.path.exists(default_video):
            self.input_entry.insert(0, default_video)
        self.input_entry.grid(row=1, column=0, padx=(0, 10), pady=(0, 15), ipady=3)

        browse_in_btn = ttk.Button(content_frame, text="Browse...", command=self._browse_input)
        browse_in_btn.grid(row=1, column=1, pady=(0, 15))

        # Output Destination
        output_lbl = ttk.Label(content_frame, text="Output Annotated Path:", font=("Helvetica", 10, "bold"))
        output_lbl.grid(row=2, column=0, sticky=tk.W, pady=(0, 6))

        self.output_entry = ttk.Entry(content_frame, width=62)
        default_out = os.path.join(MODULE_ROOT, "data", "demo_highway_drive_annotated.mp4")
        self.output_entry.insert(0, default_out)
        self.output_entry.grid(row=3, column=0, padx=(0, 10), pady=(0, 20), ipady=3)

        browse_out_btn = ttk.Button(content_frame, text="Browse...", command=self._browse_output)
        browse_out_btn.grid(row=3, column=1, pady=(0, 20))

        # Progress bar
        self.progress = ttk.Progressbar(content_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))

        # Status Label
        self.status_lbl = ttk.Label(content_frame, text="Status: Ready to process.", font=("Helvetica", 9), foreground="#475569")
        self.status_lbl.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(0, 25))

        # Action Buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=6, column=0, columnspan=2)

        self.start_btn = tk.Button(
            btn_frame,
            text="Start Processing",
            command=self._start_thread,
            bg="#2563EB",
            fg="white",
            font=("Helvetica", 11, "bold"),
            padx=20,
            pady=8,
            relief=tk.FLAT
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)

        cal_btn = ttk.Button(btn_frame, text="Verify Calibration", command=self._verify_calibration)
        cal_btn.pack(side=tk.LEFT, padx=10)

    def _browse_input(self):
        f = filedialog.askopenfilename(
            title="Select Driving Video or Image",
            filetypes=[("Media Files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        if f:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, f)
            # Auto set output path
            base, ext = os.path.splitext(f)
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, f"{base}_annotated{ext}")

    def _browse_output(self):
        f = filedialog.asksaveasfilename(
            title="Select Output Destination",
            filetypes=[("MP4 Video", "*.mp4"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        if f:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, f)

    def _verify_calibration(self):
        try:
            if not self.pipeline:
                self.pipeline = LaneDetectionPipeline()
            mtx = self.pipeline.calibration.mtx
            fx, fy = mtx[0, 0], mtx[1, 1]
            messagebox.showinfo(
                "Camera Calibration Verified",
                f"Camera Intrinsic Matrix K Loaded Successfully:\n\n"
                f"Focal Length fx: {fx:.2f} px\n"
                f"Focal Length fy: {fy:.2f} px\n"
                f"Principal Point (cx, cy): ({mtx[0,2]:.1f}, {mtx[1,2]:.1f})\n\n"
                f"Distortion coefficients cached and ready."
            )
        except Exception as e:
            messagebox.showerror("Calibration Error", str(e))

    def _start_thread(self):
        self.start_btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="Status: Initializing pipeline...", foreground="#0284C7")
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        try:
            in_path = self.input_entry.get().strip()
            out_path = self.output_entry.get().strip()

            if not os.path.exists(in_path):
                messagebox.showerror("File Error", f"Input file does not exist: {in_path}")
                self.start_btn.config(state=tk.NORMAL)
                return

            if not self.pipeline:
                self.pipeline = LaneDetectionPipeline()

            lower = in_path.lower()
            if lower.endswith((".mp4", ".avi", ".mov")):
                self.status_lbl.config(text="Status: Processing video stream frame-by-frame...")
                self.pipeline.process_video(in_path, out_path, max_frames=125, show_progress=False)
            else:
                self.status_lbl.config(text="Status: Processing image frame...")
                self.pipeline.process_image(in_path, out_path)

            self.status_lbl.config(text=f"Status: Complete! Saved to {os.path.basename(out_path)}", foreground="#16A34A")
            messagebox.showinfo("Success", f"Processing complete!\nSaved to:\n{out_path}")
        except Exception as e:
            self.status_lbl.config(text=f"Error: {str(e)}", foreground="#DC2626")
            messagebox.showerror("Execution Error", str(e))
        finally:
            self.start_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = LaneDetectionGUI(root)
    root.mainloop()
