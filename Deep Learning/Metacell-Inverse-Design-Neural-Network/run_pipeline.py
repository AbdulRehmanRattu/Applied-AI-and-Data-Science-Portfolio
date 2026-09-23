#!/usr/bin/env python3
"""
Metacell AI: Unified CLI Pipeline Entry Point
==============================================
Physics-informed deep surrogate neural networks for forward prediction
and inverse design of large-phase-shift metacells in computational electromagnetics.

Author: Abdul Rehman Rattu
License: MIT
"""

import argparse
import os
import sys
import torch
import numpy as np

# Ensure local package path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metacell_ai.data_loader import MetacellDataLoader
from metacell_ai.models import ForwardSurrogateNetwork, InverseSynthesisNetwork, train_metacell_model
from metacell_ai.evaluator import MetacellEvaluator


def run_pipeline(mode="all", target_phase=-75.0, target_mag=0.90, epochs=200, generate_visuals=False):
    print("=" * 80)
    print("   METACELL DEEP SURROGATE & INVERSE DESIGN NEURAL NETWORK PIPELINE")
    print("   Computational Electromagnetics | 728 Full-Wave CST Simulations (1.952 GHz)")
    print("=" * 80)

    # 1. Load Data
    data_loader = MetacellDataLoader()
    X_train_f, y_train_f, X_test_f, y_test_f = data_loader.get_forward_data(as_tensors=True)
    X_train_i, y_train_i, X_test_i, y_test_i = data_loader.get_inverse_data(as_tensors=True)

    print(f"[+] Loaded Dataset: {len(data_loader.df)} finite-element simulation samples.")
    print(f"    - Training Set: {len(X_train_f)} samples (80%)")
    print(f"    - Holdout Validation Set: {len(X_test_f)} samples (20%)\n")

    torch.manual_seed(42)
    forward_model = ForwardSurrogateNetwork()
    inverse_model = InverseSynthesisNetwork()

    if mode in ["train", "all"]:
        print("[+] Training Forward Surrogate Model (Geometry -> Scattering Parameters)...")
        f_hist = train_metacell_model(forward_model, X_train_f, y_train_f, X_test_f, y_test_f, epochs=epochs, lr=0.002)
        print(f"    - Forward Final Validation Loss: {f_hist['val_loss'][-1]:.6f}")

        print("\n[+] Training Inverse Synthesis Model (Target Specs -> Optimal Geometry)...")
        i_hist = train_metacell_model(inverse_model, X_train_i, y_train_i, X_test_i, y_test_i, epochs=epochs, lr=0.002)
        print(f"    - Inverse Final Validation Loss: {i_hist['val_loss'][-1]:.6f}\n")

    if mode in ["evaluate", "all"]:
        print("[+] Evaluating Forward Model Accuracy on 20% Unseen Test Set...")
        f_metrics = MetacellEvaluator.evaluate_model(forward_model, X_test_f, y_test_f, data_loader, is_forward=True)
        MetacellEvaluator.print_metrics_table("Forward Surrogate Model", f_metrics)

        print("[+] Evaluating Inverse Model Accuracy on 20% Unseen Test Set...")
        i_metrics = MetacellEvaluator.evaluate_model(inverse_model, X_test_i, y_test_i, data_loader, is_forward=False)
        MetacellEvaluator.print_metrics_table("Inverse Design Model", i_metrics)

    if mode in ["inverse-design", "all"]:
        print("=" * 80)
        print(f"   INVERSE METACELL SYNTHESIS FOR TARGET SPECIFICATIONS")
        print("=" * 80)
        print(f"Target Transmission Magnitude |S21|: {target_mag:.2f}")
        print(f"Target Transmission Phase Shift Φ:   {target_phase:.1f}°\n")

        scaled_input = data_loader.scale_em_input([target_mag, target_phase])
        inverse_model.eval()
        with torch.no_grad():
            synth_geom_scaled = inverse_model(scaled_input)
            synth_geom = data_loader.inverse_transform_geom(synth_geom_scaled)[0]

        print("Synthesized Metacell Micro-Geometry Parameters:")
        print(f"  * C1 (Top Capacitive Patch):       {synth_geom[0]:.3f} mm/pF")
        print(f"  * C2 (Intermediate Substrate):     {synth_geom[1]:.3f} mm/pF")
        print(f"  * C3 (Bottom Ground Coupled Res.): {synth_geom[2]:.3f} mm/pF\n")

        # Forward verification
        scaled_synth_geom = data_loader.scale_geom_input(synth_geom)
        forward_model.eval()
        with torch.no_grad():
            verified_em_scaled = forward_model(scaled_synth_geom)
            verified_em = data_loader.inverse_transform_em(verified_em_scaled)[0]

        print("Closed-Loop Forward Verification of Synthesized Geometry:")
        print(f"  * Predicted Magnitude |S21|:       {verified_em[0]:.3f} (Error: {abs(verified_em[0] - target_mag):.3f})")
        print(f"  * Predicted Phase Shift Φ:         {verified_em[1]:.1f}° (Error: {abs(verified_em[1] - target_phase):.1f}°)")
        print("=" * 80 + "\n")

    if generate_visuals:
        print("[+] Generating 300 DPI visual documentation assets...")
        from generate_visuals import generate_architecture_diagram, generate_parity_benchmarks
        os.makedirs("docs/assets", exist_ok=True)
        generate_architecture_diagram()
        generate_parity_benchmarks()
        print("[+] All assets generated successfully in docs/assets/\n")


def main():
    parser = argparse.ArgumentParser(
        description="Metacell Inverse Design & Forward Surrogate Deep Neural Network Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "inverse-design", "all"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    parser.add_argument(
        "--target-phase",
        type=float,
        default=-75.0,
        help="Target transmission phase shift in degrees for inverse synthesis (default: -75.0)"
    )
    parser.add_argument(
        "--target-mag",
        type=float,
        default=0.90,
        help="Target transmission magnitude for inverse synthesis (default: 0.90)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--generate-visuals",
        action="store_true",
        help="Generate 300 DPI publication-grade visual assets in docs/assets/"
    )

    args = parser.parse_args()
    run_pipeline(
        mode=args.mode,
        target_phase=args.target_phase,
        target_mag=args.target_mag,
        epochs=args.epochs,
        generate_visuals=args.generate_visuals
    )


if __name__ == '__main__':
    main()
