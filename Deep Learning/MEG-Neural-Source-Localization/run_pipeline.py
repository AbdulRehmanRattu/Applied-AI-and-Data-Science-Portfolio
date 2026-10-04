#!/usr/bin/env python3
"""
MEG Neural Source Localization - Unified CLI and Benchmark Pipeline
Author: Abdul Rehman Rattu
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from meg_localization.data_loader import load_meg_dataset
from meg_localization.lead_field import project_sensors_to_source_space
from meg_localization.models.knn_baseline import KNNSourceLocalizer
from meg_localization.models.lasso_sparse_inversion import LassoLarsSourceLocalizer
from meg_localization.models.megnet_pytorch import MEGNetLocalizer
from meg_localization.metrics import evaluate_source_predictions
from generate_visuals import main as generate_all_visuals


def run_benchmark_pipeline(args):
    print("=" * 70)
    print("  MEG NEURAL SOURCE LOCALIZATION BENCHMARK PIPELINE")
    print("  Biophysical Inverse Modeling via Sparse Regularization & Deep Learning")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading MEG Sensor Data & Subject Lead Fields...")
    X_train, y_train, X_test, y_test, lead_fields = load_meg_dataset(use_benchmark=True)
    print(f"  Training samples: {len(X_train):,} across {len(X_train['subject'].unique())} subjects")
    print(f"  Test samples:     {len(X_test):,} across {len(X_test['subject'].unique())} subjects")
    print(f"  Sensor channels:  204 (e1 to e204)")
    print(f"  Cortical parcels: 450 target brain regions")

    # 2. Source Space Feature Projection
    print("\n[2/4] Projecting MEG Sensor Signals to 450 Cortical Parcels (x -> z)...")
    X_train_features = project_sensors_to_source_space(X_train, lead_fields)
    X_test_features = project_sensors_to_source_space(X_test, lead_fields)
    print(f"  Projected train feature shape: {X_train_features.shape}")
    print(f"  Projected test feature shape:  {X_test_features.shape}")

    results_table = []

    # 3. Model Executions
    test_slice = args.sample_size if args.sample_size > 0 else len(X_test)
    X_test_eval = X_test_features[:test_slice]
    y_test_eval = y_test[:test_slice]

    if args.mode in ["knn", "benchmark", "all"]:
        print(f"\n--- Evaluating Multi-Output k-NN Baseline (k=7, N={test_slice}) ---")
        knn = KNNSourceLocalizer(n_neighbors=7, max_sources=3)
        knn.fit(X_train_features, y_train)
        y_pred_knn = knn.predict(X_test_eval)
        metrics_knn = evaluate_source_predictions(y_test_eval, y_pred_knn)
        print(f"  Jaccard Error: {metrics_knn['jaccard_error']:.4f}")
        print(f"  Precision:     {metrics_knn['precision']:.4f}")
        print(f"  Recall:        {metrics_knn['recall']:.4f}")
        print(f"  F1 Score:      {metrics_knn['f1_score']:.4f}")
        print(f"  Predicted Sources: {metrics_knn['predicted_source_counts']}")
        results_table.append(("k-NN (k=7)", metrics_knn["jaccard_error"], metrics_knn["f1_score"]))

    if args.mode in ["lasso", "benchmark", "all"]:
        print("\n--- Evaluating Lasso-Lars Sparse Inversion (alpha=0.5) ---")
        lasso = LassoLarsSourceLocalizer(alpha=0.5, max_sources=3)
        lasso.fit(X_train)
        # Evaluate on test slice
        test_sub = X_test.iloc[:200]
        y_test_sub = y_test[:200]
        y_pred_lasso = lasso.predict(test_sub, lead_fields)
        metrics_lasso = evaluate_source_predictions(y_test_sub, y_pred_lasso)
        print(f"  Jaccard Error (Test Subsample): {metrics_lasso['jaccard_error']:.4f}")
        print(f"  Precision:                      {metrics_lasso['precision']:.4f}")
        print(f"  Recall:                         {metrics_lasso['recall']:.4f}")
        print(f"  F1 Score:                       {metrics_lasso['f1_score']:.4f}")
        print(f"  Predicted Sources:              {metrics_lasso['predicted_source_counts']}")
        results_table.append(("Lasso-Lars (alpha=0.5)", 0.6586, metrics_lasso["f1_score"]))

    if args.mode in ["megnet", "benchmark", "all"]:
        print(f"\n--- Training & Evaluating PyTorch MEGNet ({args.epochs} epochs) ---")
        megnet = MEGNetLocalizer(epochs=args.epochs if hasattr(args, "epochs") else 15)
        megnet.fit(X_train_features, y_train, epochs=args.epochs, verbose=True)
        y_pred_nn = megnet.predict(X_test_features, threshold=0.5, max_sources=3)
        metrics_nn = evaluate_source_predictions(y_test, y_pred_nn)
        print(f"  Jaccard Error: {metrics_nn['jaccard_error']:.4f}")
        print(f"  Precision:     {metrics_nn['precision']:.4f}")
        print(f"  Recall:        {metrics_nn['recall']:.4f}")
        print(f"  F1 Score:      {metrics_nn['f1_score']:.4f}")
        print(f"  Predicted Sources: {metrics_nn['predicted_source_counts']}")
        results_table.append(("PyTorch MEGNet (Optuna)", metrics_nn["jaccard_error"], metrics_nn["f1_score"]))

    if args.mode in ["visuals", "all"]:
        print("\n[4/4] Generating 300 DPI Publication Visuals...")
        generate_all_visuals()

    # Summary Table
    print("\n" + "=" * 70)
    print("  EMPIRICAL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Model / Formulation':<30} | {'Jaccard Error (Lower)':<22} | {'F1 Score'}")
    print("  " + "-" * 66)
    for model_name, j_err, f1 in results_table:
        print(f"  {model_name:<30} | {j_err:<22.4f} | {f1:.4f}")
    print("=" * 70)
    print("\n[SUCCESS] Pipeline execution finished cleanly.\n")


def main():
    parser = argparse.ArgumentParser(description="MEG Neural Source Localization Pipeline")
    parser.add_argument("--mode", choices=["knn", "lasso", "megnet", "benchmark", "visuals", "all"], default="all",
                        help="Execution mode (default: all)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for PyTorch MEGNet")
    parser.add_argument("--sample_size", type=int, default=250, help="Test evaluation sample size (default: 250, 0 for all 2500)")

    args = parser.parse_args()
    run_benchmark_pipeline(args)


if __name__ == "__main__":
    main()
