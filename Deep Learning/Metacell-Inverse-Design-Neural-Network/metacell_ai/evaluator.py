"""
Evaluation and Parity Metrics for Metacell Deep Learning Pipeline
==================================================================
Computes R^2, MSE, RMSE, and MAE across electromagnetic scattering
parameters and synthesized structural geometry against CST ground truth.
"""

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MetacellEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test, data_loader, is_forward=True):
        """
        Evaluate surrogate model predictions in true unscaled physical units.
        """
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_test).cpu().numpy()
            y_true_scaled = y_test.cpu().numpy()
        
        if is_forward:
            preds_raw = data_loader.inverse_transform_em(preds_scaled)
            y_true_raw = data_loader.inverse_transform_em(y_true_scaled)
            var_names = ["Transmission Magnitude |S21|", "Transmission Phase Shift Φ (deg)"]
        else:
            preds_raw = data_loader.inverse_transform_geom(preds_scaled)
            y_true_raw = data_loader.inverse_transform_geom(y_true_scaled)
            var_names = ["Geometric C1 (mm/pF)", "Geometric C2 (mm/pF)", "Geometric C3 (mm/pF)"]
        
        metrics = {}
        for i, name in enumerate(var_names):
            y_t = y_true_raw[:, i]
            y_p = preds_raw[:, i]
            r2 = r2_score(y_t, y_p)
            mse = mean_squared_error(y_t, y_p)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_t, y_p)
            
            metrics[name] = {
                "R2": r2,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "y_true": y_t,
                "y_pred": y_p
            }
        
        return metrics

    @staticmethod
    def print_metrics_table(title, metrics):
        print("=" * 80)
        print(f"   {title.upper()} - EMPIRICAL BENCHMARK METRICS")
        print("=" * 80)
        print(f"{'Target Variable':<38} | {'R² Score':<10} | {'RMSE':<12} | {'MAE':<12}")
        print("-" * 80)
        for var, m in metrics.items():
            print(f"{var:<38} | {m['R2']:<10.4f} | {m['RMSE']:<12.4f} | {m['MAE']:<12.4f}")
        print("=" * 80 + "\n")
