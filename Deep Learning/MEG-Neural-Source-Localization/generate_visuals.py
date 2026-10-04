"""
Publication-Grade Visual Asset Generator for MEG Neural Source Localization.
Produces 300 DPI diagrams and benchmark performance plots.
Author: Abdul Rehman Rattu
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Apply clean publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "docs", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def plot_architecture_schematic():
    fig, ax = plt.subplots(figsize=(13, 6.8), dpi=300)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FFFFFF")

    modules = [
        {
            "x": 0.04, "y": 0.52, "w": 0.26, "h": 0.38,
            "title": "Biophysical Forward Model", "color": "#0284C7",
            "items": [
                "• 450 Cortical Brain Parcels (z)",
                "• Maxwell Lead Field Matrix (L)",
                "• 204 MEG Magnetic Helmet Sensors (x)",
                "• Governing Physics: x = Lz + n",
            ]
        },
        {
            "x": 0.37, "y": 0.52, "w": 0.26, "h": 0.38,
            "title": "Feature Inversion & Scaling", "color": "#059669",
            "items": [
                "• Subject-Specific Lead Fields (x1e8)",
                "• L-Norm Column Normalization",
                "• Sensor-to-Parcel Projection (z_est)",
                "• Multi-Subject Invariant Scaling",
            ]
        },
        {
            "x": 0.70, "y": 0.52, "w": 0.26, "h": 0.38,
            "title": "Inverse Modeling Solvers", "color": "#7C3AED",
            "items": [
                "• Multi-Output k-NN Baseline",
                "• Lasso-Lars L1 Sparse Regression",
                "• PyTorch MEGNet (3-Layer MLP)",
                "• Optuna Bayesian Hyperparameter Search",
            ]
        },
        {
            "x": 0.20, "y": 0.08, "w": 0.60, "h": 0.32,
            "title": "Dynamic Sparsity Decoding & Evaluation", "color": "#EA580C",
            "items": [
                "• Multi-Label Thresholding: Enforces <= 3 Active Cortical Neural Sources",
                "• Evaluation Metric: Jaccard Error = 1 - Jaccard Score (Multi-Label)",
                "• Empirical Best Result: Lasso-Lars (alpha=0.5) achieves 0.6586 Jaccard Error",
            ]
        },
    ]

    for m in modules:
        rect = patches.FancyBboxPatch(
            (m["x"], m["y"]), m["w"], m["h"],
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor="#FFFFFF", edgecolor=m["color"], linewidth=2.0, zorder=2
        )
        ax.add_patch(rect)

        # Header badge
        badge = patches.FancyBboxPatch(
            (m["x"], m["y"] + m["h"] - 0.08), m["w"], 0.08,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=m["color"], edgecolor="none", zorder=3
        )
        ax.add_patch(badge)
        ax.text(m["x"] + m["w"]/2, m["y"] + m["h"] - 0.04, m["title"],
                ha="center", va="center", color="#FFFFFF", fontsize=11, fontweight="bold", zorder=4)

        # Body items
        y_cursor = m["y"] + m["h"] - 0.14
        for item in m["items"]:
            ax.text(m["x"] + 0.02, y_cursor, item,
                    ha="left", va="center", color="#334155", fontsize=9.5, zorder=4)
            y_cursor -= 0.06

    # Arrows
    arrow_props = dict(facecolor="#64748B", edgecolor="none", width=2.0, headwidth=8.0, headlength=7.0)
    ax.annotate("", xy=(0.36, 0.71), xytext=(0.31, 0.71), arrowprops=arrow_props)
    ax.annotate("", xy=(0.69, 0.71), xytext=(0.64, 0.71), arrowprops=arrow_props)
    ax.annotate("", xy=(0.50, 0.41), xytext=(0.50, 0.51), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.title("MEG Neural Source Localization: Biophysical Inverse Pipeline",
              fontsize=14, fontweight="bold", pad=15, color="#0F172A")

    out_file = os.path.join(ASSETS_DIR, "meg_lead_field_architecture.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_lasso_regularization_path():
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    jaccard_errors = [0.9401, 0.8346, 0.8008, 0.6586, 1.0000]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    ax.plot(alphas, jaccard_errors, marker="o", color="#0284C7", linewidth=2.5, markersize=8, label="Lasso-Lars Inversion Path")
    ax.scatter([0.5], [0.6586], color="#DC2626", s=140, zorder=5, label="Optimal Sparsity (alpha = 0.5, Error = 0.6586)")

    ax.set_title("Lasso-Lars Regularization Path vs Jaccard Error", fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel("L1 Regularization Strength (Alpha)", fontsize=11, fontweight="medium", color="#1E293B")
    ax.set_ylabel("Jaccard Error (1 - Jaccard Score)", fontsize=11, fontweight="medium", color="#1E293B")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=10)

    out_file = os.path.join(ASSETS_DIR, "lasso_regularization_path.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_optuna_convergence():
    trials = list(range(20))
    objective_vals = [
        0.8355, 0.8248, 0.8978, 0.8672, 0.8350, 0.8582, 0.8483, 0.8539, 0.8308, 0.8177,
        0.8123, 0.8214, 0.8256, 0.8477, 0.8399, 0.8193, 0.8239, 0.8312, 0.8288, 0.8241
    ]
    best_vals = np.minimum.accumulate(objective_vals)

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    ax.scatter(trials, objective_vals, color="#2563EB", s=45, alpha=0.85, label="Trial Objective Value")
    ax.plot(trials, best_vals, color="#EA580C", linewidth=2.5, label="Best Value Convergence (0.8123)")

    ax.set_title("Optuna Bayesian Hyperparameter Optimization History (PyTorch MEGNet)",
                 fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel("Optimization Trial Number", fontsize=11, fontweight="medium", color="#1E293B")
    ax.set_ylabel("Jaccard Error Objective Value", fontsize=11, fontweight="medium", color="#1E293B")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=10)

    out_file = os.path.join(ASSETS_DIR, "optuna_convergence_history.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_hyperparameter_importance():
    params = ["learning_rate", "hidden_size", "dropout", "epochs", "positive_weight"]
    importances = [0.38, 0.36, 0.13, 0.09, 0.04]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    y_pos = np.arange(len(params))
    colors = ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]

    bars = ax.barh(y_pos, importances, color=colors, edgecolor="#1E40AF", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=10.5, color="#1E293B")
    ax.invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height()/2, f"{w:.2f}",
                ha="left", va="center", fontsize=9.5, fontweight="bold", color="#1E293B")

    ax.set_xlim(0, 0.45)
    ax.set_title("Optuna Hyperparameter Importance Scores (fANOVA Evaluation)",
                 fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xlabel("Normalized Relative Importance", fontsize=11, fontweight="medium", color="#1E293B")
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    out_file = os.path.join(ASSETS_DIR, "hyperparameter_importance.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def plot_model_comparison():
    models = [
        "k-NN (k=3)",
        "k-NN (k=5)",
        "k-NN (k=7)",
        "PyTorch MEGNet (Optuna)",
        "Lasso-Lars (alpha=0.5)"
    ]
    errors = [0.9140, 0.9120, 0.9108, 0.8123, 0.6586]
    colors = ["#94A3B8", "#64748B", "#475569", "#2563EB", "#059669"]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    bars = ax.bar(models, errors, color=colors, edgecolor="#1E293B", width=0.55)
    ax.set_ylabel("Jaccard Error (Lower is Better)", fontsize=11, fontweight="medium", color="#1E293B")
    ax.set_title("Model Benchmark Comparison on MEG Neural Source Localization",
                 fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0F172A")

    plt.xticks(rotation=15, ha="right", fontsize=9.5)
    out_file = os.path.join(ASSETS_DIR, "model_performance_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_file}")


def main():
    print("Generating 300 DPI publication assets for MEG Neural Source Localization...")
    plot_architecture_schematic()
    plot_lasso_regularization_path()
    plot_optuna_convergence()
    plot_hyperparameter_importance()
    plot_model_comparison()
    print("All visuals generated successfully.")


if __name__ == "__main__":
    main()
