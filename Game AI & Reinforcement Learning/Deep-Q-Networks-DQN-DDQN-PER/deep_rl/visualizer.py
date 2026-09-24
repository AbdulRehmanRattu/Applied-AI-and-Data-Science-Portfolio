"""
Publication-Grade 300 DPI Visualization Generator for Deep RL Suite.
Produces high-resolution architectural schematics and empirical convergence figures.
Author: Abdul Rehman Rattu
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np


class RLVisualizer:
    """Generates 300 DPI publication-ready visualizations."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Apply publication style
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        })

    def generate_architecture_flowchart(self, save_filename: str = "deep_rl_pipeline_architecture.png") -> str:
        """
        Renders a 300 DPI white-themed system architecture schematic for DQN, DDQN, and PER.
        Features clean orthogonal routing, clear modular boundaries, and mathematical formulations.
        """
        fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 8.5)
        ax.axis("off")

        # Background canvas
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        # Color palette
        c_env = "#F0FDF4"      # light emerald
        c_env_b = "#16A34A"    # emerald border
        c_agent = "#EFF6FF"    # light blue
        c_agent_b = "#2563EB"  # blue border
        c_mem = "#FEF3C7"      # light amber
        c_mem_b = "#D97706"    # amber border
        c_net = "#F5F3FF"      # light purple
        c_net_b = "#7C3AED"    # purple border
        c_loss = "#FFF1F2"     # light rose
        c_loss_b = "#E11D48"   # rose border
        c_text = "#0F172A"     # slate 900
        c_muted = "#475569"    # slate 600

        # Title header
        ax.text(7.5, 8.15, "DEEP REINFORCEMENT LEARNING ARCHITECTURE (DQN / DDQN / PER)",
                ha="center", va="center", fontsize=15, fontweight="bold", color=c_text)
        ax.text(7.5, 7.82, "Closed-Loop Markov Decision Process (MDP) with Binary SumTree Prioritized Replay & Decoupled Q-Target Evaluation",
                ha="center", va="center", fontsize=10, color=c_muted)

        # 1. Environment Block (Left)
        rect_env = patches.FancyBboxPatch((0.5, 4.4), 3.2, 2.8, boxstyle="round,pad=0.15",
                                          edgecolor=c_env_b, facecolor=c_env, linewidth=2.0)
        ax.add_patch(rect_env)
        ax.text(2.1, 6.9, "Gymnasium Environment", ha="center", va="center", fontsize=11, fontweight="bold", color="#14532D")
        ax.text(2.1, 6.45, "CartPole-v1 / MountainCar-v0", ha="center", va="center", fontsize=9.5, fontweight="semibold", color=c_text)
        ax.text(2.1, 5.75, "State Dynamics Transition:\n"
                           r"$s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$" "\n"
                           "Engineered Reward Step",
                ha="center", va="center", fontsize=8.5, color=c_muted)
        ax.text(2.1, 4.8, r"Observation $s_t \in \mathbf{R}^d$" "\n" r"Action $a_t \in \mathcal{A}$",
                ha="center", va="center", fontsize=8.5, fontweight="medium", color="#15803D")

        # 2. Agent Policy Block (Center-Left)
        rect_agent = patches.FancyBboxPatch((4.3, 5.2), 3.2, 2.0, boxstyle="round,pad=0.15",
                                            edgecolor=c_agent_b, facecolor=c_agent, linewidth=2.0)
        ax.add_patch(rect_agent)
        ax.text(5.9, 6.9, "Behavior Policy", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E3A8A")
        ax.text(5.9, 6.4, r"$\epsilon$-Greedy Action Selection", ha="center", va="center", fontsize=9.5, fontweight="semibold", color=c_text)
        ax.text(5.9, 5.75, r"With prob $\epsilon$: explore random $a$" "\n"
                           r"With prob $1-\epsilon$: $\arg\max_a Q(s_t, a; \theta)$" "\n"
                           r"Anneal: $\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \lambda)$",
                ha="center", va="center", fontsize=8.2, color=c_muted)

        # 3. SumTree Replay Memory Block (Bottom-Left)
        rect_mem = patches.FancyBboxPatch((0.5, 0.7), 4.8, 3.1, boxstyle="round,pad=0.15",
                                          edgecolor=c_mem_b, facecolor=c_mem, linewidth=2.0)
        ax.add_patch(rect_mem)
        ax.text(2.9, 3.45, "Prioritized Experience Replay (PER)", ha="center", va="center", fontsize=11, fontweight="bold", color="#78350F")
        ax.text(2.9, 3.05, "Binary SumTree Data Structure", ha="center", va="center", fontsize=9.5, fontweight="semibold", color=c_text)
        ax.text(2.9, 2.3, r"Capacity $N = 20,000$ | $O(\log N)$ updates" "\n"
                           r"Sampling Probability: $P(i) = p_i^\alpha / \sum_k p_k^\alpha$" "\n"
                           r"Importance-Sampling Weight:" "\n"
                           r"$w_i = (N \cdot P(i))^{-\beta} / \max_j w_j \quad (\beta \to 1.0)$",
                ha="center", va="center", fontsize=8.2, color=c_muted)
        ax.text(2.9, 1.1, r"Transition: $(s_t, a_t, r_t, s_{t+1}, \mathrm{done}_t)$",
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="#B45309")

        # 4. Neural Network Blocks (Center-Right)
        # Primary Online Q-Network
        rect_q = patches.FancyBboxPatch((8.2, 4.6), 3.2, 2.6, boxstyle="round,pad=0.15",
                                        edgecolor=c_net_b, facecolor=c_net, linewidth=2.0)
        ax.add_patch(rect_q)
        ax.text(9.8, 6.85, "Primary Q-Network", ha="center", va="center", fontsize=11, fontweight="bold", color="#4C1D95")
        ax.text(9.8, 6.4, r"Online Parameters $\theta$", ha="center", va="center", fontsize=9.5, fontweight="semibold", color=c_text)
        ax.text(9.8, 5.7, "Architecture: MLP (64-64-ReLU)\n"
                           r"Estimates $Q(s_t, \cdot; \theta) \in \mathbf{R}^{|\mathcal{A}|}$" "\n"
                           "DDQN Action Selection:\n"
                           r"$a^* = \arg\max_{a'} Q(s_{t+1}, a'; \theta)$",
                ha="center", va="center", fontsize=8.2, color=c_muted)
        ax.text(9.8, 4.9, r"Optimizer: Adam ($\eta = 10^{-3}$)", ha="center", va="center", fontsize=8.5, fontweight="medium", color="#6D28D9")

        # Target Q-Network
        rect_tgt = patches.FancyBboxPatch((8.2, 0.7), 3.2, 2.6, boxstyle="round,pad=0.15",
                                          edgecolor="#0284C7", facecolor="#E0F2FE", linewidth=2.0)
        ax.add_patch(rect_tgt)
        ax.text(9.8, 2.95, "Target Q-Network", ha="center", va="center", fontsize=11, fontweight="bold", color="#0369A1")
        ax.text(9.8, 2.55, r"Target Parameters $\theta^-$", ha="center", va="center", fontsize=9.5, fontweight="semibold", color=c_text)
        ax.text(9.8, 1.85, "Evaluates Selected Action Value:\n"
                           r"$Q_{target} = Q(s_{t+1}, a^*; \theta^-)$" "\n"
                           "Decoupled Target: Prevents\n"
                           "Maximization Bias Overestimation",
                ha="center", va="center", fontsize=8.2, color=c_muted)
        ax.text(9.8, 1.05, r"Polyak Soft Update: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$",
                ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0284C7")

        # 5. Loss & TD-Error Optimization Block (Far Right)
        rect_loss = patches.FancyBboxPatch((12.0, 2.3), 2.5, 4.0, boxstyle="round,pad=0.15",
                                           edgecolor=c_loss_b, facecolor=c_loss, linewidth=2.0)
        ax.add_patch(rect_loss)
        ax.text(13.25, 5.95, "Loss & Optimization", ha="center", va="center", fontsize=11, fontweight="bold", color="#9F1239")
        ax.text(13.25, 5.45, r"Bellman Target $y_i$:", ha="center", va="center", fontsize=9.0, fontweight="semibold", color=c_text)
        ax.text(13.25, 4.95, r"$y_i = r_i + \gamma Q(s'_i, a^*; \theta^-)$" "\n" r"$\times (1 - \mathrm{done}_i)$",
                ha="center", va="center", fontsize=8.2, color="#BE123C")
        ax.text(13.25, 4.25, r"TD Error $\delta_i$:" "\n" r"$\delta_i = y_i - Q(s_i, a_i; \theta)$",
                ha="center", va="center", fontsize=8.5, fontweight="medium", color=c_text)
        ax.text(13.25, 3.4, "Weighted Huber Loss:\n"
                             r"$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B w_i \mathcal{H}(\delta_i)$",
                ha="center", va="center", fontsize=8.2, color=c_muted)
        ax.text(13.25, 2.65, r"Priority Update: $p_i \leftarrow |\delta_i|^\alpha$",
                ha="center", va="center", fontsize=8.2, fontweight="bold", color="#E11D48")

        # Connecting Arrows
        arrow_style = dict(arrowstyle="->", color="#334155", lw=1.6, mutation_scale=14)

        # Env -> Agent (State)
        ax.annotate("", xy=(4.3, 6.2), xytext=(3.7, 6.2), arrowprops=arrow_style)
        ax.text(4.0, 6.35, r"$s_t$", fontsize=9, fontweight="bold", color="#2563EB", ha="center")

        # Agent -> Env (Action)
        ax.annotate("", xy=(3.7, 5.4), xytext=(4.3, 5.4), arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.6, mutation_scale=14))
        ax.text(4.0, 5.15, r"$a_t$", fontsize=9, fontweight="bold", color="#16A34A", ha="center")

        # Env -> Replay Buffer (Experience)
        ax.annotate("", xy=(2.1, 3.8), xytext=(2.1, 4.4), arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.8, mutation_scale=14))
        ax.text(2.6, 4.1, r"$(s, a, r, s', d)$", fontsize=8.5, fontweight="bold", color="#D97706")

        # Replay Buffer -> Primary Network (Mini-batch)
        ax.annotate("", xy=(8.2, 5.4), xytext=(5.3, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=1.6, mutation_scale=14, connectionstyle="arc3,rad=-0.12"))
        ax.text(6.8, 4.2, r"Batch $(s_i, a_i, w_i)$", fontsize=8.5, fontweight="bold", color="#7C3AED")

        # Replay Buffer -> Target Network (Next states)
        ax.annotate("", xy=(8.2, 2.0), xytext=(5.3, 1.8),
                    arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.6, mutation_scale=14))
        ax.text(6.8, 1.5, r"Batch $(s'_i, r_i, d_i)$", fontsize=8.5, fontweight="bold", color="#0284C7")

        # Primary Q-Net -> Target Q-Net (Polyak Sync)
        ax.annotate("", xy=(9.8, 3.3), xytext=(9.8, 4.6),
                    arrowprops=dict(arrowstyle="->", color="#6366F1", lw=1.6, ls="--", mutation_scale=12))
        ax.text(10.5, 3.95, r"Soft Polyak $\tau$", fontsize=8, color="#6366F1")

        # Primary & Target -> Loss Block
        ax.annotate("", xy=(12.0, 5.0), xytext=(11.4, 5.5), arrowprops=arrow_style)
        ax.annotate("", xy=(12.0, 3.8), xytext=(11.4, 2.3), arrowprops=arrow_style)

        # Loss Block -> SumTree (Priority Feedback)
        ax.annotate("", xy=(4.5, 0.7), xytext=(13.25, 2.3),
                    arrowprops=dict(arrowstyle="->", color="#E11D48", lw=1.6, ls=":", mutation_scale=14, connectionstyle="arc3,rad=0.2"))
        ax.text(9.0, 0.4, r"Priority Feedback $p_i \leftarrow |\delta_i|^\alpha$", fontsize=8.5, fontweight="bold", color="#E11D48")

        save_path = os.path.join(self.output_dir, save_filename)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        return save_path

    def generate_empirical_benchmark_figure(
        self,
        telemetry_dir: str,
        save_filename: str = "empirical_convergence_benchmark.png",
    ) -> str:
        """
        Renders a 4-panel publication-grade empirical comparison figure of DQN, DDQN, and PER
        across CartPole-v1 and MountainCar-v0 using real laboratory empirical data.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
        fig.patch.set_facecolor("#FFFFFF")

        # High-contrast accessible color palette
        color_dqn = "#2563EB"   # Vivid Blue
        color_ddqn = "#D97706"  # Amber Orange
        color_per = "#059669"   # Emerald Green

        # --- Panel 1: CartPole 100-Episode Moving Average Reward ---
        ax1 = axes[0, 0]
        ax1.set_facecolor("#FAFAFA")
        ax1.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")

        cp_files = {
            "Standard DQN": (os.path.join(telemetry_dir, "cartpole_dqn.tsv"), color_dqn),
            "Double DQN (DDQN)": (os.path.join(telemetry_dir, "cartpole_ddqn.tsv"), color_ddqn),
            "Prioritized Replay (PER)": (os.path.join(telemetry_dir, "cartpole_per.tsv"), color_per),
        }

        for label, (p, col) in cp_files.items():
            if os.path.exists(p):
                df = pd.read_csv(p, sep=r"\s+", header=None)
                df.columns = ["episode", "reward", "avg_reward", "avg100_reward"]
                ax1.plot(df["episode"], df["avg100_reward"], label=f"{label} (Final: {df['avg100_reward'].iloc[-1]:.1f})",
                         color=col, linewidth=2.2)

        ax1.axhline(200.0, color="#DC2626", linestyle=":", linewidth=1.8, label="Solved Threshold (R = 200.0)")
        ax1.set_title("CartPole-v1: 100-Episode Rolling Average Reward", fontweight="bold", pad=10)
        ax1.set_xlabel("Training Episode")
        ax1.set_ylabel("Rolling Mean Reward (100 Ep)")
        ax1.set_ylim(0, 215)
        ax1.legend(loc="lower right", framealpha=0.95)

        # --- Panel 2: CartPole Raw Episodic Stability Comparison ---
        ax2 = axes[0, 1]
        ax2.set_facecolor("#FAFAFA")
        ax2.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")

        # Plot raw episode rewards with lower alpha and overlay rolling mean
        if os.path.exists(cp_files["Standard DQN"][0]):
            df_dqn = pd.read_csv(cp_files["Standard DQN"][0], sep=r"\s+", header=None)
            ax2.plot(df_dqn[0], df_dqn[1], color=color_dqn, alpha=0.25, linewidth=0.9)
            ax2.plot(df_dqn[0], df_dqn[2], color=color_dqn, linewidth=2.0, label="DQN Mean Return")

        if os.path.exists(cp_files["Prioritized Replay (PER)"][0]):
            df_per = pd.read_csv(cp_files["Prioritized Replay (PER)"][0], sep=r"\s+", header=None)
            ax2.plot(df_per[0], df_per[1], color=color_per, alpha=0.3, linewidth=0.9)
            ax2.plot(df_per[0], df_per[2], color=color_per, linewidth=2.2, label="PER Mean Return")

        ax2.set_title("CartPole-v1: Sample Efficiency & Convergence Speed", fontweight="bold", pad=10)
        ax2.set_xlabel("Training Episode")
        ax2.set_ylabel("Cumulative Episode Steps / Score")
        ax2.legend(loc="upper left", framealpha=0.95)

        # --- Panel 3: MountainCar-v0 Cumulative Reward Progression ---
        ax3 = axes[1, 0]
        ax3.set_facecolor("#FAFAFA")
        ax3.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")

        mc_files = {
            "Standard DQN": (os.path.join(telemetry_dir, "mountaincar_dqn.tsv"), color_dqn),
            "Double DQN (DDQN)": (os.path.join(telemetry_dir, "mountaincar_ddqn.tsv"), color_ddqn),
            "Prioritized Replay (PER)": (os.path.join(telemetry_dir, "mountaincar_per.tsv"), color_per),
        }

        for label, (p, col) in mc_files.items():
            if os.path.exists(p):
                df = pd.read_csv(p, sep=r"\s+", header=None)
                df.columns = ["episode", "ep_reward", "avg_reward", "car_position", "ep_steps"]
                ax3.plot(df["episode"], df["avg_reward"], label=f"{label} (Final: {df['avg_reward'].iloc[-1]:.1f})",
                         color=col, linewidth=2.2)

        ax3.set_title("MountainCar-v0: Average Cumulative Reward Trajectory", fontweight="bold", pad=10)
        ax3.set_xlabel("Training Episode")
        ax3.set_ylabel("Average Episodic Shaped Reward")
        ax3.legend(loc="lower right", framealpha=0.95)

        # --- Panel 4: MountainCar-v0 Maximum Car Position & Goal Achievement ---
        ax4 = axes[1, 1]
        ax4.set_facecolor("#FAFAFA")
        ax4.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")

        for label, (p, col) in mc_files.items():
            if os.path.exists(p):
                df = pd.read_csv(p, sep=r"\s+", header=None)
                df.columns = ["episode", "ep_reward", "avg_reward", "car_position", "ep_steps"]
                ax4.plot(df["episode"], df["car_position"], label=label, color=col, linewidth=1.8, marker="o", markersize=3.5)

        ax4.axhline(0.5, color="#DC2626", linestyle=":", linewidth=2.0, label="Flag Goal Line (x = 0.5)")
        ax4.set_title("MountainCar-v0: Maximum Hill Position per Episode", fontweight="bold", pad=10)
        ax4.set_xlabel("Training Episode")
        ax4.set_ylabel("Maximum Horizontal Position (x)")
        ax4.set_ylim(-0.2, 0.6)
        ax4.legend(loc="lower right", framealpha=0.95)

        plt.suptitle("EMPIRICAL REINFORCEMENT LEARNING BENCHMARK (DQN vs DDQN vs PER)",
                     fontsize=15, fontweight="bold", y=0.98, color="#0F172A")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        return save_path
