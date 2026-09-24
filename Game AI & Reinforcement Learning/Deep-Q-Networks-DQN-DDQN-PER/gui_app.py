"""
Interactive Deep Reinforcement Learning Telemetry & Policy Simulation GUI.
Visualizes CartPole-v1 balance control and MountainCar-v0 valley climb
alongside live real-time neural network Q-value distributions.
Author: Abdul Rehman Rattu
"""

import math
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch

from deep_rl import (
    DQNAgent,
    DDQNAgent,
    PERAgent,
    CartPoleEnvWrapper,
    MountainCarEnvWrapper,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")


class DeepRLDashboard(tk.Tk):
    """Modern Desktop Telemetry Dashboard for Deep RL Agents."""

    def __init__(self):
        super().__init__()
        self.title("Deep RL Autonomous Controller - Lab 20")
        self.geometry("960x700")
        self.minsize(880, 620)
        self.configure(bg="#0F172A")

        self.env_type = "cartpole"
        self.algo_type = "per"
        self.running = False
        self.step_delay = 0.02
        self.current_state = None
        self.current_step = 0
        self.episode_reward = 0.0

        self.env = None
        self.agent = None

        self._build_ui()
        self._init_simulation()

    def _build_ui(self):
        """Construct the UI layout."""
        # Top Header
        header = tk.Frame(self, bg="#1E293B", height=65)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="Deep Reinforcement Learning Telemetry Dashboard",
            font=("Helvetica", 16, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        title.pack(side=tk.LEFT, padx=20, pady=12)

        author = tk.Label(
            header,
            text="Author: Abdul Rehman Rattu | Lab 20",
            font=("Helvetica", 10),
            fg="#94A3B8",
            bg="#1E293B",
        )
        author.pack(side=tk.RIGHT, padx=20, pady=15)

        # Control Panel Ribbon
        control_ribbon = tk.Frame(self, bg="#334155", padx=15, pady=8)
        control_ribbon.pack(fill=tk.X)

        # Environment Selector
        tk.Label(control_ribbon, text="Environment:", fg="#F8FAFC", bg="#334155", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.env_var = tk.StringVar(value="CartPole-v1")
        env_cb = ttk.Combobox(control_ribbon, textvariable=self.env_var, values=["CartPole-v1", "MountainCar-v0"], state="readonly", width=14)
        env_cb.pack(side=tk.LEFT, padx=(0, 15))
        env_cb.bind("<<ComboboxSelected>>", self._on_env_change)

        # Algorithm Selector
        tk.Label(control_ribbon, text="Algorithm:", fg="#F8FAFC", bg="#334155", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.algo_var = tk.StringVar(value="Prioritized Replay (PER)")
        algo_cb = ttk.Combobox(
            control_ribbon,
            textvariable=self.algo_var,
            values=["Standard DQN", "Double DQN (DDQN)", "Prioritized Replay (PER)"],
            state="readonly",
            width=22,
        )
        algo_cb.pack(side=tk.LEFT, padx=(0, 15))
        algo_cb.bind("<<ComboboxSelected>>", self._on_algo_change)

        # Action Buttons
        self.btn_run = tk.Button(
            control_ribbon,
            text="▶ Run Episode",
            command=self._start_single_episode,
            bg="#2563EB",
            fg="white",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
        )
        self.btn_run.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(
            control_ribbon,
            text="⏹ Stop / Reset",
            command=self._reset_sim,
            bg="#DC2626",
            fg="white",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Main Visualization Layout (Split left canvas, right telemetry)
        main_body = tk.Frame(self, bg="#0F172A")
        main_body.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left Canvas for Physics Rendering
        canvas_container = tk.Frame(main_body, bg="#1E293B", bd=1, relief=tk.SOLID)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        canvas_header = tk.Label(canvas_container, text="Real-Time Kinematic Simulation", font=("Helvetica", 11, "bold"), fg="#93C5FD", bg="#1E293B")
        canvas_header.pack(anchor="w", padx=15, pady=10)

        self.canvas = tk.Canvas(canvas_container, bg="#020617", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Right Panel for Q-Values & Metrics Telemetry
        telemetry_panel = tk.Frame(main_body, bg="#1E293B", width=340, bd=1, relief=tk.SOLID)
        telemetry_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        telemetry_panel.pack_propagate(False)

        q_header = tk.Label(telemetry_panel, text="Neural Network Q-Value Distribution", font=("Helvetica", 11, "bold"), fg="#FCD34D", bg="#1E293B")
        q_header.pack(anchor="w", padx=15, pady=10)

        # Q-Value Bar Visualizer Canvas
        self.q_canvas = tk.Canvas(telemetry_panel, bg="#020617", height=180, highlightthickness=0)
        self.q_canvas.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Telemetry Labels
        tk.Label(telemetry_panel, text="Live Controller State:", font=("Helvetica", 10, "bold"), fg="#94A3B8", bg="#1E293B").pack(anchor="w", padx=15, pady=(5, 5))

        self.lbl_step = tk.Label(telemetry_panel, text="Step: 0", font=("Helvetica", 11), fg="#F8FAFC", bg="#1E293B")
        self.lbl_step.pack(anchor="w", padx=25)

        self.lbl_score = tk.Label(telemetry_panel, text="Cumulative Reward: 0.0", font=("Helvetica", 11), fg="#38BDF8", bg="#1E293B")
        self.lbl_score.pack(anchor="w", padx=25)

        self.lbl_pos = tk.Label(telemetry_panel, text="Position: 0.000", font=("Helvetica", 11), fg="#4ADE80", bg="#1E293B")
        self.lbl_pos.pack(anchor="w", padx=25)

        self.lbl_vel = tk.Label(telemetry_panel, text="Velocity: 0.000", font=("Helvetica", 11), fg="#F472B6", bg="#1E293B")
        self.lbl_vel.pack(anchor="w", padx=25)

        self.lbl_action = tk.Label(telemetry_panel, text="Selected Action: None", font=("Helvetica", 11, "bold"), fg="#A78BFA", bg="#1E293B")
        self.lbl_action.pack(anchor="w", padx=25, pady=(5, 0))

        # Bottom Information Box
        info_frame = tk.Frame(telemetry_panel, bg="#0B132B", padx=10, pady=10, relief=tk.GROOVE)
        info_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=15)

        tk.Label(info_frame, text="Optimization Engine:", font=("Helvetica", 9, "bold"), fg="#60A5FA", bg="#0B132B").pack(anchor="w")
        self.lbl_info = tk.Label(
            info_frame,
            text="Prioritized Replay (SumTree)\nDouble Q-Target Estimation\nSmooth L1 Huber Loss",
            font=("Helvetica", 8),
            fg="#94A3B8",
            bg="#0B132B",
            justify=tk.LEFT,
        )
        self.lbl_info.pack(anchor="w", pady=(3, 0))

    def _on_env_change(self, event=None):
        val = self.env_var.get()
        if "MountainCar" in val:
            self.env_type = "mountaincar"
        else:
            self.env_type = "cartpole"
        self._init_simulation()

    def _on_algo_change(self, event=None):
        val = self.algo_var.get()
        if "Double" in val:
            self.algo_type = "ddqn"
        elif "Prioritized" in val:
            self.algo_type = "per"
        else:
            self.algo_type = "dqn"
        self._init_simulation()

    def _init_simulation(self):
        """Initialize environment and agent instances."""
        self.running = False
        if self.env_type == "cartpole":
            self.env = CartPoleEnvWrapper()
            if self.algo_type == "dqn":
                self.agent = DQNAgent(self.env.state_dim, self.env.action_dim)
            elif self.algo_type == "ddqn":
                self.agent = DDQNAgent(self.env.state_dim, self.env.action_dim)
            else:
                self.agent = PERAgent(self.env.state_dim, self.env.action_dim)

            # Check if saved PyTorch weights exist
            pt_path = os.path.join(WEIGHTS_DIR, "cartpole_per_pytorch.pth")
            if os.path.exists(pt_path):
                try:
                    self.agent.load_checkpoint(pt_path)
                except Exception:
                    pass
        else:
            self.env = MountainCarEnvWrapper()
            if self.algo_type == "dqn":
                self.agent = DQNAgent(self.env.state_dim, self.env.action_dim)
            elif self.algo_type == "ddqn":
                self.agent = DDQNAgent(self.env.state_dim, self.env.action_dim)
            else:
                self.agent = PERAgent(self.env.state_dim, self.env.action_dim)

        self._reset_sim()

    def _reset_sim(self):
        """Reset environment state and update display."""
        self.running = False
        self.current_state = self.env.reset()
        self.current_step = 0
        self.episode_reward = 0.0
        self._draw_canvas()
        self._update_telemetry(0, [0.0] * self.env.action_dim)

    def _start_single_episode(self):
        if not self.running:
            self.running = True
            self.current_state = self.env.reset()
            self.current_step = 0
            self.episode_reward = 0.0
            self._step_simulation()

    def _step_simulation(self):
        if not self.running:
            return

        state = self.current_state
        # Compute Q-values for telemetry
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.agent.device)
            q_vals = self.agent.primary_network(s_tensor).cpu().numpy().flatten()

        action = int(np.argmax(q_vals))

        if self.env_type == "cartpole":
            next_state, reward, done, _ = self.env.step(action, shape_reward=False)
            self.current_step += 1
            self.episode_reward += 1.0
            if self.current_step >= 500:
                done = True
        else:
            next_state, reward, done, _ = self.env.step(action, state, shape_reward=False)
            self.current_step += 1
            self.episode_reward += reward
            if self.current_step >= 200:
                done = True

        self.current_state = next_state
        self._draw_canvas()
        self._update_telemetry(action, q_vals)

        if done:
            self.running = False
            return

        self.after(int(self.step_delay * 1000), self._step_simulation)

    def _draw_canvas(self):
        """Render physics visualization on Tkinter canvas."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10:
            w, h = 500, 400

        if self.env_type == "cartpole":
            # CartPole Rendering
            cart_x = self.current_state[0]
            pole_angle = self.current_state[2]

            # Scale coordinates
            track_y = h * 0.7
            cx = w / 2 + (cart_x / 2.4) * (w / 2 * 0.8)
            cart_w, cart_h = 70, 35

            # Draw Track
            self.canvas.create_line(30, track_y, w - 30, track_y, fill="#475569", width=3)

            # Draw Cart
            self.canvas.create_rectangle(
                cx - cart_w / 2, track_y - cart_h, cx + cart_w / 2, track_y,
                fill="#3B82F6", outline="#60A5FA", width=2
            )

            # Draw Wheels
            self.canvas.create_oval(cx - 25, track_y - 4, cx - 11, track_y + 10, fill="#1E293B", outline="#94A3B8")
            self.canvas.create_oval(cx + 11, track_y - 4, cx + 25, track_y + 10, fill="#1E293B", outline="#94A3B8")

            # Draw Pole
            pole_len = 110
            tip_x = cx + pole_len * math.sin(pole_angle)
            tip_y = (track_y - cart_h / 2) - pole_len * math.cos(pole_angle)

            self.canvas.create_line(
                cx, track_y - cart_h / 2, tip_x, tip_y,
                fill="#F59E0B", width=6, capstyle=tk.ROUND
            )
            # Pole Tip Bulb
            self.canvas.create_oval(tip_x - 5, tip_y - 5, tip_x + 5, tip_y + 5, fill="#EF4444", outline="#FCA5A5")

            # Center Axle Pin
            self.canvas.create_oval(cx - 4, track_y - cart_h / 2 - 4, cx + 4, track_y - cart_h / 2 + 4, fill="#FFFFFF")

        else:
            # MountainCar Rendering
            car_x = self.current_state[0]  # -1.2 to 0.6
            # Sinusoidal valley height function: y = sin(3x)
            pts = []
            for px in range(20, w - 20, 4):
                sim_x = -1.2 + (px - 20) / (w - 40) * 1.8
                sim_y = math.sin(3 * sim_x)
                py = h * 0.65 - sim_y * (h * 0.25)
                pts.extend([px, py])

            self.canvas.create_line(pts, fill="#334155", width=4, smooth=True)

            # Draw Goal Flag at x = 0.5
            flag_sim_x = 0.5
            flag_px = 20 + ((flag_sim_x - (-1.2)) / 1.8) * (w - 40)
            flag_py = h * 0.65 - math.sin(3 * flag_sim_x) * (h * 0.25)
            self.canvas.create_line(flag_px, flag_py, flag_px, flag_py - 40, fill="#FFFFFF", width=2)
            self.canvas.create_polygon([flag_px, flag_py - 40, flag_px + 18, flag_py - 30, flag_px, flag_py - 20], fill="#EF4444")

            # Draw Car on Valley Curve
            car_px = 20 + ((car_x - (-1.2)) / 1.8) * (w - 40)
            car_py = h * 0.65 - math.sin(3 * car_x) * (h * 0.25)

            # Car slope angle
            slope = 3 * math.cos(3 * car_x)
            angle = math.atan(slope)

            # Draw oriented rectangular car
            c_w, c_h = 24, 12
            dx = (c_w / 2) * math.cos(angle)
            dy = -(c_w / 2) * math.sin(angle)
            self.canvas.create_line(car_px - dx, car_py - dy - 6, car_px + dx, car_py + dy - 6, fill="#38BDF8", width=10, capstyle=tk.ROUND)

    def _update_telemetry(self, selected_action: int, q_vals: np.ndarray):
        """Update telemetry text and Q-value bar graph."""
        self.lbl_step.config(text=f"Step: {self.current_step}")
        self.lbl_score.config(text=f"Episodic Score: {self.episode_reward:.1f}")

        if self.env_type == "cartpole":
            self.lbl_pos.config(text=f"Cart X: {self.current_state[0]:+.3f}")
            self.lbl_vel.config(text=f"Pole Angle: {self.current_state[2]:+.3f} rad")
            action_names = ["Push Left (←)", "Push Right (→)"]
        else:
            self.lbl_pos.config(text=f"Position X: {self.current_state[0]:+.3f}")
            self.lbl_vel.config(text=f"Velocity: {self.current_state[1]:+.4f}")
            action_names = ["Acc Left (←)", "Coast (•)", "Acc Right (→)"]

        act_text = action_names[selected_action] if selected_action < len(action_names) else f"Act {selected_action}"
        self.lbl_action.config(text=f"Selected: {act_text}")

        # Draw Q-values on bar canvas
        self.q_canvas.delete("all")
        qw = self.q_canvas.winfo_width()
        qh = self.q_canvas.winfo_height()
        if qw < 10:
            qw, qh = 300, 180

        num_acts = len(q_vals)
        bar_w = (qw - 40) / max(1, num_acts)
        max_q = max(0.1, float(np.max(np.abs(q_vals))))

        for i, q in enumerate(q_vals):
            bx = 20 + i * bar_w
            # Normalize bar height relative to canvas center
            bar_height = (abs(q) / max_q) * (qh * 0.35)
            y_base = qh * 0.55

            if q >= 0:
                y1 = y_base - bar_height
                y2 = y_base
                col = "#10B981" if i == selected_action else "#3B82F6"
            else:
                y1 = y_base
                y2 = y_base + bar_height
                col = "#EF4444"

            self.q_canvas.create_rectangle(bx + 4, y1, bx + bar_w - 4, y2, fill=col, outline="#FFFFFF", width=1)
            # Label action name & Q value
            act_name = action_names[i] if i < len(action_names) else f"A{i}"
            self.q_canvas.create_text(bx + bar_w / 2, qh * 0.88, text=f"{act_name}\nQ={q:.2f}", fill="#F8FAFC", font=("Helvetica", 8), justify=tk.CENTER)


def main():
    app = DeepRLDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
