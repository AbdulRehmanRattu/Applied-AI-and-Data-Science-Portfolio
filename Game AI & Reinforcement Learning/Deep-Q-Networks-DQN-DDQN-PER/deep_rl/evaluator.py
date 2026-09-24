"""
Evaluation and Benchmarking Engine for Deep RL Algorithms.
Provides statistical evaluation, validation episode execution, and empirical metric aggregation.
Author: Abdul Rehman Rattu
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np
import torch
from deep_rl.environments import CartPoleEnvWrapper, MountainCarEnvWrapper


class RLEvaluator:
    """Benchmark runner and statistical analyzer for DQN, DDQN, and PER agents."""

    @staticmethod
    def evaluate_cartpole_agent(agent, num_episodes: int = 20) -> Dict[str, float]:
        """Run validation episodes on CartPole-v1 without exploration."""
        env = CartPoleEnvWrapper()
        scores: List[float] = []

        for _ in range(num_episodes):
            state = env.reset()
            done = False
            ep_steps = 0
            while not done and ep_steps < env.max_episode_steps:
                action = agent.select_action(state, evaluate=True)
                next_state, _, done, _ = env.step(action, shape_reward=False)
                state = next_state
                ep_steps += 1
            scores.append(float(ep_steps))

        env.close()
        return {
            "mean_reward": float(np.mean(scores)),
            "std_reward": float(np.std(scores)),
            "min_reward": float(np.min(scores)),
            "max_reward": float(np.max(scores)),
            "success_rate": float(np.mean([s >= 200.0 for s in scores]) * 100.0),
        }

    @staticmethod
    def evaluate_mountaincar_agent(agent, num_episodes: int = 15) -> Dict[str, float]:
        """Run validation episodes on MountainCar-v0 without exploration."""
        env = MountainCarEnvWrapper()
        steps_list: List[int] = []
        max_positions: List[float] = []
        successes: List[bool] = []

        for _ in range(num_episodes):
            state = env.reset()
            done = False
            step_cnt = 0
            max_pos = float(state[0])

            while not done and step_cnt < env.max_episode_steps:
                action = agent.select_action(state, evaluate=True)
                next_state, _, done, _ = env.step(action, state, shape_reward=False)
                step_cnt += 1
                state = next_state
                if state[0] > max_pos:
                    max_pos = float(state[0])
                if state[0] >= 0.5:
                    break

            steps_list.append(step_cnt)
            max_positions.append(max_pos)
            successes.append(max_pos >= 0.5)

        env.close()
        return {
            "success_rate": float(np.mean(successes) * 100.0),
            "mean_steps": float(np.mean(steps_list)),
            "mean_max_position": float(np.mean(max_positions)),
            "best_steps": int(np.min(steps_list)),
        }

    @staticmethod
    def load_telemetry_metrics(telemetry_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate empirical performance metrics from original laboratory telemetry files.
        """
        metrics: Dict[str, Dict[str, Any]] = {}

        # 1. CartPole Telemetry
        cartpole_files = {
            "DQN": os.path.join(telemetry_dir, "cartpole_dqn.tsv"),
            "DDQN": os.path.join(telemetry_dir, "cartpole_ddqn.tsv"),
            "PER": os.path.join(telemetry_dir, "cartpole_per.tsv"),
        }

        for name, path in cartpole_files.items():
            if os.path.exists(path):
                df = pd.read_csv(path, sep=r"\s+", header=None)
                df.columns = ["episode", "reward", "avg_reward", "avg100_reward"]
                final_100_avg = float(df["avg100_reward"].iloc[-1])
                episodes_to_solved = int(df[df["reward"] >= 200.0]["episode"].min()) if (df["reward"] >= 200.0).any() else -1
                max_reward = float(df["reward"].max())

                metrics[f"CartPole_{name}"] = {
                    "environment": "CartPole-v1",
                    "algorithm": name,
                    "final_100_avg": final_100_avg,
                    "episodes_to_200": episodes_to_solved,
                    "max_reward": max_reward,
                    "total_episodes": len(df),
                }

        # 2. MountainCar Telemetry
        mountaincar_files = {
            "DQN": os.path.join(telemetry_dir, "mountaincar_dqn.tsv"),
            "DDQN": os.path.join(telemetry_dir, "mountaincar_ddqn.tsv"),
            "PER": os.path.join(telemetry_dir, "mountaincar_per.tsv"),
        }

        for name, path in mountaincar_files.items():
            if os.path.exists(path):
                df = pd.read_csv(path, sep=r"\s+", header=None)
                df.columns = ["episode", "ep_reward", "avg_reward", "car_position", "ep_steps"]
                final_avg_reward = float(df["avg_reward"].iloc[-1])
                best_position = float(df["car_position"].max())
                min_steps = int(df[df["car_position"] >= 0.5]["ep_steps"].min()) if (df["car_position"] >= 0.5).any() else int(df["ep_steps"].min())

                metrics[f"MountainCar_{name}"] = {
                    "environment": "MountainCar-v0",
                    "algorithm": name,
                    "final_avg_reward": final_avg_reward,
                    "best_position": best_position,
                    "min_steps_to_goal": min_steps,
                    "total_episodes": len(df),
                }

        return metrics
