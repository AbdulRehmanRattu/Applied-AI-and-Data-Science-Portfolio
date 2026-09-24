"""
Base Abstract Reinforcement Learning Agent.
Handles policy exploration, device placement, checkpointing, and target network synchronization.
Author: Abdul Rehman Rattu
"""

from typing import Tuple, Optional
import os
import random
import numpy as np
import torch
import torch.nn as nn
from deep_rl.network import QNetwork


class BaseRLAgent:
    """
    Base Agent encapsulation providing common reinforcement learning utilities:
    - Epsilon-greedy exploration schedule
    - Primary and target Q-network synchronization (soft and hard)
    - Checkpoint persistence and restoration
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        hidden_dims: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        # Primary online network
        self.primary_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        # Target network
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.synchronize_target_network(tau=1.0)  # Hard initialize target weights
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.primary_network.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss for outlier robustness

        self.training_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Epsilon-greedy policy for action selection.
        If evaluate is True, acts purely greedily (deterministic argmax).
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        self.primary_network.eval()
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if s_tensor.dim() == 1:
                s_tensor = s_tensor.unsqueeze(0)
            q_values = self.primary_network(s_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        """Anneal exploration epsilon down to epsilon_min."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def synchronize_target_network(self, tau: float = 0.005) -> None:
        """
        Polyak soft target network update:
        theta_target = tau * theta_primary + (1 - tau) * theta_target
        If tau == 1.0, performs exact hard copy.
        """
        with torch.no_grad():
            for target_param, primary_param in zip(self.target_network.parameters(), self.primary_network.parameters()):
                target_param.data.copy_(tau * primary_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, filepath: str) -> None:
        """Persist agent state dictionary."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "primary_network_state": self.primary_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.primary_network.load_state_dict(checkpoint["primary_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.training_steps = checkpoint.get("training_steps", 0)
