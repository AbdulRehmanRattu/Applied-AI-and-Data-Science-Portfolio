"""
Deep Neural Network Architectures for Q-Value Function Approximation.
Implements multi-layer perceptron (MLP) Q-networks with Kaiming He initialization.
Author: Abdul Rehman Rattu
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """
    Parametric Q-value approximator Q(s, a; theta).
    Maps continuous observation vector s to a discrete vector of action-values.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers: List[nn.Module] = []
        in_dim = state_dim

        for h_dim in hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            # He / Kaiming Uniform initialization
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            in_dim = h_dim

        # Output head: Linear projection to action space
        output_layer = nn.Linear(in_dim, action_dim)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation.
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            Tensor of shape (batch_size, action_dim) containing Q-values.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state)

    def get_q_values(self, state: np.ndarray, device: torch.device) -> np.ndarray:
        """Convenience evaluation for NumPy arrays in evaluation loops."""
        self.eval()
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            if s_tensor.dim() == 1:
                s_tensor = s_tensor.unsqueeze(0)
            q_vals = self.network(s_tensor)
            return q_vals.cpu().numpy()
