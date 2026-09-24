"""
Standard Deep Q-Network (DQN) Agent.
Reference: Mnih et al., Nature 2015 ("Human-level control through deep reinforcement learning").
Author: Abdul Rehman Rattu
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from deep_rl.agents.base_agent import BaseRLAgent
from deep_rl.replay_buffer import UniformReplayBuffer


class DQNAgent(BaseRLAgent):
    """
    Standard Deep Q-Network Agent.
    Computes Bellman target via max operator over target network action-values:
        y_i = r_i + gamma * max_a' Q(s'_i, a'; theta_target) * (1 - done_i)
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
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        tau: float = 0.005,
        hidden_dims: Tuple[int, ...] = (64, 64),
        device: Optional[str] = None,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            hidden_dims=hidden_dims,
            device=device,
        )
        self.batch_size = batch_size
        self.tau = tau
        self.memory = UniformReplayBuffer(capacity=buffer_capacity)

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Record transition in uniform replay buffer."""
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        """Sample mini-batch, compute standard DQN target, and perform gradient step."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.primary_network.train()
        # Q(s, a; theta)
        current_q = self.primary_network(s_t).gather(1, a_t)

        with torch.no_grad():
            # max_a' Q(s', a'; theta^-)
            next_q = self.target_network(ns_t).max(1, keepdim=True)[0]
            # Standard Bellman target
            target_q = r_t + (self.gamma * next_q * (1.0 - d_t))

        # Smooth L1 Huber loss
        loss = self.loss_fn(current_q, target_q).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.primary_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        self.synchronize_target_network(tau=self.tau)

        return {
            "loss": float(loss.item()),
            "avg_q": float(current_q.mean().item()),
            "epsilon": float(self.epsilon),
        }
