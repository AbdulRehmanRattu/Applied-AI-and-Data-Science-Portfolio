"""
Double Deep Q-Network (DDQN) Agent.
Reference: Van Hasselt, Guez & Silver, AAAI 2016 ("Deep Reinforcement Learning with Double Q-learning").
Author: Abdul Rehman Rattu
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from deep_rl.agents.base_agent import BaseRLAgent
from deep_rl.replay_buffer import UniformReplayBuffer


class DDQNAgent(BaseRLAgent):
    """
    Double Deep Q-Network Agent.
    Eliminates maximization bias by decoupling greedy action selection
    from target value evaluation:
        a* = argmax_a' Q(s'_i, a'; theta_primary)
        y_i = r_i + gamma * Q(s'_i, a*; theta_target) * (1 - done_i)
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
        """Sample mini-batch, compute DDQN target, and update network parameters."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.primary_network.train()
        # Primary evaluation Q(s, a; theta)
        current_q = self.primary_network(s_t).gather(1, a_t)

        with torch.no_grad():
            # 1. Action selection using primary online network: a* = argmax_a' Q(s', a'; theta)
            next_actions = self.primary_network(ns_t).argmax(dim=1, keepdim=True)
            # 2. Action evaluation using target network: Q(s', a*; theta^-)
            target_eval = self.target_network(ns_t).gather(1, next_actions)
            # 3. Decoupled Bellman target
            target_q = r_t + (self.gamma * target_eval * (1.0 - d_t))

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
