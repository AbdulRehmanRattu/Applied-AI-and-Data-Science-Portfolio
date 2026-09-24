"""
Prioritized Experience Replay (PER) Agent with Double DQN Core.
Reference: Schaul et al., ICLR 2016 ("Prioritized Experience Replay").
Author: Abdul Rehman Rattu
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from deep_rl.agents.base_agent import BaseRLAgent
from deep_rl.replay_buffer import PrioritizedReplayBuffer


class PERAgent(BaseRLAgent):
    """
    Double Q-Network Agent coupled with Binary SumTree Prioritized Experience Replay.
    Samples transitions proportionally to their TD-error magnitudes.
    Applies Importance-Sampling (IS) weights to ensure unbiased gradient expectation.
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
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 0.001,
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
        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_increment=beta_increment,
        )

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """Record transition with maximal priority to ensure sampling at least once."""
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Sample prioritized mini-batch via SumTree, compute Double DQN target,
        apply importance-sampling weighted loss, and update SumTree leaf priorities.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, is_weights, tree_indices = (
            self.memory.sample(self.batch_size)
        )

        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        w_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.primary_network.train()
        # Q(s, a; theta)
        current_q = self.primary_network(s_t).gather(1, a_t)

        with torch.no_grad():
            # DDQN action selection: a* = argmax_a' Q(s', a'; theta)
            next_actions = self.primary_network(ns_t).argmax(dim=1, keepdim=True)
            # DDQN action evaluation: Q(s', a*; theta^-)
            target_eval = self.target_network(ns_t).gather(1, next_actions)
            # Decoupled Bellman target
            target_q = r_t + (self.gamma * target_eval * (1.0 - d_t))

        # Absolute TD-error for SumTree priority update
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy().flatten()
        self.memory.update_priorities(tree_indices, td_errors)

        # Importance-sampling weighted element-wise Huber loss
        elementwise_loss = self.loss_fn(current_q, target_q)
        weighted_loss = (w_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.primary_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        self.synchronize_target_network(tau=self.tau)

        return {
            "loss": float(weighted_loss.item()),
            "avg_q": float(current_q.mean().item()),
            "mean_td_error": float(np.mean(td_errors)),
            "beta": float(self.memory.beta),
            "epsilon": float(self.epsilon),
        }
