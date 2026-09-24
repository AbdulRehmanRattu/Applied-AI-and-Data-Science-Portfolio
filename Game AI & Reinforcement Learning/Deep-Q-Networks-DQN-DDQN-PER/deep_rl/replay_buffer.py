"""
Experience Replay Buffers for Deep Reinforcement Learning.
Includes Uniform FIFO Replay Buffer and SumTree-backed Prioritized Experience Replay (PER).
Author: Abdul Rehman Rattu
"""

from typing import Tuple, List, Any
import numpy as np
from deep_rl.sum_tree import SumTree


class UniformReplayBuffer:
    """
    Standard Experience Replay Buffer with uniform random mini-batch sampling.
    Breaks temporal correlations across consecutive environment transitions.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = int(capacity)
        self.storage = np.zeros(self.capacity, dtype=object)
        self.pointer = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Store a single transition tuple (s, a, r, s', done)."""
        self.storage[self.pointer] = (state, action, reward, next_state, done)
        self.pointer = (self.pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a mini-batch of transitions."""
        if self.size < batch_size:
            raise ValueError(f"Buffer contains {self.size} transitions; cannot sample {batch_size}.")

        indices = np.random.choice(self.size, size=batch_size, replace=False)
        batch = [self.storage[i] for i in indices]

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer using a Binary SumTree.
    Transitions with higher TD-errors are sampled with higher probability:
        P(i) = p_i^alpha / sum_k p_k^alpha

    Importance sampling (IS) weights correct for sampling bias:
        w_i = (N * P(i))^(-beta) / max_j w_j
    """

    def __init__(
        self,
        capacity: int = 20000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-4,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.beta_increment = float(beta_increment)
        self.epsilon = float(epsilon)
        self.max_priority = 1.0

        self.tree = SumTree(self.capacity)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Add experience with highest existing priority to guarantee immediate trial."""
        priority = (self.max_priority ** self.alpha)
        transition = (state, action, reward, next_state, done)
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Sample a prioritized mini-batch via stratified range partitioning.
        Returns:
            states, actions, rewards, next_states, dones, is_weights, tree_indices
        """
        if len(self.tree) < batch_size:
            raise ValueError(f"Buffer contains {len(self.tree)} transitions; cannot sample {batch_size}.")

        tree_indices = []
        batch = []
        priorities = []

        total_p = self.tree.total_priority
        segment = total_p / batch_size

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)
            val = np.random.uniform(start, end)
            tree_idx, p, transition = self.tree.get_leaf(val)

            # Fallback if unpopulated leaf selected
            if transition == 0 or transition is None:
                tree_idx, p, transition = self.tree.get_leaf(np.random.uniform(0, max(1e-5, total_p)))

            tree_indices.append(tree_idx)
            priorities.append(max(self.epsilon, p))
            batch.append(transition)

        # Compute sampling probabilities
        sampling_probs = np.array(priorities, dtype=np.float32) / (total_p + 1e-8)
        n = len(self.tree)

        # Importance-sampling weights w_i = (N * P(i))^(-beta)
        weights = (n * sampling_probs) ** (-self.beta)
        # Normalize by max weight for stability
        weights = weights / (weights.max() + 1e-8)
        weights = np.array(weights, dtype=np.float32)

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, weights, tree_indices

    def update_priorities(self, tree_indices: List[int], td_errors: np.ndarray) -> None:
        """Update transition priorities in the SumTree based on freshly computed TD-errors."""
        td_errors = np.abs(td_errors) + self.epsilon
        clipped_errors = np.minimum(td_errors, 100.0)
        priorities = clipped_errors ** self.alpha

        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(tree_idx, float(priority))
            if float(priority) > self.max_priority:
                self.max_priority = float(priority)

    def __len__(self) -> int:
        return len(self.tree)
