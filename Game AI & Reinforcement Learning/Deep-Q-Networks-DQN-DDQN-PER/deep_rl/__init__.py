"""
Deep Reinforcement Learning (DQN, Double DQN, and Prioritized Experience Replay).
Laboratory 20 - Applied AI and Data Science Portfolio.
Author: Abdul Rehman Rattu
"""

from deep_rl.sum_tree import SumTree
from deep_rl.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from deep_rl.network import QNetwork
from deep_rl.agents.base_agent import BaseRLAgent
from deep_rl.agents.dqn_agent import DQNAgent
from deep_rl.agents.ddqn_agent import DDQNAgent
from deep_rl.agents.per_agent import PERAgent
from deep_rl.environments import CartPoleEnvWrapper, MountainCarEnvWrapper

__version__ = "1.0.0"
__author__ = "Abdul Rehman Rattu"

__all__ = [
    "SumTree",
    "UniformReplayBuffer",
    "PrioritizedReplayBuffer",
    "QNetwork",
    "BaseRLAgent",
    "DQNAgent",
    "DDQNAgent",
    "PERAgent",
    "CartPoleEnvWrapper",
    "MountainCarEnvWrapper",
]
