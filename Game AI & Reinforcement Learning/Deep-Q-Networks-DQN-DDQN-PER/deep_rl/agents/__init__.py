"""
Agents Subpackage.
Exposes standard DQN, Double DQN (DDQN), and Prioritized Experience Replay (PER).
Author: Abdul Rehman Rattu
"""

from deep_rl.agents.base_agent import BaseRLAgent
from deep_rl.agents.dqn_agent import DQNAgent
from deep_rl.agents.ddqn_agent import DDQNAgent
from deep_rl.agents.per_agent import PERAgent

__all__ = ["BaseRLAgent", "DQNAgent", "DDQNAgent", "PERAgent"]
