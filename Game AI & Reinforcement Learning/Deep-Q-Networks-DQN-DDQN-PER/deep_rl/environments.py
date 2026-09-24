"""
Environment Wrappers for CartPole-v1 and MountainCar-v0.
Standardizes gym / gymnasium interfaces and provides potential-based reward shaping.
Author: Abdul Rehman Rattu
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

# Try importing gymnasium first, fallback to gym
try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    import gym
    GYMNASIUM_AVAILABLE = False


class CartPoleEnvWrapper:
    """
    Wrapper for CartPole-v1 environment.
    Observation space: 4 continuous variables [cart_pos, cart_vel, pole_angle, pole_tip_vel].
    Action space: 2 discrete actions [0: Push Left, 1: Push Right].
    """

    def __init__(self, render_mode: Optional[str] = None, max_episode_steps: int = 500):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        if GYMNASIUM_AVAILABLE:
            self.env = gym.make("CartPole-v1", render_mode=render_mode)
        else:
            self.env = gym.make("CartPole-v1")

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def reset(self) -> np.ndarray:
        """Reset environment to initial distribution and return state vector."""
        res = self.env.reset()
        if isinstance(res, tuple):
            return np.array(res[0], dtype=np.float32)
        return np.array(res, dtype=np.float32)

    def step(self, action: int, shape_reward: bool = True) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action.
        Returns:
            next_state, reward, done, info
        """
        step_res = self.env.step(action)
        if len(step_res) == 5:
            next_state, raw_reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            next_state, raw_reward, done, info = step_res

        # Reward engineering: penalize premature failure
        if shape_reward:
            reward = float(raw_reward) if not done else -100.0
        else:
            reward = float(raw_reward)

        return np.array(next_state, dtype=np.float32), reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class MountainCarEnvWrapper:
    """
    Wrapper for MountainCar-v0 environment.
    Observation space: 2 continuous variables [car_position, car_velocity].
    Action space: 3 discrete actions [0: Accelerate Left, 1: Coast, 2: Accelerate Right].
    """

    def __init__(self, render_mode: Optional[str] = None, max_episode_steps: int = 200):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        if GYMNASIUM_AVAILABLE:
            self.env = gym.make("MountainCar-v0", render_mode=render_mode)
        else:
            self.env = gym.make("MountainCar-v0")

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def reset(self) -> np.ndarray:
        """Reset environment and return observation."""
        res = self.env.reset()
        if isinstance(res, tuple):
            return np.array(res[0], dtype=np.float32)
        return np.array(res, dtype=np.float32)

    def step(self, action: int, current_state: np.ndarray, shape_reward: bool = True) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action with potential-based kinematic reward shaping.
        """
        step_res = self.env.step(action)
        if len(step_res) == 5:
            next_state, raw_reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            next_state, raw_reward, done, info = step_res

        next_state = np.array(next_state, dtype=np.float32)

        if shape_reward:
            pos_t = current_state[0]
            pos_next = next_state[0]
            vel_t = current_state[1]

            # Goal achieved bonus: flag position is x >= 0.5
            if pos_next >= 0.5:
                reward = 200.0
            else:
                # Potential-based kinematic momentum incentive
                reward = 5.0 * abs(pos_next - pos_t) + 3.0 * abs(vel_t)
        else:
            reward = float(raw_reward)

        return next_state, float(reward), done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
