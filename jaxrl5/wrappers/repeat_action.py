import gym
import numpy as np


class RepeatAction(gym.Wrapper):
    def __init__(self, env, action_repeat=4) -> None:
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray):
        total_reward = 0.0
        terminated = None
        truncated = None
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, combined_info