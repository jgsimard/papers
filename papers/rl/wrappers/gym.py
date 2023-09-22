import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation

from papers.rl.wrappers.single_precision import SinglePrecision


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:  # noqa: FBT001, FBT002
    env = SinglePrecision(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    return gym.wrappers.ClipAction(env)
