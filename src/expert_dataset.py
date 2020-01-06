from src.expert_agent import ExpertAgent
import gym
from typing import Dict
import numpy as np


def generate_expert_dataset(expert_agent: ExpertAgent, env: gym.Env,
                            num_steps: int) -> Dict[str, np.ndarray]:

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    states = np.array((num_steps, state_dim), dtype="float32")
    actions = np.array((num_steps, action_dim), dtype="float32")

    obs = env.reset()
    for i in range(num_steps):
        action = expert_agent.predict(obs)

        states[i] = obs
        actions[i] = action

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    return {"actions": actions, "states": states}
