from src.expert_agent import ExpertAgent
import gym
from typing import Dict
import numpy as np


def generate_expert_dataset(expert_agent: ExpertAgent, env: gym.Env,
                            num_steps: int) -> Dict[str, np.ndarray]:
    pass
