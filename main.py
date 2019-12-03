import gym
from src.expert_agent import ExpertAgent
from src.expert_dataset import ExpertDataset
import numpy as np

env = gym.make("CartPole-v1")
expert_agent = ExpertAgent()
expert_agent.load("data/expert.agent")
expert_dataset = ExpertDataset(env, expert_agent)
num_steps = 10**5
states, actions = expert_dataset.create_dataset(int(num_steps))
np.save("data/states.npy", states)
np.save("data/actions.npy", actions)

