import gym
import numpy as np
from typing import Tuple
from src.expert_agent import ExpertAgent
from tqdm import tqdm


class ExpertDataset:

    def __init__(self, env: gym.Env, agent: ExpertAgent):
        self.env = env
        self.agent = agent

    def create_dataset(self, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        dataset_states = np.empty((num_steps, 
                                   self.env.observation_space.shape[0]),
                                   dtype=np.float32)
        dataset_actions = np.zeros((num_steps, self.env.action_space.n),
                                    dtype=bool)
        state = self.env.reset()
        for i in tqdm(range(num_steps)):
            action = self.agent.act(state)
            next_state, _, done, _ = self.env.step(action)
            if done:
                next_state = self.env.reset()
            dataset_states[i] = state
            dataset_actions[i, action] = 1
            state = next_state
        dataset_states, dataset_actions = self.clean_dataset(dataset_states,
                                                             dataset_actions)
        return dataset_states, dataset_actions

    def clean_dataset(self, states: np.array, 
                      actions: np.array) -> Tuple[np.ndarray, np.ndarray]:
        actions_dim = actions.shape[1]
        states_dim = states.shape[1]
        dataset = np.concatenate((states, actions), axis=1)
        dataset = np.unique(dataset, axis=0)
        states = dataset[:, :states_dim]
        actions = dataset[:, -actions_dim:]
        return states, actions

if __name__ == "__main__":
    env = gym.make("CartPole-v1a")
    expert_agent = ExpertAgent()
    expert_agent.load("expert.agent")
    expert_dataset = ExpertDataset()
    #number of rows to make 2gb
    num_steps = 1.6*10**10/(env.observation_space.shape[0]*32)
    data = expert_dataset.create_dataset()
