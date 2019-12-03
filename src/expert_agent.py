import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
import gym


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64],
                                              layer_norm=True,
                                              feature_extraction="mlp")


class ExpertAgent:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.n_timesteps = 10**5
        self.learning_rate = 10**-3
        self.buffer_size = 5*10**4
        self.exploration_fraction = 0.1
        self.exploration_final_eps = 0.02
        self.prioritized_replay = True
        self.model = DQN(CustomDQNPolicy, self.env, buffer_size=self.buffer_size,
                         learning_rate=self.learning_rate, verbose=1,
                         exploration_fraction=self.exploration_fraction,
                         exploration_final_eps=self.exploration_final_eps,
                         prioritized_replay=self.prioritized_replay)


    def train(self) -> None:
        self.model.learn(self.n_timesteps)
    
    def save(self, file_name: str) -> None:
        self.model.save(file_name)

    def load(self, file_name: str) -> None:
        self.model.load(file_name)

    def act(self, observation: np.ndarray) -> int:
        probs = self.model.action_probability(observation)
        actions = list(range(self.env.action_space.n))
        return np.random.choice(actions, p=probs)
        
if __name__ == "__main__":
    agent = ExpertAgent()
    agent.train()
    agent.save("expert.agent")
