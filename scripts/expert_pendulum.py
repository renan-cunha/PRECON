import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from src.expert_agent import ExpertAgent
from stable_baselines.common.base_class import BaseRLModel
from src.expert_dataset import generate_expert_dataset


class ExpertPendulum(ExpertAgent):

    def __init__(self, model: BaseRLModel):
        self.model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.model.predict(observation)


total_timesteps = 2*10**5
env = gym.make('Pendulum-v0')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                            sigma=float(0.1) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise,
             action_noise=action_noise, memory_limit=5*10**4)
model.learn(total_timesteps=total_timesteps)


expert_pendulum = ExpertPendulum(model)
dataset = generate_expert_dataset(expert_pendulum, env, total_timesteps)
np.save("states.npy", dataset["states"])
np.save("actions.npy", dataset["actions"])




