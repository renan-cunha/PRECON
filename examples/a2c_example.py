from precon.pretrain import Pretrain
from stable_baselines import A2C, DDPG
from precon.expert_dataset import generate_expert_dataset
from precon.expert_agent import ExpertAgent
from stable_baselines.common import BaseRLModel
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import gym


num_steps = 2*10**6  # 10**3
num_epochs = 1000
env = gym.make('Pendulum-v0')

### Expert Agent ###
# You can do anything on the predict function, just implement the interface :)

class ExpertPendulum(ExpertAgent):

    def __init__(self, model: BaseRLModel):
        self.model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.model.predict(observation)


n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                            sigma=float(0.1) * np.ones(n_actions))

model = DDPG("MlpPolicy", env, verbose=1, param_noise=param_noise,
             action_noise=action_noise, memory_limit=5*10**4)
model.learn(total_timesteps=num_steps)


expert_agent = ExpertPendulum(model)

### Make Data ###
data = generate_expert_dataset(expert_agent, env, num_steps)

### Pre-Training ###
pre_trained_model = A2C("MlpPolicy", "Pendulum-v0")
pretrain = Pretrain(pre_trained_model, data['states'], data['actions'])
pretrain.fit(num_epochs=num_epochs)
pre_trained_model = pretrain.get_pretrained_model()

### Continue Learning ###
model.learn(total_timesteps=num_steps)