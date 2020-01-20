import gym
from src.pretrain import Pretrain
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
env = make_vec_env('LunarLanderContinuous-v2', n_envs=16)
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/a2c/",
            ent_coef=0.001, gamma=0.999, lr_schedule="linear")
actions = np.load("scripts/actions.npy")
states = np.load("scripts/states.npy")
pretrain = Pretrain(model, states, actions)
#pretrain.fit(num_epochs=1000, batch_size=512)
#model = pretrain.get_pretrained_model()
model.learn(total_timesteps=5*10**6)
