import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

# Parallel environments
raw_env = gym.make("Pendulum-v0")
env = DummyVecEnv([lambda: gym.make("Pendulum-v0")])
model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25)
raw_state = raw_env.reset()
state = env.reset()
print(model.predict(raw_state))
print(model.predict(state))
# TODO: Test dummy env on my workspace
