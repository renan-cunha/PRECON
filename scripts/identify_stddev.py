import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
env = make_vec_env('Pendulum-v0', n_envs=4)

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

obs = env.reset()
while True:
    print(model.action_probability(obs))
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()