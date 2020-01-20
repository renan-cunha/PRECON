import gym
from precon.pretrain import Pretrain
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
from copy import deepcopy
from stable_baselines.common import BaseRLModel
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parallel environments
env = make_vec_env('Pendulum-v0', n_envs=8)
actions = np.load("scripts/actions.npy")
states = np.load("scripts/states.npy")


# make results reproducible
# plot

timesteps = 2*10**6
num_epochs = 1000
batch_size = 512
num_seeds = 3
num_evaluations = 30

total_rewards = np.zeros((num_seeds, num_evaluations, 2))  #raw, pretrain


def evaluate_model(model: BaseRLModel, env: gym.Env) -> float:
    obs = env.reset()
    evaluation_reward = 0
    while True:
        action = model.predict(obs)
        action = action[0]
        obs, reward, done, _ = evaluation_env.step(action)
        evaluation_reward += reward
        if done:
            break
    return evaluation_reward


def construct_model(seed:int) -> A2C:
    return A2C(MlpPolicy, env, verbose=0, tensorboard_log="/tmp/a2c/",
               ent_coef=0.0, gamma=0.95, n_cpu_tf_sess=1, seed=seed)

pbar = tqdm(total=num_seeds)
for seed in range(num_seeds):
    construct_seed = seed+100
    raw_model = construct_model(construct_seed)

    pretrain_model = construct_model(construct_seed)
    pretrain = Pretrain(pretrain_model, states, actions)
    pretrain.fit(num_epochs=num_epochs, batch_size=batch_size)
    pretrain_model = pretrain.get_pretrained_model()

    raw_model.learn(total_timesteps=timesteps, tb_log_name="raw")
    pretrain_model.learn(total_timesteps=timesteps, tb_log_name="pretrained")

    for model_index, model in enumerate([raw_model, pretrain_model]):
        for evaluation in range(num_evaluations):
            evaluation_env = make_vec_env('Pendulum-v0', n_envs=1,
                                          seed=evaluation)
            evaluation_reward = evaluate_model(model, evaluation_env)
            total_rewards[seed, evaluation, model_index] = evaluation_reward
    pbar.update(1)
pbar.close()


raw_rewards = total_rewards[:, :, 0].flatten()
pretrain_rewards = total_rewards[:, :, 1].flatten()
raw_mean = raw_rewards.mean()
pretrain_mean = pretrain_rewards.mean()

raw_std = np.std(raw_rewards)
pretrain_std = np.std(pretrain_rewards)

plt.grid()
plt.bar(x=["Baseline", "Pretrained"], height=[raw_mean, pretrain_mean],
        yerr=[raw_std, pretrain_std])
plt.title("Comparison between baseline model and (pretrained model + default "
          "learning)")
plt.ylabel("Reward")
plt.show()