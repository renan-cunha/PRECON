import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

num_iterations = 3
raws_reward = np.zeros((1000, num_iterations))
pretrained_rewards = np.zeros_like(raws_reward)

for array_index, array in enumerate([raws_reward, pretrained_rewards]):
    for i in range(num_iterations):
        if array_index == 0:
            file_name = f"run-raw_{i+1}.csv"
        else:
            file_name = f"run-pretrained_{i + 1}.csv"
        csv = pd.read_csv(file_name)
        values = csv["Value"]
        values = savgol_filter(values, 51, 3)
        array[:, i] = values


sns.tsplot(raws_reward.transpose(), color="red")
sns.tsplot(pretrained_rewards.transpose())
plt.legend(["Baseline", "Pretrained"])
plt.ylabel("Episode Reward")
plt.title("Comparison on Pendulum-v0")
plt.grid()
plt.show()

