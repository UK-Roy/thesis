import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
# vec_env = make_vec_env("PandaReach-v3", n_envs=4)
env = gym.make("PandaReach-v3")

model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(10_000)

# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
model.save("ddpg_PandaReach")