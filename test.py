import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("PandaReach-v3", n_envs=4)

model = DDPG.load("ddpg_PandaReach")
# model = PPO.load("ppo_PandaReach")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")