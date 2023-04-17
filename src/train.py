import gym
import matplotlib.pyplot as plt
from newDecisionMaking import HighwayEnv
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

env = HighwayEnv()
# check_env(env)

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=2000)
# model.save("model")
print(f"Model saved")

# del model
# model = DQN.load("log/best_model.zip", env=env)
model = DQN.load("model", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=40)
print(f"Mean reward = {mean_reward}, std_reward = {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(2000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()