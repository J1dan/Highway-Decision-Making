import os
import gymnasium as gym
from stable_baselines3 import A2C, SAC, PPO, TD3

# Create save dir
save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = PPO("MlpPolicy", "Pendulum-v1", verbose=0).learn(8_000)
# The model will be saved under PPO_tutorial.zip
model.save(f"{save_dir}/PPO_tutorial")

# sample an observation from the environment
obs = model.env.observation_space.sample()

# Check prediction before saving
print("pre saved", model.predict(obs, deterministic=True))

del model  # delete trained model to demonstrate loading

loaded_model = PPO.load(f"{save_dir}/PPO_tutorial")
# Check that the prediction is the same after loading (for the same observation)
print("loaded", loaded_model.predict(obs, deterministic=True))

import os
from stable_baselines3.common.vec_env import DummyVecEnv

# Create save dir
save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = A2C("MlpPolicy", "Pendulum-v1", verbose=0, gamma=0.9, n_steps=20).learn(8000)
# The model will be saved under A2C_tutorial.zip
model.save(f"{save_dir}/A2C_tutorial")

del model  # delete trained model to demonstrate loading

# load the model, and when loading set verbose to 1
loaded_model = A2C.load(f"{save_dir}/A2C_tutorial", verbose=1)

# show the save hyperparameters
print(f"loaded: gamma={loaded_model.gamma}, n_steps={loaded_model.n_steps}")

# as the environment is not serializable, we need to set a new instance of the environment
loaded_model.set_env(DummyVecEnv([lambda: gym.make("Pendulum-v1")]))
# and continue training
loaded_model.learn(8_000)