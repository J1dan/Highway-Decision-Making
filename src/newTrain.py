import os
import numpy as np

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from newDecisionMaking import HighwayEnv
from utility import train
from utility import SaveOnBestTrainingRewardCallback
from utility import viz

log_dir = "log/"
env = HighwayEnv()

method = 'PPO' # 'DQN', 'A2C', 'PPO', 'RecurrentPPO'
CONTINUE = True # Continue learning
model = train(method, env, 5e3, log_dir, verbose=0, continual=CONTINUE, force_update=1) 

viz(model, env, method)