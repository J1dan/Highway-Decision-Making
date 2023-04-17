import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO

from newDecisionMaking import HighwayEnv
from utility import train
from utility import viz

log_dir = "log/"
env = HighwayEnv()

method = 'RecurrentPPO' # 'DQN', 'A2C', 'PPO', 'RecurrentPPO'
CONTINUE = False # Continue learning
model = train(method, env, 5e5, log_dir, verbose=0, continual=CONTINUE, force_update=1) 
# model = RecurrentPPO.load(log_dir+"/RecurrentPPO/best_model.zip", env=env)
viz(model, env, method)