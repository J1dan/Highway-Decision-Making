# Highway-Decison-Making

## Introduction
We implement a highway environment to train the ego vehicle to conduct decision-making tasks. The main functions of scripts are:

*DecisionMakingEnv.py*: The gym environment in which decision trees are deployed on all vehicles. Observation space, action space and the rewards are all defined here.

*train_viz.py*: A file that is to be launched in the terminal for either training or visualizing trained model. Detailed usage is illustrated in the next section.

*train.py*:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This script is used to train and visualize the model. You can choose different RL algorithms as well as whether to conduct continual learning here.

*utility*:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This is what *train.py* imports from. The RL algorithms, the training monitor, and the visualization function are implemented here. You can adjust the parameters of the RL methods in this file. 

*experiment.py*: Here we can conduct experiments to evaluate the trained model as well as the decision tree. 

## Usage
1. Clone the repository.

2. Install required libraries through pip or conda.
```bash
    pip install -r requirements.txt
```
or
```bash
    env create --file environment.yml
    conda activate dm
```

3. To visualize our trained model, put the models in the /log directory.
```bash
    cd src
    train_viz.py --viz true --method DQN
```
You can select different methods here. Options: A2C, DQN, PPO, RecurrentPPO