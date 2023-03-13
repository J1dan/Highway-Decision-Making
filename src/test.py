import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Vehicle:
    def __init__(self, position, speed, lane, acceleration=0, sign=0, t=0.1):
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.lane = lane
        self.sign = sign
        self.signtime = -1
        self.t = t

ego = Vehicle(1, 0, 1)
obstacle2 = Vehicle(2, 0, 1)
obstacle1 = Vehicle(1, 0, 3)
obstacles = [obstacle1, obstacle2]
sorted_obstacles = sorted(obstacles, key=lambda o: (ego.position-o.position)^2 + 9*(ego.lane-o.lane)^2, reverse=True)
print(sorted_obstacles[1].lane)

# class AutomaticParkingEnv(gym.Env):
#     def __init__(self):
#         self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([10, 10, 2]), dtype=np.float32)
#         self.action_space = spaces.Discrete(4)
#         self.goal_position = np.array([5, 5])
#         self.goal_radius = 0.5
#         self.car_size = np.array([1, 2])
#         self.car_position = np.array([0, 0])
#         self.car_heading = 0
#         self.max_steps = 100
#         self.current_step = 0
#         self.reward_range = (-float('inf'), float('inf'))

#     def reset(self):
#         self.car_position = np.array([0, 0])
#         self.car_heading = 0
#         self.current_step = 0
#         return self._get_observation()

#     def step(self, action):
#         self.current_step += 1
#         if action == 0:  # move forward
#             self.car_position += np.array([np.cos(self.car_heading), np.sin(self.car_heading)]) * 0.1
#         elif action == 1:  # move backward
#             self.car_position -= np.array([np.cos(self.car_heading), np.sin(self.car_heading)]) * 0.1
#         elif action == 2:  # turn left
#             self.car_heading += np.pi/6
#         elif action == 3:  # turn right
#             self.car_heading -= np.pi/6

#         if self.car_position[0] < 0:
#             self.car_position[0] = 0
#         elif self.car_position[0] > 10:
#             self.car_position[0] = 10
#         if self.car_position[1] < 0:
#             self.car_position[1] = 0
#         elif self.car_position[1] > 10:
#             self.car_position[1] = 10

#         done = False
#         reward = 0
#         if np.linalg.norm(self.car_position - self.goal_position) <= self.goal_radius:
#             done = True
#             reward = 1000
#         elif self.current_step >= self.max_steps:
#             done = True
#             reward = -1000

#         return self._get_observation(), reward, done, {}

#     def render(self, mode='human'):
#         pass

#     def _get_observation(self):
#         return np.concatenate([self.car_position, [self.car_heading]])

# gym.envs.register(
#     id='AutomaticParking-v0',
#     entry_point='automatic_parking_env:AutomaticParkingEnv',
# )

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=64):
#         super().__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)

#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class QAgent:
#     def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
#         self.q_net = QNetwork(state_dim, action_dim, hidden_dim)
#         self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
#         self.loss_fn = nn.MSELoss()
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon

#     def get_action(self, state):
#         if torch.rand(1) < self.epsilon:
#             return torch.randint(0, self.q_net.fc3.out_features, (1,))
#         else:
#             with torch.no_grad():
#                 q_values = self.q_net(state)
#                 return torch.argmax(q_values)

#     def update(self, state, action, next_state, reward, done):
#         self.q_net.train()
#         q_values = self.q_net(state)
#         next_q_values = self.q_net(next_state)
#         target_q = reward + (1 - done) * self.gamma * torch.max(next_q_values, dim=1)[0]
#         target_q = target_q.unsqueeze(1)
#         q_value = q_values.gather(1, action.unsqueeze(1))
#         loss = self.loss_fn(q_value, target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)



# env = gym.make('AutomaticParking-v0')
# agent = QAgent(env.observation_space.shape[0], env.action_space.n)

# n_episodes = 1000
# max_steps_per_episode = 1000
# scores = []
# for i_episode in range(n_episodes):
#     state = env.reset()
#     score = 0
#     for t in range(max_steps_per_episode):
#         action = agent.get_action(torch.FloatTensor(state))
#         next_state, reward, done, _ = env.step(action.item())
#         agent.update(torch.FloatTensor(state), action, torch.FloatTensor(next_state), reward, done)
#         state = next_state
#         score += reward
#         if done:
#             break
#     scores.append(score)
#     print(f"Episode {i_episode}: score={score}")
# print(f"Average score over {n_episodes} episodes: {np.mean(scores)}")
