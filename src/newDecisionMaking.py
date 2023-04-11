import math
# import gymnasium as gym
# from gymnasium import spaces
import gym
from gym import spaces
import numpy as np

class Vehicle:
    def __init__(self, position, speed, lane, acceleration=0, sign=0, t=0.1, target_speed=30):
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.lane = lane
        self.sign = sign
        self.signtime = -1
        self.t = t
        self.target_speed = target_speed

    def act(self, action):
        if action == 'vKeeping' or action == 0:
            self.position += self.speed * self.t
        elif action == 'accelerate' or action == 1:
            self.acceleration = 1
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2
        elif action == 'decelerate' or action == 2:
            self.acceleration = -1
            self.speed += self.acceleration * 0.1
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2
        elif action == 'changeLaneR' or action == 3:
            # print(f"Before, ego's lane: {self.lane}")
            self.lane = self.lane - 1
            self.position += self.speed * self.t
            self.sign = 0
            # print(f"After, ego's lane: {self.lane}\n")
        elif action == 'changeLaneL' or action == 4:
            # print(f"Before, ego's lane: {self.lane}")
            self.lane = self.lane + 1
            self.position += self.speed * self.t
            self.sign = 0
            # print(f"After, ego's lane: {self.lane}\n")

class RoadManager(object):
    def __init__(self, num_lanes=4):
        self.holding_system = []
        for i in range(num_lanes):
            self.holding_system.append([])

    def add(self, vehicle):
        self.holding_system[vehicle.lane].append(vehicle)

    def delete(self, vehicle):
        self.holding_system[vehicle.lane].remove(vehicle)

class HighwayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.car_length = 4.5 # length of the ego car
        self.car_width = 2.0 # width of the ego car
        self.lane_width = 4.0 # width of each lane
        self.num_lanes = 4 # number of lanes
        self.num_obstacles = 9
        self.min_speed = 0.0 # minimum speed limit
        self.max_speed = 50.0 # maximum speed limit
        self.max_acceleration = 2.0 # maximum acceleration
        self.max_deceleration = 5.0 # maximum deceleration
        self.max_lane_change = 1 # maximum number of lanes that can be changed at once
        self.time_step = 0
        self.t = 0.1
        self.max_time_step = 1200
        self.obstacle_speeds = [20, 30, 40]

        self.ego = None
        self.obstacles = []
        self.nearest_obstacles = []

        # Initialize the state of the environment
        self.reset()

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(5) # four actions: change lane right, left, decelerate, accelerate, velocitykeeping
        self.observation_space = gym.spaces.Box(low=np.array([0, -1, 0, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1]), 
                                    high=np.array([1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1]), 
                                    dtype=np.float32)

    def reset(self):
        self.time_step = 0

        self.manager = RoadManager(self.num_lanes)

        # Initialize the ego
        self.ego = Vehicle(position=50, speed=np.random.randint(30, 50), acceleration=0, lane=np.random.randint(0, self.num_lanes), sign='none', target_speed=50)
        
        self.manager.add(self.ego)
        
        self.obstacles = []

        # Initialize the obstacle
        for i in range(self.num_obstacles):
            feasible = False
            while not feasible:
                position = np.random.uniform(0, 100)
                lane = np.random.randint(0, self.num_lanes)
                # If the generated obstacle does not collide with the ego and other vehicles, considered feasible
                if (abs(position - self.ego.position) > 10 or lane != self.ego.lane) and (abs(position - o.position) > 10 or lane != o.lane for o in self.obstacles):
                    feasible = True
            speed = np.random.randint(30, 40) if np.random.random()<0.5 else np.random.randint(40, 50)
            obstacle = Vehicle(position, speed, lane, acceleration=0, target_speed=speed)
            self.obstacles.append(obstacle)

            self.manager.add(obstacle)
        # Get nearest obstacles
        self.nearest_obstacles = sorted(self.obstacles, key=lambda o: (self.ego.position-o.position)**2 + 9*(self.ego.lane-o.lane)**2, reverse=False)[:5]
        return self._get_observation()

    def step(self, action):
        # print(f"Excecuted action: {action}")
        # Increment the step count
        self.time_step += 1
        self.manager.delete(self.ego)
        self.ego.act(action)
        self.manager.add(self.ego)
        # Decision Tree

        # Update the position and speed of the obstacles
        for obstacle in self.obstacles:

            FINISH = 0
            CHANGELANE = 1
            if obstacle.signtime == self.time_step: # Time to change lane
                for nearObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                    if abs(nearObs.position - obstacle.position) < 6: # If change lane, collide
                        # Reset, not finish
                        obstacle.sign = 0
                        obstacle.signtime = -1
                        CHANGELANE = 0
                        break
                if CHANGELANE:
                    if obstacle.sign == -1:
                        self.manager.delete(obstacle)
                        obstacle.act('changeLaneR')
                        self.manager.add(obstacle)
                        
                    else:
                        self.manager.delete(obstacle)
                        obstacle.act('changeLaneL')
                        self.manager.add(obstacle)
                    obstacle.signtime = -1 # Reset signtime
                    FINISH = 1 # Done, no operation needed

            # See if is too close to the obstacles ahead
            if not FINISH:
                for o in self.manager.holding_system[obstacle.lane]:
                    if o.position - obstacle.position < 20: # Too close
                        if o.speed < obstacle.speed:
                            obstacle.act('decelerate')
                            obstacle.signtime = self.time_step + 20
                            if obstacle.lane == 0:
                                obstacle.sign = 1 # Change lane to left
                                for nearObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                    if abs(nearObs.position - obstacle.position) < 6:
                                        obstacle.sign = 0
                                        obstacle.signtime = -1
                                        break

                            elif obstacle.lane == 3:
                                obstacle.sign = -1 # Change lane to right
                                for nearObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                    if abs(nearObs.position - obstacle.position) < 6:
                                        obstacle.sign = 0
                                        obstacle.signtime = -1
                                        break
                            
                            else:
                                obstacle.sign = 1 # Change lane to left
                                LABEL = 1 # OK to change lane
                                for nearObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                    if abs(nearObs.position - obstacle.position) < 6:
                                        obstacle.sign = 0
                                        obstacle.signtime = -1
                                        LABEL = 0
                                        break
                                if LABEL == 0:
                                    obstacle.sign = -1
                                    for nearObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                        if abs(nearObs.position - obstacle.position) < 6:
                                            obstacle.sign = 0
                                            obstacle.signtime = -1
                                            LABEL = 0
                                            break 
                        
                        FINISH = 1 # current obstacle update done, continue to the next obstacle
                        break
            if FINISH:
                continue

            if obstacle.speed < obstacle.target_speed:
                obstacle.act('accelerate')
            else:
                obstacle.act('vKeeping')
        

            # After 3s, excecute lane changing
            if self.time_step == obstacle.signtime:
                obstacle.lane += obstacle.sign
                if abs(obstacle.position - self.ego.position) <= 10 and obstacle.lane == self.ego.lane:
                    obstacle.lane -= obstacle.sign
                obstacle.signtime = -1

            obstacle.position += obstacle.speed * self.t

        self.nearest_obstacles = sorted(self.obstacles, key=lambda o: (self.ego.position-o.position)**2 + 9*(self.ego.lane-o.lane)**2, reverse=False)[:5]

        done = False
        # Check for collisions between the ego and the boundary
        
        if self.ego.lane < 0 or self.ego.lane > 3:
            # print(f"Ego's lane: {self.ego.lane}")
            print(f"Boundary Collision at timestep {self.time_step}")
            reward = -100
            done = True 

        # Check for collisions between the ego car and obstacles
        if not done:
            for obstacle in self.obstacles:
                if obstacle.lane == self.ego.lane and abs(obstacle.position - self.ego.position) < self.car_length:
                    print(f"Obs Collision at timestep {self.time_step}")
                    reward = -100
                    done = True
                    break

        # Reward the ego car for maintaining speed and changing lanes
        if not done:
            reward = self.ego.speed/self.max_speed/100
            # reward = 0
            if action == 0 or action == 1:
                reward -= 0.1

        if self.time_step > 1800:
            done = True

        # Return the observation, reward, done flag, and additional info
        observation = self._get_observation()
        return observation, reward, done, {}

    def _get_observation(self):
        # Get the state of the ego car and obstacles
        observation = [self.ego.speed/self.max_speed, self.ego.acceleration, (self.ego.lane+1)/4]
        for x in self.nearest_obstacles:
            observation.extend([(self.ego.position-x.position)**2 + 9*(self.ego.lane-x.lane)**2, \
                                    x.speed/self.max_speed, x.lane/3, x.sign])
        observation = np.array(observation, dtype=np.float32)
        # print(f"Shape of returned _get_observation: {observation.shape}")
        return observation

    def render(self, mode='human'):
        # Display the current state of the environment
        pass

if __name__=='__main__':
    env = HighwayEnv()

    # Print the initial state of the ego 
    print(f"Timestep {env.time_step}:")
    print(f"Ego's position:{env.ego.position}\nEgo's speed: {env.ego.speed}\nEgo's acc: {env.ego.acceleration}\nEgo's lane: {env.ego.lane}")
    # Print the initial state of the obstacles
    for i in range(len(env.obstacles)):
        obs = env.obstacles[i]
        print(f"Vehicle_{i}'s position:{obs.position}\nVehicle_{i}'s speed: {obs.speed}\nVehicle_{i}'s lane: {obs.lane}")

    obs, reward, done, _ = env.step('accelerate')
    print(f"Timestep {env.time_step}:")
    print(f"Ego's position:{env.ego.position}\nEgo's speed: {env.ego.speed}\nEgo's acc: {env.ego.acceleration}\nEgo's lane: {env.ego.lane}\n")
    for i in range(len(env.nearest_obstacles)):
        obs = env.nearest_obstacles[i]
        print(f"Nearest vehicle_{i}'s position:{obs.position}\nVehicle_{i}'s speed: {obs.speed}\nVehicle_{i}'s lane: {obs.lane}\n")

    print(f"Reward = {reward}, done = {done}")
    for i in range(len(env.manager.holding_system)):
        laneObs = []
        for o in env.manager.holding_system[i]:
            laneObs.append(o.position)
        laneObs = sorted(laneObs)
        print(f"At lane {i}, the positions of the obstacles are {laneObs}")