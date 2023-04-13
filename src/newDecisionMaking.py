import math
# import gymnasium as gym
# from gymnasium import spaces
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        if action == 'maintain' or action == 0:
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t

        elif action == 'changeLaneR' or action == 1:
            self.lane = self.lane - 1
            self.position += self.speed * self.t
            self.sign = 0

        elif action == 'changeLaneL' or action == 2:
            self.lane = self.lane + 1
            self.position += self.speed * self.t
            self.sign = 0

        elif action == 'accelerate_0.2' or action == 3:
            self.acceleration += 0.2
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'accelerate_0.4' or action == 4:
            self.acceleration += 0.4
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'accelerate_0.6' or action == 5:
            self.acceleration += 0.6
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'accelerate_0.8' or action == 6:
            self.acceleration += 0.8
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'accelerate_1.0' or action == 7:
            self.acceleration += 1.0
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'decelerate_0.2' or action == 8:
            self.acceleration -= 0.2
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'decelerate_0.4' or action == 9:
            self.acceleration -= 0.4
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'decelerate_0.6' or action == 10:
            self.acceleration -= 0.6
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'decelerate_0.8' or action == 11:
            self.acceleration -= 0.8
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

        elif action == 'decelerate_1.0' or action == 12:
            self.acceleration -= 1.0
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

    def DTact(self, action):
        if action == 'maintain' or action == 0:
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t

        elif action == 'changeLaneR' or action == 1:
            self.lane = self.lane - 1
            self.position += self.speed * self.t
            self.sign = 0

        elif action == 'changeLaneL' or action == 2:
            self.lane = self.lane + 1
            self.position += self.speed * self.t
            self.sign = 0

        else:
            if action - self.acceleration > 1:
                self.acceleration += 1
            elif action - self.acceleration < -1:
                self.acceleration -= 1
            else:
                 self.acceleration = action
            self.speed += self.acceleration * self.t
            self.position += self.speed * self.t - 0.5 * self.acceleration * self.t * self.t  # vt*t-0.5*a*t**2

class RoadManager(object):
    def __init__(self, num_lanes=4):
        self.holding_system = []
        for i in range(num_lanes):
            self.holding_system.append([])

    def add(self, vehicle):
        if vehicle.lane < 0:
            self.holding_system[vehicle.lane+1].append(vehicle)
        elif vehicle.lane > 3:
            self.holding_system[vehicle.lane-1].append(vehicle)
        else:
            self.holding_system[vehicle.lane].append(vehicle)

    def delete(self, vehicle):
        if vehicle.lane < 0:
            self.holding_system[vehicle.lane+1].remove(vehicle)
        elif vehicle.lane > 3:
            self.holding_system[vehicle.lane-1].remove(vehicle)
        else:
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
        self.action_space = spaces.Discrete(13)
        
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1, -np.inf, 0, 0, -1]), 
                                    high=np.array([1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1, np.inf, 1, 1, 1]), 
                                    dtype=np.float32)

    def reset(self):
        self.time_step = 0

        self.manager = RoadManager(self.num_lanes)

        # Initialize the ego
        self.ego = Vehicle(position=0, speed=np.random.randint(30, 50), acceleration=0, lane=np.random.randint(0, self.num_lanes), sign='none', target_speed=50)
        
        self.manager.add(self.ego)
        
        self.obstacles = []

        # Initialize the obstacle
        for i in range(self.num_obstacles):
            FEASIBLE = False
            while not FEASIBLE:
                position = np.random.uniform(0, 100)
                lane = np.random.randint(0, self.num_lanes)

                # If the generated obstacle does not collide with the ego and other vehicles, considered FEASIBLE
                if (abs(position - self.ego.position) > 10 or lane != self.ego.lane):
                    for o in self.obstacles:
                        if abs(position - o.position) <= 10 and lane == o.lane:
                            FEASIBLE = False
                            break
                    else:
                        FEASIBLE = True
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

        # Update each obstacle's state
        for obstacle in self.obstacles:
        
        # --------------------- Decision Tree ------------------------
            FINISH = 0
            CHANGELANE = 1
            if obstacle.signtime == self.time_step: # Time to change lane
                for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                    if abs(nearbyObs.position - obstacle.position) < 6: # If change lane, collide
                        # Reset, not finish
                        obstacle.sign = 0
                        obstacle.signtime = -1
                        CHANGELANE = 0
                        break
                if CHANGELANE:
                    if obstacle.sign == -1:
                        self.manager.delete(obstacle)
                        obstacle.DTact('changeLaneR')
                        self.manager.add(obstacle)
                        
                    else:
                        self.manager.delete(obstacle)
                        obstacle.DTact('changeLaneL')
                        self.manager.add(obstacle)
                    obstacle.signtime = -1 # Reset signtime
                    FINISH = 1 # Done, no further operation needed

            # See if is too close to the obstacles ahead
            if not FINISH:
                dangerObs = []
                for o in self.manager.holding_system[obstacle.lane]:
                    if 0 < o.position - obstacle.position < 15: # Too close
                        dangerObs.append(o)

                    if len(dangerObs)>0:
                        obs_ahead = min(dangerObs, key=lambda obs:obs.position)

                        if obs_ahead.speed < obstacle.speed:
                            obstacle.act(-1)
                            if obstacle.sign != 0: # Already turned on light, do not update the signtime
                                FINISH = 1
                                break
                            
                            obstacle.signtime = self.time_step + 20 # Turn on the turn signal

                            if obstacle.lane == 0:
                                obstacle.sign = 1 # Change lane to left
                                for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                    # If nearby lane has dangerous obstacle, TURN OFF
                                    if abs(nearbyObs.position - obstacle.position) < 6: 
                                        obstacle.sign = 0
                                        obstacle.signtime = -1
                                        break

                            elif obstacle.lane == 3:
                                obstacle.sign = -1 # Change lane to right
                                for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                    if abs(nearbyObs.position - obstacle.position) < 6:
                                        obstacle.sign = 0
                                        obstacle.signtime = -1
                                        break
                            
                            else:
                                obstacle.sign = 1 # Change lane to left
                                LABEL = 1 # OK to change lane
                                for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                    if abs(nearbyObs.position - obstacle.position) < 6:
                                        obstacle.sign = 0
                                        obstacle.signtime = -1
                                        LABEL = 0
                                        break
                                    
                                if LABEL == 0: # Cannot change to left, try changing to right
                                    obstacle.sign = -1
                                    for nearbyObs in self.manager.holding_system[obstacle.lane+obstacle.sign]:
                                        if abs(nearbyObs.position - obstacle.position) < 6:
                                            obstacle.sign = 0
                                            obstacle.signtime = -1
                                            LABEL = 0
                                            break

                            FINISH = 1 # current obstacle update done, continue to the next obstacle
                            break
                    
            if FINISH:
                continue

            # No obstacles ahead
            if obstacle.speed < obstacle.target_speed:
                obstacle.sign = 0
                obstacle.signtime = -1
                obstacle.act((_, 1))
            else:
                obstacle.sign = 0
                obstacle.signtime = -1
                obstacle.DTact('maintain')
            #  --------------------- Decision Tree Ends Here------------------------

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
        observation = [self.ego.speed/self.max_speed, (self.ego.acceleration+6)/16, (self.ego.lane+1)/4]
        for x in self.nearest_obstacles:
            observation.extend([(self.ego.position-x.position)**2 + 9*(self.ego.lane-x.lane)**2, \
                                    x.speed/self.max_speed, x.lane/3, x.sign])
        observation = np.array(observation, dtype=np.float32)
        # print(f"Shape of returned _get_observation: {observation.shape}")
        # print(f"returned _get_observation: {observation}")
        return observation
    
    def render(self, mode='human'):
        if mode == 'human':
        #     fig, ax = plt.subplots(figsize=(10, 5))
            plt.cla()
            # stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            ax.set_xlim([self.ego.position-10, self.ego.position+100])
            ax.set_ylim([-(self.num_lanes * self.lane_width), 0])
            ax.set_xlabel('Position')
            ax.set_ylabel('Lane')
            ax.set_facecolor('#d3d3d3')  # Set the background color to grey
            ax.set_aspect('equal')
            for i in range(self.num_lanes):  # draw lane line
                y = -i * self.lane_width
                ax.axhline(y=y, color='w', linestyle='--')

            # Plot ego vehicle
            ego_vehicle = patches.Rectangle((self.ego.position - self.car_length / 2, -2-self.ego.lane * self.lane_width - self.car_width / 2),
                                             self.car_length, self.car_width, fc='b', label='Ego Vehicle')
            ax.add_patch(ego_vehicle)

            # Plot obstacles
            for obstacle in self.obstacles:
                obstacle_vehicle = patches.Rectangle((obstacle.position - self.car_length / 2, -2-obstacle.lane * self.lane_width - self.car_width / 2),
                                                     self.car_length, self.car_width, fc='r', label='Obstacle')
                ax.add_patch(obstacle_vehicle)
                if obstacle.sign == 1: # Change lane to right
                    arrow = patches.Arrow(obstacle.position - self.car_length / 4,  -2-obstacle.lane * self.lane_width, 0, -2, width=1, color='yellow')
                    ax.add_patch(arrow)
                if obstacle.sign == -1: # Change lane to right
                    arrow = patches.Arrow(obstacle.position - self.car_length / 4,  -2-obstacle.lane * self.lane_width, 0, 2, width=1, color='yellow')
                    ax.add_patch(arrow)

            # Plot nearest obstacles
            for obstacle in self.nearest_obstacles:
                obstacle_vehicle = patches.Rectangle((obstacle.position - self.car_length / 2, -2-obstacle.lane * self.lane_width - self.car_width / 2),
                                                     self.car_length, self.car_width, fc='g', label='Nearest Obstacle')
                ax.add_patch(obstacle_vehicle)
                if obstacle.sign == 1: # Change lane to right
                    arrow = patches.Arrow(obstacle.position - self.car_length / 4,  -2-obstacle.lane * self.lane_width, 0, -2, width=1, color='yellow')
                    ax.add_patch(arrow)
                if obstacle.sign == -1: # Change lane to right
                    arrow = patches.Arrow(obstacle.position - self.car_length / 4,  -2-obstacle.lane * self.lane_width, 0, 2, width=1, color='yellow')
                    ax.add_patch(arrow)
            # Set legend
            # ax.legend()

            plt.title(f'Step: {self.time_step}, Speed: {self.ego.speed:.2f}, Lane: {self.ego.lane}')
            # plt.show(block=False)
            plt.pause(0.01)

            # Update the environment for one time step
            observation, reward, done, info = self.step(0)

            self.time_step += 1

            # Check if the episode is over
            if self.time_step >= self.max_time_step or done:
                self.reset()
                return

            # Render the updated visualization
            self.render()

if __name__=='__main__':
    env = HighwayEnv()
    fig, ax = plt.subplots(figsize=(10, 5))
    # env.reset()
    plt.show(block=False)
    env.render()
    for i in range(len(env.manager.holding_system)):
        laneObs = []
        for o in env.manager.holding_system[i]:
            laneObs.append(o.position)
        laneObs = sorted(laneObs)
        print(f"At lane {i}, the positions of the obstacles are {laneObs}")
    # Print the initial state of the ego 
    # print(f"Timestep {env.time_step}:")
    # print(f"Ego's position:{env.ego.position}\nEgo's speed: {env.ego.speed}\nEgo's acc: {env.ego.acceleration}\nEgo's lane: {env.ego.lane}")
    # Print the initial state of the obstacles

    # for i in range(len(env.obstacles)):
    #     obs = env.obstacles[i]
    #     print(f"Vehicle_{i}'s position:{obs.position}\nVehicle_{i}'s speed: {obs.speed}\nVehicle_{i}'s lane: {obs.lane}")

    obs, reward, done, _ = env.step(('maintain', 0))
    # print(f"Timestep {env.time_step}:")
    # print(f"Ego's position:{env.ego.position}\nEgo's speed: {env.ego.speed}\nEgo's acc: {env.ego.acceleration}\nEgo's lane: {env.ego.lane}\n")
    # for i in range(len(env.nearest_obstacles)):
    #     obs = env.nearest_obstacles[i]
    #     print(f"Nearest vehicle_{i}'s position:{obs.position}\nVehicle_{i}'s speed: {obs.speed}\nVehicle_{i}'s lane: {obs.lane}\n")

    print(f"Reward = {reward}, done = {done}")
    for i in range(len(env.manager.holding_system)):
        laneObs = []
        for o in env.manager.holding_system[i]:
            laneObs.append(o.position)
        laneObs = sorted(laneObs)
        print(f"At lane {i}, the positions of the obstacles are {laneObs}")