import numpy as np
from EgoDecisionTree import HighwayEnv

env = HighwayEnv()
env.reset()
timestep = 10000

collision_time     = 0
outOfBoundary_time = 0
lane_change_time   = 0
v_list             = []
a_list             = []
num_scenario       = 0

for i in range(timestep):
    _, reward, done, _ = env.step('dummy')
    # env.render()
    if done:
        collision_time     += env.collision_time
        outOfBoundary_time += env.outOfBoundary_time
        lane_change_time   += env.lane_change_time
        v_list             += env.v_list
        a_list             += env.a_list
        num_scenario       += env.num_scenario
        env.reset()
    
    # plt.show(block=False)

collision_time     = env.collision_time
outOfBoundary_time = env.outOfBoundary_time
lane_change_time   = env.lane_change_time
v_list             = env.v_list
a_list             = env.a_list
num_scenario       = env.num_scenario

w_v  = 0.001
w_a  = 0.03
w_j  = 0.1
w_c  = 100
w_lc = 30

j_v  = w_v  * sum([(env.ego.target_speed - v)**2 for v in v_list])
j_a  = w_a  * sum([a**2 for a in a_list])
j_j  = w_j  * sum([(a_list[i+1] - a_list[i])**2 for i in range(len(a_list) - 1)])
j_c  = w_c  * collision_time
j_lc = w_lc * lane_change_time
j_total = j_v + j_a + j_j + j_c + j_lc

print(f"j_v     = {j_v}")
print(f"j_a     = {j_a}")
print(f"j_j     = {j_j}")
print(f"j_c     = {j_c}")
print(f"j_lc    = {j_lc}")
print(f"j_total = {j_total}")


print(f"Out-of-boundary rate: {outOfBoundary_time/env.num_scenario:.2f}")
print(f"Collision rate: {collision_time/env.num_scenario:.2f}")
print(f"Lane-changing rate: {lane_change_time/env.num_scenario:.2f}")
print(f"Number of scenarios: {num_scenario}")