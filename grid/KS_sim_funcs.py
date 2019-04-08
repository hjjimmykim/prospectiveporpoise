from Schwab_brain import Agent
import numpy as np
import copy
import torch



# Translates action space to a direction
def get_dir(action):	# NESW
    dir_list = [np.array([-1,0]),np.array([0,1]),np.array([1,0]),np.array([0,-1])]
    return dir_list[action]


# Return queue of surviving agents (the order in which they will act)
def action_queue():
    return np.random.shuffle(np.array([1,2]))\


# Initialize map, agents, and team scores
def initialize_1p(map_in, spawn_loc, input_dim):
    p1 = Agent(1, spawn_loc, input_dim)  # Create agent object and put it in the dictionary
#    p1.memory.wipe()                                    # Reset memory
    map = copy.deepcopy(map_in)
    map[spawn_loc[0],spawn_loc[1]] = 1                  # Record on the map
    return p1, map


# Reset map and team scores, preserve agents
def reset_1p(player, map_in, spawn_loc):
    player.loc = spawn_loc              # Respawn
    player.has_key = False
    player.reward = 0
    map = copy.deepcopy(map_in)
    map[spawn_loc[0],spawn_loc[1]] = 1  # Record on the map
    return map


# Policy update function
def REINFORCE(probability, reward, baseline):
    rl = -torch.log(probability) * (reward-baseline)
    return rl
