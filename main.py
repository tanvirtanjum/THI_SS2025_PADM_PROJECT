# Imports:
# --------
import numpy as np
from CustomEnv import createEnv
from QLearning import train_q_learning, visualize_q_table, test_q_learning

# User definitions:
# -----------------
train = True
visualize_results = True
test = False

"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
random_initialization = False  # If True, the Q-table will be initialized randomly

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1_000  # Number of episodes

goal_coordinates = {'Bar1' : np.array([3, 8]), 'Bar2' : np.array([4, 8]), 'Bar3' : np.array([5, 8])}

# Define all hell state coordinates as a tuple within a list
 # 1st Line Defenders
#         env.addDanger(coordinates=(3, 2), role="D")
#         env.addDanger(coordinates=(5, 2), role="D")
#         # 2nd Line Defenders
#         env.addDanger(coordinates=(1, 5), role="D")
#         env.addDanger(coordinates=(4, 4), role="D")
#         env.addDanger(coordinates=(7, 5), role="D")
#         #  Goal Keeper
#         env.addDanger(coordinates=(4, 6), role="GK")

danger_coordinates = [
    {"coordinates": (3, 2), "role": "D"},
    {"coordinates": (5, 2), "role": "D"},
    {"coordinates": (1, 5), "role": "D"},
    {"coordinates": (4, 4), "role": "D"},
    {"coordinates": (7, 5), "role": "D"},
    {"coordinates": (4, 6), "role": "GK"}
]


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = createEnv(goal_coordinates=goal_coordinates,
                     danger_coordinates=danger_coordinates,
                     random_initialization=random_initialization,
                     sound=True)

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma,
                     random_init=False,
                     render=True)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(danger_coordinates=danger_coordinates,
                      goal_coordinates=goal_coordinates,
                      q_values_path = "q_table.npy")
    
if test:
    env = createEnv(goal_coordinates=goal_coordinates,
                     danger_coordinates=danger_coordinates,
                     random_initialization=random_initialization,
                     sound=True)
    test_q_learning(env, q_table_path="q_table.npy", render=True)

