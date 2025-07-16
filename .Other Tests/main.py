import numpy as np

from CustomEnv import createEnv
from QLearning import train_q_learning, visualize_q_table, test_q_learning

# ? Setting Flags
# ! train = True : System will train the agent. train = False : It won't train.
train = False
# ! visualize_results = True : Save the learning data as image. visualize_results = False : It won't save.
visualize_results = train
# ! test = True : Only exploit using q_table.npy. test = False : It won't test.
test = True
# ! render = True : Render the Environment. render = False : It won't render.
render = True
# ! sound = True : Play sound. sound = False : It won't play.
sound = True
# TODO: "random_initialization" for future development. Initialize agent randomly.
random_initialization = False  

# ! Training Values
learning_rate = 0.01    # ? Learning rate
gamma = 0.99            # ? Discount factor
epsilon = 1.0           # ? Exploration rate
epsilon_min = 0.1       # ? Minimum exploration rate
epsilon_decay = 0.995   # ? Decay rate for exploration
no_episodes = 1_000     # ? Number of episodes

# ! Environmental Values
goal_coordinates = {
    'Bar1' : np.array([3, 8]), 
    'Bar2' : np.array([4, 8]), 
    'Bar3' : np.array([5, 8])
}

danger_coordinates = [
    {"coordinates": (3, 2), "role": "D"},
    {"coordinates": (5, 2), "role": "D"},
    {"coordinates": (1, 5), "role": "D"},
    {"coordinates": (4, 4), "role": "D"},
    {"coordinates": (7, 5), "role": "D"},
    {"coordinates": (4, 6), "role": "GK"}
]

# ! Training
if train:
    env = createEnv(goal_coordinates = goal_coordinates,
                    danger_coordinates = danger_coordinates,
                    random_initialization = random_initialization,
                    sound = sound)

    train_q_learning(env = env,
                    no_episodes = no_episodes,
                    epsilon = epsilon,
                    epsilon_min = epsilon_min,
                    epsilon_decay = epsilon_decay,
                    alpha = learning_rate,
                    gamma = gamma,
                    random_init = random_initialization,
                    render = render)

# ! Visualizing
if visualize_results:
    visualize_q_table(danger_coordinates = danger_coordinates,
                      goal_coordinates = goal_coordinates,
                      q_values_path = "q_table.npy")

# ! Testing   
if test:
    env = createEnv(goal_coordinates = goal_coordinates,
                    danger_coordinates = danger_coordinates,
                    random_initialization = random_initialization,
                    sound = sound)
    
    test_q_learning(env = env, 
                    q_table_path = "q_table.npy", 
                    render = render)

