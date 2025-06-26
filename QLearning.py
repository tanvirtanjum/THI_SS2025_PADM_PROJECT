import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from FileSystem import FileSystem


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy",
                     random_init=False,
                     render=False):

    # Initialize the Q-table:
    # -----------------------
    if random_init:
        q_table = np.random.rand(env.grid_size, env.grid_size, env.action_space.n)
    else:
        q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    for episode in range(no_episodes):
        state, _ = env.reset()
        
        state = tuple(map(int, state))
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, done, reward, _ = env.step(action)
            if render:
                env.render()
                
            next_state = tuple(map(int, next_state))
            total_reward += reward

            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])
            state = next_state

            if done:
                break


        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Mode: {'Exploit' if np.argmax(q_table[state]) == action else 'Explore'}")

    env.close()
    print("Training finished.\n")
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(danger_coordinates=[
                        {"coordinates": (3, 2), "role": "D"},
                        {"coordinates": (5, 2), "role": "D"},
                        {"coordinates": (1, 5), "role": "D"},
                        {"coordinates": (4, 4), "role": "D"},
                        {"coordinates": (7, 5), "role": "D"},
                        {"coordinates": (4, 6), "role": "GK"}
                    ],
                    goal_coordinates={
                        'Bar1': np.array([3, 8]), 
                        'Bar2': np.array([4, 8]), 
                        'Bar3': np.array([5, 8])
                    },
                    actions=["Up", "Down", "Right", "Left"],
                    q_values_path="q_table.npy"):

    tempDanger = [d['coordinates'] for d in danger_coordinates]
    danger_coordinates = tempDanger

    goal_coordinates = [goal_coordinates["Bar1"], goal_coordinates["Bar2"], goal_coordinates["Bar3"]]

    try:
        q_table = np.load(q_values_path)
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            mask = np.zeros_like(heatmap_data, dtype=bool)
            for coord in goal_coordinates + danger_coordinates:
                mask[coord[0], coord[1]] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            for g in goal_coordinates:
                ax.text(g[1] + 0.5, g[0] + 0.5, 'G', color='green',
                        ha='center', va='center', weight='bold', fontsize=14)
            for d in danger_coordinates:
                ax.text(d[1] + 0.5, d[0] + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')
            
        FS = FileSystem()
        plt.tight_layout()
        plt.savefig("./Learning Data/"+FS.getNewFileName(extension=".png")+".png")
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")


# Function 3: Test with the Q-table
# -----------

def test_q_learning(env, q_table_path="q_table.npy", render=True):
    # Load the trained Q-table
    if os.path.isfile(q_table_path):
        q_table = np.load(q_table_path)
    else:
        q_table = []

    if(q_table is not None and len(q_table)>0):
        state, info = env.reset()
        state = tuple(map(int, state))

        total_reward = 0
        step_count = 0

        while True:
            action = np.argmax(q_table[state])  # Exploit only (no exploration)
            next_state, done, reward, info = env.step(action)
            if render:
                env.render()

            next_state = tuple(map(int, next_state))
            total_reward += reward
            step_count += 1
            state = next_state

            if done:
                break

        env.close()
        print(f"Test completed. Total Reward: {total_reward:.2f}, Steps Taken: {step_count}")
        
    else:
        print("Train the environment first.")
        