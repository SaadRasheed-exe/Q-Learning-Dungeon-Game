import matplotlib.pyplot as plt
import numpy as np
import os
from .config import EnvConfig as envcfg
from .config import LearningConfig as lcfg

def make_action_grid(q_table, experiment_path=None):
    """
    Plots the action values from the Q-table in a grid format.
    Args:
        q_table (dict): The Q-table where keys are states and values are lists of action values.
    """

    state, actions = next(iter(q_table.items()))
    if len(state) != 2:
        make_action_values_2(q_table, experiment_path)
        return

    best_action_grid = np.full(envcfg.GRID_SIZE, -1)
    fig = plt.figure(figsize=(6, 6))

    for state, actions in q_table.items():
        x, y = state
        
        best_action = np.argmax(actions)
        best_action_grid[x, y] = best_action

        # Draw arrows for each action on the grid
        if best_action == envcfg.UP:
            plt.arrow(y + 0.5, x + 0.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif best_action == envcfg.DOWN:
            plt.arrow(y + 0.5, x + 0.5, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif best_action == envcfg.LEFT:
            plt.arrow(y + 0.5, x + 0.5, -0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif best_action == envcfg.RIGHT:
            plt.arrow(y + 0.5, x + 0.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    plt.xticks(np.arange(envcfg.GRID_SIZE[1] + 1), np.arange(envcfg.GRID_SIZE[1] + 1))
    plt.yticks(np.arange(envcfg.GRID_SIZE[0] + 1), np.arange(envcfg.GRID_SIZE[0] + 1))
    plt.grid()
    plt.title("Best Action Grid")

    if lcfg.VISUALIZE_RESULTS:
        plt.show()
    
    if experiment_path:
        imgcounter = 0
        while os.path.exists(f'{experiment_path}/action_grid_{imgcounter}.png'):
            imgcounter += 1
        fig.savefig(f"{experiment_path}/action_grid_{imgcounter}.png")

    plt.close()

def make_action_values_2(q_table, experiment_path=None):

    best_action_grids = [np.full(envcfg.GRID_SIZE, -1) for _ in range(3)]

    for state, actions in q_table.items():
        x, y, c1, c2 = state
        if not (c1 or c2):
            best_action = np.argmax(actions)
            best_action_grids[0][x, y] = best_action
        elif c1 and not c2:
            best_action = np.argmax(actions)
            best_action_grids[1][x, y] = best_action
        elif c1 and c2:
            best_action = np.argmax(actions)
            best_action_grids[2][x, y] = best_action

    fig = plt.figure(figsize=(12, 12))
    for i, grid in enumerate(best_action_grids):
        plt.subplot(2, 2, i + 1)

        for (x, y), best_action in np.ndenumerate(grid):
            if best_action == envcfg.UP:
                plt.arrow(y + 0.5, x + 0.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif best_action == envcfg.DOWN:
                plt.arrow(y + 0.5, x + 0.5, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif best_action == envcfg.LEFT:
                plt.arrow(y + 0.5, x + 0.5, -0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif best_action == envcfg.RIGHT:
                plt.arrow(y + 0.5, x + 0.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

        plt.title(f"Best action for condition {i}")
        plt.xticks(np.arange(envcfg.GRID_SIZE[1] + 1), np.arange(envcfg.GRID_SIZE[1] + 1))
        plt.yticks(np.arange(envcfg.GRID_SIZE[0] + 1), np.arange(envcfg.GRID_SIZE[0] + 1))
        plt.grid()

    plt.tight_layout()

    if lcfg.VISUALIZE_RESULTS:
        plt.show()
    
    if experiment_path:
        fig.savefig(f"{experiment_path}/action_grid.png")

    plt.close()


def make_epsilon_decay(epsilon_history, experiment_path=None):
    """
    Plots the epsilon decay over time.
    Args:
        epsilon_history (list): List of epsilon values over episodes.
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(epsilon_history, label='Epsilon Decay', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value')
    plt.title('Epsilon Decay Over Time')
    plt.legend()
    plt.grid()

    if lcfg.VISUALIZE_RESULTS:
        plt.show()

    if experiment_path:
        imgcounter = 0
        while os.path.exists(f'{experiment_path}/epsilon_decay_{imgcounter}.png'):
            imgcounter += 1
        fig.savefig(f"{experiment_path}/epsilon_decay_{imgcounter}.png")

    plt.close()

def make_reward_history(reward_history, downsample_factor=1, experiment_path=None):
    """
    Plots the reward history over episodes.
    Args:
        reward_history (list): List of total rewards per episode.
    """
    if downsample_factor > 1:
        reward_history = reward_history[::downsample_factor]

    fig = plt.figure(figsize=(10, 5))
    plt.scatter(range(len(reward_history)), reward_history, label='Total Reward', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward History')
    plt.legend()
    plt.grid()

    if lcfg.VISUALIZE_RESULTS:
        plt.show()

    if experiment_path:
        fig.savefig(f"{experiment_path}/reward_history.png")

    plt.close()

if __name__ == "__main__":
    q_table = {
        (0, 0): [0.1, 0.2, 0.3, 0.4],
        (0, 1): [0.2, 0.1, 0.4, 0.3],
        (1, 0): [0.3, 0.4, 0.1, 0.2],
        (1, 1): [0.4, 0.3, 0.2, 0.1],
        # Add more states as needed
    }
    make_action_grid(q_table)