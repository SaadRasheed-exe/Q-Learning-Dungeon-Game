import matplotlib.pyplot as plt
import numpy as np
from .config import EnvConfig as cfg

def plot_action_grid(q_table):
    """
    Plots the action values from the Q-table in a grid format.
    Args:
        q_table (dict): The Q-table where keys are states and values are lists of action values.
    """

    state, actions = next(iter(q_table.items()))
    if len(state) != 2:
        plot_action_values_2(q_table)
        return

    best_action_grid = np.full(cfg.GRID_SIZE, -1)
    plt.figure(figsize=(6, 6))

    for state, actions in q_table.items():
        x, y = state
        
        best_action = np.argmax(actions)
        best_action_grid[x, y] = best_action

        # Draw arrows for each action on the grid
        if best_action == cfg.UP:
            plt.arrow(y + 0.5, x + 0.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif best_action == cfg.DOWN:
            plt.arrow(y + 0.5, x + 0.5, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif best_action == cfg.LEFT:
            plt.arrow(y + 0.5, x + 0.5, -0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif best_action == cfg.RIGHT:
            plt.arrow(y + 0.5, x + 0.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    plt.xticks(np.arange(cfg.GRID_SIZE[1] + 1), np.arange(cfg.GRID_SIZE[1] + 1))
    plt.yticks(np.arange(cfg.GRID_SIZE[0] + 1), np.arange(cfg.GRID_SIZE[0] + 1))
    plt.grid()
    plt.title("Best Action Grid")
    plt.show()

def plot_action_values_2(q_table):

    best_action_grids = [np.full(cfg.GRID_SIZE, -1) for _ in range(3)]

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

    plt.figure(figsize=(12, 12))
    for i, grid in enumerate(best_action_grids):
        plt.subplot(2, 2, i + 1)

        for (x, y), best_action in np.ndenumerate(grid):
            if best_action == cfg.UP:
                plt.arrow(y + 0.5, x + 0.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif best_action == cfg.DOWN:
                plt.arrow(y + 0.5, x + 0.5, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif best_action == cfg.LEFT:
                plt.arrow(y + 0.5, x + 0.5, -0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            elif best_action == cfg.RIGHT:
                plt.arrow(y + 0.5, x + 0.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

        plt.title(f"Best action for condition {i}")
        plt.xticks(np.arange(cfg.GRID_SIZE[1] + 1), np.arange(cfg.GRID_SIZE[1] + 1))
        plt.yticks(np.arange(cfg.GRID_SIZE[0] + 1), np.arange(cfg.GRID_SIZE[0] + 1))
        plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    q_table = {
        (0, 0): [0.1, 0.2, 0.3, 0.4],
        (0, 1): [0.2, 0.1, 0.4, 0.3],
        (1, 0): [0.3, 0.4, 0.1, 0.2],
        (1, 1): [0.4, 0.3, 0.2, 0.1],
        # Add more states as needed
    }
    plot_action_grid(q_table)