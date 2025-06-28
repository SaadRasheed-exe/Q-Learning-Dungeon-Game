import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from .config import EnvConfig as envcfg
from .config import LearningConfig as lcfg

def _get_next_counter(experiment_path, base_name):
    counter = 0
    while os.path.exists(f'{experiment_path}/{base_name}_{counter}.png'):
        counter += 1
    return counter

def _draw_action_arrow(x, y, action):
    if action == envcfg.UP:
        plt.arrow(y + 0.5, x + 0.5, 0, -0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    elif action == envcfg.DOWN:
        plt.arrow(y + 0.5, x + 0.5, 0, 0.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    elif action == envcfg.LEFT:
        plt.arrow(y + 0.5, x + 0.5, -0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    elif action == envcfg.RIGHT:
        plt.arrow(y + 0.5, x + 0.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

def _setup_grid_plot(title="Policy Visualization"):
    plt.xticks(np.arange(envcfg.GRID_SIZE[1] + 1), np.arange(envcfg.GRID_SIZE[1] + 1))
    plt.yticks(np.arange(envcfg.GRID_SIZE[0] + 1), np.arange(envcfg.GRID_SIZE[0] + 1))
    plt.grid()
    plt.title(title)


class PolicyVisualizer:
    def __init__(self, experiment_path=None, show_plots=None):
        self.experiment_path = experiment_path
        self.show_plots = show_plots if show_plots is not None else lcfg.VISUALIZE_RESULTS
    
    def visualize_policy(self, q_table, suffix="", title=None):
        if not q_table:
            print("Warning: Empty Q-table provided")
            return
            
        state_sample = next(iter(q_table.keys()))
        if len(state_sample) == 2:
            self._visualize_simple_policy(q_table, suffix, title)
        else:
            self._visualize_multi_state_policy(q_table, suffix, title)
    
    def visualize_training_progress(self, epsilon_history, reward_history=None, suffix=""):
        self._visualize_epsilon_decay(epsilon_history, suffix)
        if reward_history is not None:
            self._visualize_reward_history(reward_history, suffix)
    
    def visualize_value_heatmap(self, q_table, suffix="", title=None):
        if not q_table:
            print("Warning: Empty Q-table provided")
            return
            
        state_sample = next(iter(q_table.keys()))
        if len(state_sample) == 2:
            self._visualize_simple_value_heatmap(q_table, suffix, title)
        else:
            self._visualize_multi_state_value_heatmap(q_table, suffix, title)
    
    def _visualize_simple_policy(self, q_table, suffix="", title=None):
        fig = plt.figure(figsize=(8, 8))
        
        for state, actions in q_table.items():
            if all(a == 0 for a in actions):
                continue
            x, y = state
            best_action = np.argmax(actions)
            _draw_action_arrow(x, y, best_action)
        
        _setup_grid_plot(title or "Policy Visualization")
        self._save_and_show(fig, f"policy_simple{suffix}")
    
    def _visualize_multi_state_policy(self, q_table, suffix="", title=None):
        # Group states by condition
        state_groups = self._group_states_by_condition(q_table)
        
        fig = plt.figure(figsize=(15, 5))
        for i, (condition, states) in enumerate(state_groups.items()):
            plt.subplot(1, len(state_groups), i + 1)
            
            for state, actions in states.items():
                x, y = state[:2]  # Extract position coordinates
                best_action = np.argmax(actions)
                _draw_action_arrow(x, y, best_action)
            
            _setup_grid_plot(f"Policy: {condition}")
        
        plt.tight_layout()
        self._save_and_show(fig, f"policy_multi{suffix}")
    
    def _visualize_epsilon_decay(self, epsilon_history, suffix=""):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(epsilon_history, label='Epsilon Decay', color='orange', linewidth=2)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Value')
        plt.title('Exploration Rate Decay Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        self._save_and_show(fig, f"epsilon_decay{suffix}")
    
    def _visualize_reward_history(self, reward_history, suffix="", downsample_factor=1):
        if downsample_factor > 1:
            reward_history = reward_history[::downsample_factor]
            episodes = np.arange(len(reward_history)) * downsample_factor
        else:
            episodes = np.arange(len(reward_history))
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(episodes, reward_history, alpha=0.7, color='green', linewidth=1)
        
        if len(reward_history) > 100:
            window = min(100, len(reward_history) // 10)
            moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
            moving_episodes = episodes[window-1:]
            plt.plot(moving_episodes, moving_avg, color='red', linewidth=2, 
                    label=f'Moving Average (window={window})')
            plt.legend()
        
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Training Reward Progress')
        plt.grid(True, alpha=0.3)
        
        self._save_and_show(fig, f"reward_history{suffix}")
    
    def _group_states_by_condition(self, q_table):
        groups = {}
        for state, actions in q_table.items():
            if len(state) == 4:
                _, _, c1, c2 = state
                condition = f"{int(c1)}{int(c2)} keys"
            else:
                condition = "default"
            
            if condition not in groups:
                groups[condition] = {}
            groups[condition][state] = actions
        
        return groups
    
    def _visualize_simple_value_heatmap(self, q_table, suffix="", title=None):
        value_grid = np.zeros(envcfg.GRID_SIZE)
        value_grid.fill(np.nan)
        
        for state, actions in q_table.items():
            x, y = state
            if 0 <= x < envcfg.GRID_SIZE[0] and 0 <= y < envcfg.GRID_SIZE[1]:
                value_grid[x, y] = np.max(actions)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(value_grid, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='viridis', 
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'label': 'Value Function (Max Q-value)'},
                    ax=ax)
        
        ax.set_title(title or "Value Function Heatmap")
        ax.set_xlabel("Y Coordinate")
        ax.set_ylabel("X Coordinate")
        ax.invert_yaxis()
        
        self._save_and_show(fig, f"value_heatmap{suffix}")
    
    def _visualize_multi_state_value_heatmap(self, q_table, suffix="", title=None):
        state_groups = self._group_states_by_condition(q_table)
        
        n_conditions = len(state_groups)
        fig, axes = plt.subplots(1, n_conditions, figsize=(6 * n_conditions, 6))
        
        if n_conditions == 1:
            axes = [axes]
        
        for idx, (condition, states) in enumerate(state_groups.items()):
            value_grid = np.zeros(envcfg.GRID_SIZE)
            value_grid.fill(np.nan)
            
            for state, actions in states.items():
                x, y = state[:2]
                if 0 <= x < envcfg.GRID_SIZE[0] and 0 <= y < envcfg.GRID_SIZE[1]:
                    value_grid[x, y] = np.max(actions)
            
            sns.heatmap(value_grid,
                        annot=True,
                        fmt='.2f',
                        cmap='viridis',
                        center=0,
                        square=True,
                        linewidths=0.5,
                        cbar_kws={'label': 'Value Function'},
                        ax=axes[idx])
            
            axes[idx].set_title(f"Value Function: {condition}")
            axes[idx].set_xlabel("Y Coordinate")
            axes[idx].set_ylabel("X Coordinate")
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        self._save_and_show(fig, f"multi_value_heatmap{suffix}")
    
    def _save_and_show(self, fig, base_filename):
        if self.show_plots:
            plt.show()
        
        if self.experiment_path:
            counter = _get_next_counter(self.experiment_path, base_filename)
            filename = f"{base_filename}_{counter}.png"
            fig.savefig(f"{self.experiment_path}/{filename}", dpi=300, bbox_inches='tight')
        
        plt.close(fig)

if __name__ == "__main__":
    q_table = {
        (0, 0): [0.1, 0.2, 0.3, 0.4],
        (0, 1): [0.2, 0.1, 0.4, 0.3],
        (1, 0): [0.3, 0.4, 0.1, 0.2],
        (1, 1): [0.4, 0.3, 0.2, 0.1],
    }
    visualizer = PolicyVisualizer(experiment_path="./visualizations", show_plots=True)
    visualizer.visualize_policy(q_table)
    visualizer.visualize_value_heatmap(q_table)