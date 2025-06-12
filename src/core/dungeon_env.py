import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from time import sleep
from .config import EnvConfig as cfg

class DungeonEnv(gym.Env):

    
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([10, 10, 1, 1]),
            dtype=np.int32
        )

    def reset(self):
        self.grid = np.full(shape=cfg.GRID_SIZE, fill_value=cfg.LAVA, dtype=np.int32)
        self.agent_pos = np.array(list(cfg.INITIAL_AGENT_POSITION), dtype=np.int32)
        self.carrying = np.zeros(2, dtype=np.int32)

        # walls
        self.grid[5, 5:] = cfg.WALL
        self.grid[3:8, 5] = cfg.WALL
        self.grid[3, :5] = cfg.WALL
        self.grid[7, :5] = cfg.WALL
        self.grid[6, 0] = cfg.WALL
        self.grid[4, 0] = cfg.WALL

        # walkable
        self.grid[1:10, 2] = cfg.WALKABLE
        self.grid[5, 0:5] = cfg.WALKABLE
        self.grid[4, 1:5] = cfg.WALKABLE
        self.grid[6, 1:5] = cfg.WALKABLE
        self.grid[9, 2:10] = cfg.WALKABLE
        self.grid[1, 2:10] = cfg.WALKABLE

        # agent position
        self.grid[cfg.INITIAL_AGENT_POSITION] = cfg.AGENT

        # keys
        self.grid[1, 9] = cfg.KEY
        self.grid[9, 9] = cfg.KEY

        # goal
        self.grid[cfg.GOAL_POSITION] = cfg.GOAL

        self.state = (
            self.agent_pos[0],
            self.agent_pos[1],
            self.carrying[0],
            self.carrying[1]
        )

        return self.state

    def step(self, action):
        if action == cfg.UP:
            new_pos = self.agent_pos + np.array([-1, 0])
        elif action == cfg.DOWN:
            new_pos = self.agent_pos + np.array([1, 0])
        elif action == cfg.LEFT:
            new_pos = self.agent_pos + np.array([0, -1])
        elif action == cfg.RIGHT:
            new_pos = self.agent_pos + np.array([0, 1])
        else:
            raise ValueError(f"Invalid action: {action}")

        new_pos = np.clip(new_pos, 0, np.array(cfg.GRID_SIZE) - 1)
        new_cell = self.grid[tuple(new_pos)]

        reward = 0
        done = False
        terminated = False

        if new_cell == cfg.WALL:
            reward = -0.1

        elif new_cell == cfg.LAVA:
            reward = -10
            terminated = True

        elif new_cell == cfg.KEY:
            if self.carrying[0]:
                self.carrying[1] = 1
            else:
                self.carrying[0] = 1
            reward = 10
            self.grid[tuple(self.agent_pos)] = cfg.WALKABLE
            self.grid[tuple(new_pos)] = cfg.AGENT
            self.agent_pos = new_pos

        elif new_cell == cfg.GOAL:
            if self.carrying[0] and self.carrying[1]:
                done = True
                reward = 100
                self.grid[tuple(self.agent_pos)] = cfg.WALKABLE
                self.grid[tuple(new_pos)] = cfg.AGENT
                self.agent_pos = new_pos
            else:
                reward = -1

        elif new_cell == cfg.WALKABLE:
            reward = -0.1
            self.grid[tuple(self.agent_pos)] = cfg.WALKABLE
            self.grid[tuple(new_pos)] = cfg.AGENT
            self.agent_pos = new_pos

        elif new_cell == cfg.AGENT:
            reward = -1

        else:
            raise ValueError("Invalid cell type.")

        self.state = (
            self.agent_pos[0],
            self.agent_pos[1],
            self.carrying[0],
            self.carrying[1]
        )
        
        return self.state, reward, done, terminated, {}

    def render(self):
        
        grid_display = np.array(self.grid, dtype=object)

        for i in range(cfg.GRID_SIZE[0]):
            for j in range(cfg.GRID_SIZE[1]):
                if grid_display[i, j] == cfg.WALL:
                    grid_display[i, j] = 'W'
                elif grid_display[i, j] == cfg.WALKABLE:
                    grid_display[i, j] = ' '
                elif grid_display[i, j] == cfg.AGENT:
                    grid_display[i, j] = 'A'
                elif grid_display[i, j] == cfg.KEY:
                    grid_display[i, j] = 'K'
                elif grid_display[i, j] == cfg.LAVA:
                    grid_display[i, j] = 'L'
                elif grid_display[i, j] == cfg.GOAL:
                    grid_display[i, j] = 'G'

        ax, ay = self.agent_pos
        grid_display[ax, ay] = 'A'

        print("\n----ENVIRONMENT----")
        for row in grid_display:
            print(" ".join(f"{str(cell):3}" for cell in row))
        print("-------------------")


if __name__ == "__main__":
    env = DungeonEnv()
    state = env.reset()
    done = False
    terminated = False
    while not done and not terminated:
        env.render()

        # # random action
        # action = random.choice([UP, DOWN, LEFT, RIGHT])
        
        # manual action
        action = int(input(f"Enter action ({cfg.UP}: UP, {cfg.DOWN}: DOWN, {cfg.LEFT}: LEFT, {cfg.RIGHT}: RIGHT): "))

        state, reward, done, terminated, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        sleep(0.5)