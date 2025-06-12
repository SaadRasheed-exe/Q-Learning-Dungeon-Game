import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from time import sleep
from .config import EnvConfig as envcfg

class DungeonEnv(gym.Env):

    
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([10, 10, 1, 1]),
            dtype=np.int32
        )

    def reset(self):
        self.grid = np.full(shape=envcfg.GRID_SIZE, fill_value=envcfg.LAVA, dtype=np.int32)
        self.agent_pos = np.array(list(envcfg.INITIAL_AGENT_POSITION), dtype=np.int32)
        self.carrying = np.zeros(2, dtype=np.int32)

        # walls
        self.grid[5, 5:] = envcfg.WALL
        self.grid[3:8, 5] = envcfg.WALL
        self.grid[3, :5] = envcfg.WALL
        self.grid[7, :5] = envcfg.WALL
        self.grid[6, 0] = envcfg.WALL
        self.grid[4, 0] = envcfg.WALL

        # walkable
        self.grid[1:10, 2] = envcfg.WALKABLE
        self.grid[5, 0:5] = envcfg.WALKABLE
        self.grid[4, 1:5] = envcfg.WALKABLE
        self.grid[6, 1:5] = envcfg.WALKABLE
        self.grid[9, 2:10] = envcfg.WALKABLE
        self.grid[1, 2:10] = envcfg.WALKABLE

        # agent position
        self.grid[envcfg.INITIAL_AGENT_POSITION] = envcfg.AGENT

        # keys
        self.grid[1, 9] = envcfg.KEY
        self.grid[9, 9] = envcfg.KEY

        # goal
        self.grid[envcfg.GOAL_POSITION] = envcfg.GOAL

        self.state = (
            self.agent_pos[0],
            self.agent_pos[1],
            self.carrying[0],
            self.carrying[1]
        )

        return self.state

    def step(self, action):
        if action == envcfg.UP:
            new_pos = self.agent_pos + np.array([-1, 0])
        elif action == envcfg.DOWN:
            new_pos = self.agent_pos + np.array([1, 0])
        elif action == envcfg.LEFT:
            new_pos = self.agent_pos + np.array([0, -1])
        elif action == envcfg.RIGHT:
            new_pos = self.agent_pos + np.array([0, 1])
        else:
            raise ValueError(f"Invalid action: {action}")

        new_pos = np.clip(new_pos, 0, np.array(envcfg.GRID_SIZE) - 1)
        new_cell = self.grid[tuple(new_pos)]

        reward = 0
        done = False
        terminated = False

        if new_cell == envcfg.WALL:
            reward = -0.1

        elif new_cell == envcfg.LAVA:
            reward = -10
            terminated = True

        elif new_cell == envcfg.KEY:
            if self.carrying[0]:
                self.carrying[1] = 1
            else:
                self.carrying[0] = 1
            reward = 10
            self.grid[tuple(self.agent_pos)] = envcfg.WALKABLE
            self.grid[tuple(new_pos)] = envcfg.AGENT
            self.agent_pos = new_pos

        elif new_cell == envcfg.GOAL:
            if self.carrying[0] and self.carrying[1]:
                done = True
                reward = 100
                self.grid[tuple(self.agent_pos)] = envcfg.WALKABLE
                self.grid[tuple(new_pos)] = envcfg.AGENT
                self.agent_pos = new_pos
            else:
                reward = -1

        elif new_cell == envcfg.WALKABLE:
            reward = -0.1
            self.grid[tuple(self.agent_pos)] = envcfg.WALKABLE
            self.grid[tuple(new_pos)] = envcfg.AGENT
            self.agent_pos = new_pos

        elif new_cell == envcfg.AGENT:
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

        for i in range(envcfg.GRID_SIZE[0]):
            for j in range(envcfg.GRID_SIZE[1]):
                if grid_display[i, j] == envcfg.WALL:
                    grid_display[i, j] = 'W'
                elif grid_display[i, j] == envcfg.WALKABLE:
                    grid_display[i, j] = ' '
                elif grid_display[i, j] == envcfg.AGENT:
                    grid_display[i, j] = 'A'
                elif grid_display[i, j] == envcfg.KEY:
                    grid_display[i, j] = 'K'
                elif grid_display[i, j] == envcfg.LAVA:
                    grid_display[i, j] = 'L'
                elif grid_display[i, j] == envcfg.GOAL:
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
        action = int(input(f"Enter action ({envcfg.UP}: UP, {envcfg.DOWN}: DOWN, {envcfg.LEFT}: LEFT, {envcfg.RIGHT}: RIGHT): "))

        state, reward, done, terminated, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        sleep(0.5)