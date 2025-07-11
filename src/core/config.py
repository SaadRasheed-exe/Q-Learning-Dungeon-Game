import os


class EnvConfig:
    GRID_SIZE = (11, 11)
    INITIAL_AGENT_POSITION = (5, 2)
    GOAL_POSITION = (5, 0)

    LAVA = 0
    WALL = 1
    WALKABLE = 2
    AGENT = 3
    KEY = 4
    GOAL = 5

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class LearningConfig:
    MAX_EPISODES = 20000
    LEARNING_RATE = 0.02
    DISCOUNT_FACTOR = 0.99
    EPSILON = 1.0
    EPSILON_DECAY = 0.9995
    MIN_EPSILON = 0.1

    RENDERING_ENABLED = False
    ACTION_SPACE_SIZE = 4

    VISUALIZE_RESULTS = True
    SAVE_RESULTS = True


class RenderingConfig(EnvConfig):
    CELL_SIZE = 16
    UPSCALE_FACTOR = 4
    TICKS_PER_SECOND = 20
    ASSETPATH = "assets/v2-1"


class TestConfig:
    MAX_EPISODES = 1000
    EXPERIMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 "experiments/20250619-113431")