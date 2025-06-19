import numpy as np
from .config import LearningConfig as cfg

class Policy:
    def __init__(
            self,
            q_table=None,
            learning_rate=cfg.LEARNING_RATE,
            discount_factor=cfg.DISCOUNT_FACTOR,
            epsilon=cfg.EPSILON,
            epsilon_decay=cfg.EPSILON_DECAY,
            min_epsilon=cfg.MIN_EPSILON,
            action_space_size=cfg.ACTION_SPACE_SIZE
        ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = q_table if q_table is not None else {}
        self.action_space_size = action_space_size
        self.epsilon = epsilon if q_table is None else 0.0 
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.epsilon_history = np.zeros(cfg.MAX_EPISODES, dtype=np.float32)
        self.current_episode = 0

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_space_size
        return self.epsilon_greedy_action(state)

    def epsilon_greedy_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.action_space_size))
        else:
            return self.q_table[state].index(max(self.q_table[state]))

    def update_policy(self, state, action, reward, next_state, update_epsilon):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * self.action_space_size
        
        best_next_action = np.argmax(self.q_table[next_state])
        temp = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.learning_rate * (temp - self.q_table[state][action])

        if update_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
            self.epsilon_history[self.current_episode] = self.epsilon
            self.current_episode += 1