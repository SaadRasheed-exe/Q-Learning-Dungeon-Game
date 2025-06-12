from .policy import Policy
from .visualizer import plot_action_grid

class Agent:
    def __init__(self):
        self.policy = Policy()

    def choose_action(self, state):
        return self.policy.get_action(state)

    def learn(self, state, action, reward, next_state, update_epsilon):
        self.policy.update_policy(state, action, reward, next_state, update_epsilon)
    
    def summary(self):
        plot_action_grid(self.policy.q_table)


class AgentV2:
    def __init__(self):
        self.policies = [Policy() for _ in range(3)]

    def select_policy(self, c1, c2):
        if not (c1 or c2):
            return self.policies[0]
        elif c1 and not c2:
            return self.policies[1]
        elif c1 and c2:
            return self.policies[2]
        else:
            raise ValueError("Invalid state: {} {}".format(c1, c2))

    def choose_action(self, state):
        x, y, c1, c2 = state
        policy = self.select_policy(c1, c2)
        return policy.get_action((x, y))

    def learn(self, state, action, reward, next_state, update_epsilon):
        x, y, c1, c2 = state
        next_x, next_y, _, _ = next_state
        policy = self.select_policy(c1, c2)
        policy.update_policy((x, y), action, reward, (next_x, next_y), update_epsilon)

    def summary(self):
        for policy in self.policies:
            plot_action_grid(policy.q_table)