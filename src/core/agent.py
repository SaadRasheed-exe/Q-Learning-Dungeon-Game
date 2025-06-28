from .policy import Policy
import pickle

class Agent:
    def __init__(self, q_table=None):
        if q_table is not None:
            self.policy = Policy(q_table=q_table)
        else:
            self.policy = Policy()

    def choose_action(self, state):
        return self.policy.get_action(state)

    def learn(self, state, action, reward, next_state, update_epsilon):
        self.policy.update_policy(state, action, reward, next_state, update_epsilon)
    
    def save(self, experiment_path=None):
        if experiment_path:
            with open(f"{experiment_path}/q_table.pkl", 'wb') as f:
                pickle.dump(self.policy.q_table, f)


class AgentV2:
    def __init__(self, q_table=None):
        self.policies = [Policy(q_table=q) for q in q_table] if q_table is not None else [Policy() for _ in range(3)]

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

    def save(self, experiment_path=None):
        if experiment_path:
            for i, policy in enumerate(self.policies):
                with open(f"{experiment_path}/q_table_policy_{i}.pkl", 'wb') as f:
                    pickle.dump(policy.q_table, f)

