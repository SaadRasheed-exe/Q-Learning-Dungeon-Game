from core import DungeonEnv, AgentV2, Agent, Renderer
from core.config import TestConfig as cfg
import pickle as pkl
import glob
import os


def main():

    if cfg.EXPERIMENT_PATH:
        q_tables = glob.glob(os.path.join(cfg.EXPERIMENT_PATH, "q_table_policy_*.pkl"))
        if not q_tables:
            raise FileNotFoundError("No q_table files found in the specified experiment path.")
        elif len(q_tables) != 1 and len(q_tables) != 3:
            raise ValueError("Expected either 1 or 3 Q-table files, found: {}".format(len(q_tables)))

        q_tables = sorted(q_tables, key=lambda x: int(x.split('q_table_policy_')[1].split('.')[0]))
        q_tables_loaded = [pkl.load(open(q_table, 'rb')) for q_table in q_tables]

        if len(q_tables) == 1:
            agent = Agent(q_table=q_tables_loaded[0])
        elif len(q_tables) == 3:
            agent = AgentV2(q_table=q_tables_loaded)


    env = DungeonEnv()
    renderer = Renderer()

    while True:
        state = env.reset()
        done = False
        terminated = False

        while not done and not terminated:
            action = agent.choose_action(state)
            next_state, _, done, terminated, _ = env.step(action)
            state = next_state
            renderer.draw_grid(env.grid)


if __name__ == "__main__":
    main()