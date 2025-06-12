from core import DungeonEnv, AgentV2, Agent, Renderer
from core.config import LearningConfig as cfg
import numpy as np
import os
import datetime

def main():
    env = DungeonEnv()
    agent = AgentV2()
    if cfg.RENDERING_ENABLED:
        renderer = Renderer()
    
    experiment_path = None
    if cfg.SAVE_RESULTS:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_path = f"../experiments/{timestamp}/"
        os.makedirs(experiment_path, exist_ok=True)

    # reward_history = np.zeros(cfg.MAX_EPISODES, dtype=np.float32)

    for episode in range(cfg.MAX_EPISODES):
        state = env.reset()
        done = False
        terminated = False
        total_reward = 0

        while not done and not terminated:
            action = agent.choose_action(state)
            next_state, reward, done, terminated, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done or terminated)
            state = next_state
            total_reward += reward

            if cfg.RENDERING_ENABLED:
                renderer.draw_grid(env.grid)

        # reward_history[episode] = total_reward
        
        if done:
            print("Episode finished successfully.")

        print(f"Episode {episode + 1}/{cfg.MAX_EPISODES}. Total Reward: {total_reward:.2f}.")

    _ = "Breakpoint for debugging purposes. Remove this line to avoid confusion."

    agent.results(experiment_path)
        
    if experiment_path:
        with open(f'core/config.py', 'r') as f:
            config_content = f.read()
        with open(f"{experiment_path}/config.py", 'w') as f:
            f.write(config_content)


if __name__ == "__main__":
    main()