from core import DungeonEnv, AgentV2, Agent, Renderer
from core.visualizer import PolicyVisualizer
import numpy as np
import os
import datetime
from core.config import LearningConfig as cfg

def main():
    env = DungeonEnv()
    
    # agent = Agent()      # basic Q-learning agent
    agent = AgentV2()      # multi-policy Q-learning agent

    if cfg.RENDERING_ENABLED:
        renderer = Renderer()
    
    experiment_path = None
    visualizer = None
    if cfg.SAVE_RESULTS:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_path = f"../experiments/{timestamp}/"
        os.makedirs(experiment_path, exist_ok=True)
        visualizer = PolicyVisualizer(experiment_path=experiment_path)

    reward_history = np.zeros(cfg.MAX_EPISODES, dtype=np.float32)
    epsilon_history = []

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

        reward_history[episode] = total_reward
        
        if hasattr(agent, 'policies'):
            epsilon_history.append(agent.policies[0].epsilon)
        elif hasattr(agent, 'policy'):
            epsilon_history.append(agent.policy.epsilon)

        if (episode + 1) % 100 == 0 or episode == cfg.MAX_EPISODES - 1:
            print(f"Episode {episode + 1}/{cfg.MAX_EPISODES}. Total Reward: {total_reward:.2f}. "
                  f"Avg Reward (last 100): {np.mean(reward_history[max(0, episode-99):episode+1]):.2f}")
        elif episode < 10:
            print(f"Episode {episode + 1}/{cfg.MAX_EPISODES}. Total Reward: {total_reward:.2f}.")

    print("Training completed!")
    
    if visualizer:
        print("Generating visualizations...")
        if hasattr(agent, 'policies'):
            for i, policy in enumerate(agent.policies):
                if hasattr(policy, 'q_table'):
                    visualizer.visualize_policy(policy.q_table, suffix=f"_policy_{i}", 
                                              title=f"Policy {i} ({['No Keys', 'One Key', 'Two Keys'][i]})")
        elif hasattr(agent, 'policy') and hasattr(agent.policy, 'q_table'):
            visualizer.visualize_policy(agent.policy.q_table, suffix="_single_policy")
        
        if epsilon_history:
            visualizer.visualize_training_progress(epsilon_history, reward_history)
    
    agent.save(experiment_path)
        
    if experiment_path:
        config_source_path = './core/config.py'
        with open(config_source_path, 'r') as f:
            config_content = f.read()
        with open(f"{experiment_path}/config.py", 'w') as f:
            f.write(config_content)
        
        print(f"Results saved to: {experiment_path}")


if __name__ == "__main__":
    main()