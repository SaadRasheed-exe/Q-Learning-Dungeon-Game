from core import DungeonEnv, AgentV2, Agent, Renderer
from core.config import LearningConfig as cfg

def main():
    env = DungeonEnv()
    agent = AgentV2()
    if cfg.RENDERING_ENABLED:
        renderer = Renderer()

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

        if done:
            print("Episode finished successfully.")

        print(f"Episode {episode + 1}/{cfg.MAX_EPISODES}. Total Reward: {total_reward:.2f}.")

    _ = "Breakpoint for debugging purposes. Remove this line to avoid confusion."

    agent.summary()

if __name__ == "__main__":
    main()