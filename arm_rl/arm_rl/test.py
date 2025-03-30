import rclpy
from stable_baselines3 import PPO
from arm_env import ArmEnv

def test_model(model_path="rl_models/final_model.zip", num_episodes=10):
    rclpy.init()
    env = ArmEnv(curriculum_learning=False, debug_mode=True)  # Max difficulty, debug on
    model = PPO.load(model_path, env=env)
    print(f"Loaded model from {model_path}")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        print(f"Episode {episode + 1} started with target position: {env.target_ee_pos}")
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic for best performance
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            ee_pos = env._get_ee_position()
            print(f"Step {steps}: Reward={reward:.4f}, EE Pos={ee_pos}, Target={env.target_ee_pos}")
        
        print(f"Episode {episode + 1} finished: Reward={total_reward:.4f}, Steps={steps}")
    
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    test_model()