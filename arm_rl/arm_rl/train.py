
from arm_rl.arm_env import ArmEnv
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import rclpy
import logging
import argparse


# Custom callback for tracking goal success
class GoalTrackingCallback(BaseCallback):
    def __init__(self, verbose=0, goal_interval=100):
        super(GoalTrackingCallback, self).__init__(verbose)
        self.goal_interval = goal_interval
        self.episode_count = 0
        self.success_count = 0

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        truncated = self.locals['infos'][0].get('truncated', False)
        if done or truncated:
            self.episode_count += 1
            reward = self.locals['rewards'][0]
            if done and not truncated:
                self.success_count += 1
                self.logger.info(f"Target reached at episode {self.episode_count}! Reward: {reward:.4f}")
            if self.episode_count % self.goal_interval == 0:
                success_rate = self.success_count / self.goal_interval
                self.logger.info(f"After {self.episode_count} episodes: {self.success_count}/{self.goal_interval} reached "
                                 f"(Success rate: {success_rate:.2%})")
                self.success_count = 0
        return True

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('rl_models', exist_ok=True)
os.makedirs('rl_logs', exist_ok=True)

def train(total_timesteps=10000, model_path="rl_models/final_model.zip"):
    rclpy.init()
    logger.info("ROS 2 context initialized")

    # Create environment with fixed joint-space goals
    env = DummyVecEnv([lambda: ArmEnv(
        noise_scale=0.05,
        timeout_steps=50,
        debug_mode=False
    )])
    logger.info(f"Environment created with fixed joint-space goals. Action space: {env.action_space}")

    # Initialize PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="rl_logs/",
        learning_rate=1e-4,
        batch_size=50,
        n_steps=50,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        device='auto'
    )
    logger.info("Initialized new PPO model")

    # Define callbacks
    eval_callback = EvalCallback(eval_env=env, best_model_save_path='rl_models/best_model',
                                 log_path='rl_logs/eval', eval_freq=1000, n_eval_episodes=10,
                                 deterministic=True, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='rl_models/checkpoints',
                                             name_prefix='ppo_arm', verbose=1)
    goal_tracking_callback = GoalTrackingCallback(verbose=1, goal_interval=100)

    try:
        logger.info(f"Training for {total_timesteps} timesteps")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, goal_tracking_callback],
            tb_log_name="PPO",
            progress_bar=True
        )
        logger.info(f"Training completed, total timesteps: {model.num_timesteps}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Train a PPO model for the robotic arm")
    parser.add_argument('--total_timesteps', type=int, default=10000, help="Total training timesteps")
    parser.add_argument('--model_path', type=str, default="rl_models/final_model.zip", help="Path to save the model")
    args = parser.parse_args()

    train(total_timesteps=args.total_timesteps, model_path=args.model_path)

if __name__ == "__main__":
    main()
