import os
import argparse
import numpy as np
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

from envs.planar_arm_env import PlanarArmEnv


def make_env(env_kwargs=None, seed=None):
    """
    Create environment with optional parameters and seeding.
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    env = PlanarArmEnv(**env_kwargs)
    env = Monitor(env)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def train_sac(
    total_timesteps=200_000,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    seed=42,
    env_kwargs=None,
    save_freq=50_000,
    eval_freq=10_000,
    n_eval_episodes=10,
    experiment_name=None,
):
    """Enhanced SAC training with comprehensive monitoring."""
    
    # Set random seed
    set_random_seed(seed)
    
    # Create experiment directory
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"sac_experiment_{timestamp}"
    
    experiment_dir = os.path.join("experiments", experiment_name)
    log_dir = os.path.join(experiment_dir, "logs")
    model_dir = os.path.join(experiment_dir, "models")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")
    
    # Create environments
    if env_kwargs is None:
        env_kwargs = {}
    
    train_env = DummyVecEnv([lambda: make_env(env_kwargs, seed)])
    eval_env = DummyVecEnv([lambda: make_env(env_kwargs, seed + 1)])
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )
    
    # Set up callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="sac_checkpoint",
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Save configuration
    config = {
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'batch_size': batch_size,
        'tau': tau,
        'gamma': gamma,
        'seed': seed,
        'env_kwargs': env_kwargs,
        'save_freq': save_freq,
        'eval_freq': eval_freq,
        'n_eval_episodes': n_eval_episodes,
    }
    
    import json
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train model
    print(f"Training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")
    
    # Backward compatibility - also save to old location
    os.makedirs("models", exist_ok=True) 
    model.save("models/sac_planar_arm")
    
    train_env.close()
    eval_env.close()
    
    return model, experiment_dir


def main():
    """Main training function with argument parsing."""
    
    parser = argparse.ArgumentParser(description="Train SAC for planar arm gain scheduling")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    # Environment parameters
    parser.add_argument("--w_e", type=float, default=1.0, help="Position error weight")
    parser.add_argument("--w_u", type=float, default=0.01, help="Control effort weight") 
    parser.add_argument("--w_dK", type=float, default=0.001, help="Gain variation weight")
    parser.add_argument("--w_j", type=float, default=0.001, help="Jerk weight")
    
    args = parser.parse_args()
    
    # Environment configuration
    env_kwargs = {
        'w_e': args.w_e,
        'w_u': args.w_u,
        'w_dK': args.w_dK,
        'w_j': args.w_j,
    }
    
    # Train model
    model, experiment_dir = train_sac(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        env_kwargs=env_kwargs,
        experiment_name=args.experiment_name,
    )
    
    print(f"\nTraining completed! Experiment saved in: {experiment_dir}")
    print("\nTo analyze results:")
    print(f"  tensorboard --logdir {experiment_dir}/logs")
    print(f"  python experiments/performance_analysis.py --model_path {experiment_dir}/models/final_model")


if __name__ == "__main__":
    main()
