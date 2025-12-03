"""
Hyperparameter tuning and ablation studies for the RL gain scheduler.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.planar_arm_env import PlanarArmEnv


def create_env_with_params(**kwargs):
    """Create environment with specific parameters."""
    env = PlanarArmEnv(**kwargs)
    env = Monitor(env)
    return env


def train_and_evaluate_config(config, config_name, timesteps=100000):
    """Train and evaluate a specific configuration."""
    print(f"\nTraining configuration: {config_name}")
    print(f"Parameters: {config}")
    
    # Create environment with config
    vec_env = DummyVecEnv([lambda: create_env_with_params(**config)])
    
    # Train model
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log=f"./logs/ablation/{config_name}",
    )
    
    model.learn(total_timesteps=timesteps)
    
    # Save model
    model_dir = f"models/ablation/{config_name}"
    os.makedirs(model_dir, exist_ok=True)
    model.save(f"{model_dir}/model")
    
    # Evaluate
    eval_env = PlanarArmEnv(**config)
    returns = []
    
    for ep in range(20):  # Quick evaluation
        obs, _ = eval_env.reset()
        ep_return = 0.0
        
        for t in range(400):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_return += reward
            
            if terminated or truncated:
                break
        
        returns.append(ep_return)
    
    eval_env.close()
    vec_env.close()
    
    result = {
        'config': config,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'returns': returns
    }
    
    # Save results
    with open(f"{model_dir}/results.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results: Mean={result['mean_return']:.2f} ± {result['std_return']:.2f}")
    
    return result


def reward_weight_ablation():
    """Study the effect of different reward weights."""
    
    baseline_config = {
        'w_e': 1.0,      # Position error weight
        'w_u': 0.01,     # Control effort weight  
        'w_dK': 0.001,   # Gain variation weight
        'w_j': 0.001,    # Jerk weight
    }
    
    # Test different weight combinations
    configs = {
        'baseline': baseline_config.copy(),
        'high_position': {**baseline_config, 'w_e': 5.0},
        'high_control': {**baseline_config, 'w_u': 0.1},
        'high_gain_smooth': {**baseline_config, 'w_dK': 0.01},
        'high_jerk': {**baseline_config, 'w_j': 0.01},
        'balanced': {'w_e': 2.0, 'w_u': 0.05, 'w_dK': 0.005, 'w_j': 0.005},
    }
    
    results = {}
    for name, config in configs.items():
        results[name] = train_and_evaluate_config(
            config, f"reward_weights_{name}", timesteps=100000
        )
    
    return results


def scheduler_parameter_ablation():
    """Study the effect of gain scheduler parameters."""
    
    base_env_config = {'w_e': 1.0, 'w_u': 0.01, 'w_dK': 0.001, 'w_j': 0.001}
    
    # Create custom environments with different scheduler parameters
    configs = {
        'eps_1.0': {'eps': 1.0, 'rho': 0.2},
        'eps_3.0': {'eps': 3.0, 'rho': 0.2}, 
        'eps_5.0': {'eps': 5.0, 'rho': 0.2},
        'rho_0.1': {'eps': 2.0, 'rho': 0.1},
        'rho_0.5': {'eps': 2.0, 'rho': 0.5},
        'baseline': {'eps': 2.0, 'rho': 0.2},
    }
    
    # Note: This requires modifying the environment to accept scheduler params
    # For now, we'll document this as a TODO
    
    print("Scheduler parameter ablation requires environment modification.")
    print("TODO: Add scheduler parameter configuration to environment constructor.")
    
    return {}


def baseline_gain_study():
    """Study the effect of different baseline gains Kp0, Kd0."""
    
    # This also requires environment modification
    print("Baseline gain study requires environment modification.")
    print("TODO: Add baseline gain configuration to environment constructor.")
    
    return {}


def learning_curve_analysis():
    """Analyze learning curves for different configurations."""
    
    # This would require training with different logging intervals
    # and analyzing tensorboard logs
    
    print("Learning curve analysis requires tensorboard log processing.")
    print("TODO: Implement tensorboard log parsing and plotting.")
    
    return {}


def run_all_ablations():
    """Run comprehensive ablation studies."""
    
    print("Starting Hyperparameter Tuning and Ablation Studies")
    print("="*60)
    
    os.makedirs("models/ablation", exist_ok=True)
    os.makedirs("logs/ablation", exist_ok=True)
    
    all_results = {}
    
    # 1. Reward weight ablation
    print("\n1. Reward Weight Ablation Study")
    print("-" * 40)
    reward_results = reward_weight_ablation()
    all_results['reward_weights'] = reward_results
    
    # 2. Other ablations (placeholder for now)
    print("\n2. Scheduler Parameter Ablation (TODO)")
    print("-" * 40)
    scheduler_results = scheduler_parameter_ablation()
    all_results['scheduler_params'] = scheduler_results
    
    # Generate summary plots
    plot_ablation_results(all_results)
    
    # Save combined results
    with open('ablation_study_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def plot_ablation_results(results):
    """Plot results from ablation studies."""
    
    if 'reward_weights' in results and results['reward_weights']:
        reward_results = results['reward_weights']
        
        configs = list(reward_results.keys())
        means = [reward_results[c]['mean_return'] for c in configs]
        stds = [reward_results[c]['std_return'] for c in configs]
        
        plt.figure(figsize=(10, 6))
        plt.bar(configs, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Reward Weight Ablation Study')
        plt.xlabel('Configuration')
        plt.ylabel('Mean Episode Return')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reward_weight_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nReward Weight Ablation Results:")
        print("-" * 40)
        for config in configs:
            mean = reward_results[config]['mean_return']
            std = reward_results[config]['std_return']
            print(f"{config:<15}: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    results = run_all_ablations()