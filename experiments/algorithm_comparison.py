"""
Train and compare different RL algorithms for gain scheduling.
Compare SAC with PPO, TD3, and other methods.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from envs.planar_arm_env import PlanarArmEnv


def make_env(env_kwargs=None):
    """Create environment with monitoring."""
    if env_kwargs is None:
        env_kwargs = {}
    env = PlanarArmEnv(**env_kwargs)
    env = Monitor(env)
    return env


def train_algorithm(algorithm, algorithm_name, timesteps=200000, verbose=1):
    """Train a specific algorithm and save results."""
    
    print(f"\nTraining {algorithm_name}...")
    
    # Create environments
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    # Set up model-specific parameters
    if algorithm_name == "SAC":
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=verbose,
            tensorboard_log=f"./logs/algorithm_comparison/{algorithm_name}",
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
        )
    elif algorithm_name == "PPO":
        model = PPO(
            "MlpPolicy", 
            train_env,
            verbose=verbose,
            tensorboard_log=f"./logs/algorithm_comparison/{algorithm_name}",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    elif algorithm_name == "TD3":
        model = TD3(
            "MlpPolicy",
            train_env, 
            verbose=verbose,
            tensorboard_log=f"./logs/algorithm_comparison/{algorithm_name}",
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/algorithm_comparison/{algorithm_name}/",
        log_path=f"./logs/algorithm_comparison/{algorithm_name}/",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    # Train model
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    # Save final model
    os.makedirs(f"models/algorithm_comparison/{algorithm_name}", exist_ok=True)
    model.save(f"models/algorithm_comparison/{algorithm_name}/final_model")
    
    train_env.close()
    eval_env.close()
    
    return model


def evaluate_algorithm(model_path, algorithm_name, episodes=50):
    """Evaluate a trained algorithm."""
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found: {model_path}")
        return None
    
    # Load model
    if algorithm_name == "SAC":
        model = SAC.load(model_path)
    elif algorithm_name == "PPO":
        model = PPO.load(model_path)
    elif algorithm_name == "TD3":
        model = TD3.load(model_path)
    else:
        print(f"Unknown algorithm: {algorithm_name}")
        return None
    
    # Evaluate
    env = PlanarArmEnv()
    returns = []
    success_rates = []
    final_errors = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        
        for t in range(400):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            
            if terminated or truncated:
                break
        
        returns.append(ep_return)
        
        # Calculate final error
        final_q = obs[:2]
        final_q_goal = obs[4:6]
        final_error = np.linalg.norm(final_q_goal - final_q)
        final_errors.append(final_error)
        success_rates.append(1.0 if final_error < 0.1 else 0.0)
    
    env.close()
    
    return {
        'algorithm': algorithm_name,
        'returns': returns,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'final_errors': final_errors,
        'mean_error': np.mean(final_errors),
        'success_rate': np.mean(success_rates),
    }


def run_algorithm_comparison(train_new=True, timesteps=200000):
    """Run comprehensive algorithm comparison."""
    
    algorithms = ["SAC", "PPO", "TD3"]
    
    # Create directories
    os.makedirs("models/algorithm_comparison", exist_ok=True)
    os.makedirs("logs/algorithm_comparison", exist_ok=True)
    
    # Train algorithms
    if train_new:
        trained_models = {}
        for alg in algorithms:
            try:
                model = train_algorithm(alg, alg, timesteps)
                trained_models[alg] = model
                print(f"{alg} training completed successfully!")
            except Exception as e:
                print(f"Error training {alg}: {e}")
                trained_models[alg] = None
    
    # Evaluate all algorithms
    results = {}
    for alg in algorithms:
        model_path = f"models/algorithm_comparison/{alg}/final_model"
        result = evaluate_algorithm(model_path, alg, episodes=50)
        if result:
            results[alg] = result
    
    # Generate comparison plots
    if results:
        plot_algorithm_comparison(results)
        save_comparison_results(results)
    
    return results


def plot_algorithm_comparison(results):
    """Generate comprehensive comparison plots."""
    
    algorithms = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mean Returns
    means = [results[alg]['mean_return'] for alg in algorithms]
    stds = [results[alg]['std_return'] for alg in algorithms]
    
    axes[0,0].bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'green', 'red'])
    axes[0,0].set_title('Mean Episode Returns')
    axes[0,0].set_ylabel('Return')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Success Rates
    success_rates = [results[alg]['success_rate'] * 100 for alg in algorithms]
    axes[0,1].bar(algorithms, success_rates, alpha=0.7, color=['blue', 'green', 'red'])
    axes[0,1].set_title('Success Rates')
    axes[0,1].set_ylabel('Success Rate (%)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Final Positioning Errors
    mean_errors = [results[alg]['mean_error'] for alg in algorithms]
    axes[1,0].bar(algorithms, mean_errors, alpha=0.7, color=['blue', 'green', 'red'])
    axes[1,0].set_title('Mean Final Positioning Error')
    axes[1,0].set_ylabel('Error (rad)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Return Distributions
    return_data = [results[alg]['returns'] for alg in algorithms]
    bp = axes[1,1].boxplot(return_data, labels=algorithms, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1,1].set_title('Return Distributions')
    axes[1,1].set_ylabel('Episode Return')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary table
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*70)
    print(f"{'Algorithm':<12} {'Mean Return':<12} {'Success Rate':<13} {'Mean Error':<12}")
    print("-"*70)
    for alg in algorithms:
        if alg in results:
            mean_ret = results[alg]['mean_return']
            success = results[alg]['success_rate'] * 100
            error = results[alg]['mean_error']
            print(f"{alg:<12} {mean_ret:<12.2f} {success:<13.1f}% {error:<12.4f}")
    print("="*70)


def save_comparison_results(results):
    """Save comparison results to JSON."""
    
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for alg, result in results.items():
        json_results[alg] = {
            'algorithm': result['algorithm'],
            'mean_return': float(result['mean_return']),
            'std_return': float(result['std_return']), 
            'mean_error': float(result['mean_error']),
            'success_rate': float(result['success_rate']),
            'returns': [float(x) for x in result['returns']],
            'final_errors': [float(x) for x in result['final_errors']],
        }
    
    with open('algorithm_comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("Results saved to algorithm_comparison_results.json")


def analyze_sample_efficiency():
    """Analyze sample efficiency by evaluating at different training stages."""
    
    # This would require training with checkpoints and evaluating at different timesteps
    print("Sample efficiency analysis requires checkpoint-based training.")
    print("TODO: Implement training with regular checkpoints for sample efficiency analysis.")


def run_hyperparameter_sensitivity():
    """Test sensitivity to key hyperparameters."""
    
    # SAC hyperparameter variations
    sac_configs = {
        'default': {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'tau': 0.005,
        },
        'high_lr': {
            'learning_rate': 1e-3,
            'batch_size': 256,
            'tau': 0.005,
        },
        'low_lr': {
            'learning_rate': 1e-4,
            'batch_size': 256,
            'tau': 0.005,
        },
        'large_batch': {
            'learning_rate': 3e-4,
            'batch_size': 512,
            'tau': 0.005,
        },
        'high_tau': {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'tau': 0.01,
        },
    }
    
    print("Hyperparameter sensitivity analysis configured.")
    print("Run with shorter timesteps for quick comparison.")
    
    return sac_configs


if __name__ == "__main__":
    print("Algorithm Comparison Study")
    print("="*30)
    
    # Run comparison (set train_new=False if models already exist)
    results = run_algorithm_comparison(train_new=True, timesteps=200000)
    
    if results:
        print(f"\nSuccessfully compared {len(results)} algorithms!")
        print("Check algorithm_comparison.png for results visualization.")
    else:
        print("No algorithms were successfully trained/evaluated.")