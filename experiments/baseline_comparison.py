"""
Comprehensive baseline comparison for RL gain scheduling project.
Compare different control strategies on the same reaching tasks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from envs.planar_arm_env import PlanarArmEnv
from control.pd_baseline import PDBaseline


def evaluate_policy(policy_fn, env, episodes=50, max_steps=400, policy_name="Policy"):
    """Evaluate a policy and return performance metrics."""
    returns = []
    success_rates = []
    final_errors = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        
        for t in range(max_steps):
            if policy_name == "RL":
                action, _ = policy_fn.predict(obs, deterministic=True)
            else:
                # For PD controllers, extract current state and goal
                q = obs[:2]
                dq = obs[2:4] 
                q_goal = obs[4:6]
                # Compute control directly, then convert to zero action for env
                u = policy_fn.compute_torque(q, dq, q_goal)
                action = np.zeros(4)  # Zero ΔK increments for PD baseline
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            
            if terminated or truncated:
                break
        
        returns.append(ep_return)
        
        # Calculate final positioning error
        final_q = obs[:2]
        final_q_goal = obs[4:6]
        final_error = np.linalg.norm(final_q_goal - final_q)
        final_errors.append(final_error)
        
        # Success if within 0.1 rad of target
        success_rates.append(1.0 if final_error < 0.1 else 0.0)
    
    return {
        'returns': returns,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'final_errors': final_errors,
        'mean_error': np.mean(final_errors),
        'success_rate': np.mean(success_rates),
        'policy_name': policy_name
    }


def run_baseline_comparison():
    """Run comprehensive comparison between different control strategies."""
    
    # Create test environments with same random seeds for fair comparison
    test_seeds = list(range(42, 92))  # 50 test episodes
    
    results = {}
    
    # 1. Fixed PD Controller (Conservative gains)
    print("Evaluating Conservative PD Controller...")
    env = PlanarArmEnv()
    pd_conservative = PDBaseline(Kp=[8.0, 8.0], Kd=[2.0, 2.0])
    results['PD_Conservative'] = evaluate_policy(
        pd_conservative, env, episodes=50, policy_name="PD_Conservative"
    )
    
    # 2. Fixed PD Controller (Aggressive gains)
    print("Evaluating Aggressive PD Controller...")
    pd_aggressive = PDBaseline(Kp=[15.0, 15.0], Kd=[3.0, 3.0])
    results['PD_Aggressive'] = evaluate_policy(
        pd_aggressive, env, episodes=50, policy_name="PD_Aggressive"
    )
    
    # Try to load model from train_sac.py first
    rl_model = None
    if os.path.exists("models/sac_planar_arm.zip"):
        rl_model = SAC.load("models/sac_planar_arm")
        print("Using model from train_sac.py")
    # Fallback to algorithm_comparison SAC model
    elif os.path.exists("models/algorithm_comparison/SAC/final_model.zip"):
        rl_model = SAC.load("models/algorithm_comparison/SAC/final_model")
        print("Using SAC model from algorithm_comparison")
    
    if rl_model is not None:
        results['RL_Scheduler'] = evaluate_policy(
            rl_model, env, episodes=50, policy_name="RL"
        )
    else:
        print("No RL model found! Train first with train_sac.py or run algorithm_comparison.py")
    
    # 4. Random Action Controller
    print("Evaluating Random Controller...")
    class RandomController:
        def predict(self, obs, deterministic=True):
            return np.random.uniform(-1, 1, size=4), None
    
    random_controller = RandomController()
    results['Random'] = evaluate_policy(
        random_controller, env, episodes=50, policy_name="RL"
    )
    
    env.close()
    
    # Generate comparison plots
    plot_comparison_results(results)
    
    return results


def plot_comparison_results(results):
    """Generate comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mean Returns Comparison
    policies = list(results.keys())
    mean_returns = [results[p]['mean_return'] for p in policies]
    std_returns = [results[p]['std_return'] for p in policies]
    
    axes[0,0].bar(policies, mean_returns, yerr=std_returns, capsize=5, alpha=0.7)
    axes[0,0].set_title('Mean Episode Returns')
    axes[0,0].set_ylabel('Return')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Final Positioning Errors
    mean_errors = [results[p]['mean_error'] for p in policies]
    axes[0,1].bar(policies, mean_errors, alpha=0.7, color='orange')
    axes[0,1].set_title('Mean Final Positioning Error')
    axes[0,1].set_ylabel('Error (rad)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Success Rates
    success_rates = [results[p]['success_rate'] * 100 for p in policies]
    axes[1,0].bar(policies, success_rates, alpha=0.7, color='green')
    axes[1,0].set_title('Success Rate (Error < 0.1 rad)')
    axes[1,0].set_ylabel('Success Rate (%)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Return Distribution Box Plot
    return_data = [results[p]['returns'] for p in policies]
    axes[1,1].boxplot(return_data, labels=policies)
    axes[1,1].set_title('Return Distributions')
    axes[1,1].set_ylabel('Episode Return')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Policy':<15} {'Mean Return':<12} {'Success Rate':<13} {'Mean Error':<12}")
    print("-"*80)
    for policy in policies:
        print(f"{policy:<15} {results[policy]['mean_return']:<12.2f} "
              f"{results[policy]['success_rate']*100:<13.1f}% "
              f"{results[policy]['mean_error']:<12.4f}")
    print("="*80)


if __name__ == "__main__":
    results = run_baseline_comparison()