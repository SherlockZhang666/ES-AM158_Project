"""
Comprehensive analysis and visualization tools for RL gain scheduling project.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from stable_baselines3 import SAC
from envs.planar_arm_env import PlanarArmEnv
from envs.enhanced_planar_arm_env import EnhancedPlanarArmEnv


class PerformanceAnalyzer:
    """Comprehensive performance analysis for RL gain scheduling."""
    
    def __init__(self, model_path, env_class=PlanarArmEnv):
        self.model = SAC.load(model_path) if os.path.exists(model_path + ".zip") else None
        self.env_class = env_class
        
    def analyze_trajectory(self, episodes=5, save_data=True):
        """Analyze trajectories with detailed logging."""
        
        env = self.env_class()
        trajectories = []
        
        for ep in range(episodes):
            obs, _ = env.reset()
            
            trajectory = {
                'joint_angles': [],
                'joint_velocities': [], 
                'joint_goals': [],
                'ee_positions': [],
                'ee_goals': [],
                'control_torques': [],
                'effective_gains': [],
                'rewards': [],
                'actions': [],
                'costs': {
                    'position': [],
                    'torque': [], 
                    'jerk': [],
                    'gain_smooth': []
                }
            }
            
            for t in range(400):
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                # Log state before step
                q = obs[:2].copy()
                dq = obs[2:4].copy()
                q_goal = obs[4:6].copy()
                
                trajectory['joint_angles'].append(q)
                trajectory['joint_velocities'].append(dq)
                trajectory['joint_goals'].append(q_goal)
                trajectory['actions'].append(action.copy())
                
                # Compute end-effector positions
                ee_pos = self._compute_forward_kinematics(q)
                ee_goal = self._compute_forward_kinematics(q_goal)
                trajectory['ee_positions'].append(ee_pos)
                trajectory['ee_goals'].append(ee_goal)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Log step results
                trajectory['rewards'].append(reward)
                
                if hasattr(env.scheduler, 'get_effective_gains'):
                    Kp_eff, Kd_eff = env.scheduler.get_effective_gains()
                    trajectory['effective_gains'].append({
                        'Kp': Kp_eff.copy(),
                        'Kd': Kd_eff.copy()
                    })
                
                # Log individual costs
                if 'pos_cost' in info:
                    trajectory['costs']['position'].append(info['pos_cost'])
                    trajectory['costs']['torque'].append(info['torque_cost'])
                    trajectory['costs']['jerk'].append(info['jerk_cost']) 
                    trajectory['costs']['gain_smooth'].append(info.get('dK_cost', 0))
                
                if terminated or truncated:
                    break
            
            # Convert lists to arrays
            for key in ['joint_angles', 'joint_velocities', 'joint_goals', 
                       'ee_positions', 'ee_goals', 'actions', 'rewards']:
                trajectory[key] = np.array(trajectory[key])
            
            for cost_type in trajectory['costs']:
                trajectory['costs'][cost_type] = np.array(trajectory['costs'][cost_type])
            
            trajectories.append(trajectory)
        
        env.close()
        
        if save_data:
            self._save_trajectory_data(trajectories)
        
        return trajectories
    
    def _compute_forward_kinematics(self, q, link_lengths=(1.0, 1.0)):
        """Compute end-effector position."""
        q1, q2 = q
        l1, l2 = link_lengths
        ee_x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2) 
        ee_y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        return np.array([ee_x, ee_y])
    
    def _save_trajectory_data(self, trajectories):
        """Save trajectory data for later analysis."""
        os.makedirs("analysis_results", exist_ok=True)
        
        # Convert to JSON-serializable format
        json_trajectories = []
        for traj in trajectories:
            json_traj = {}
            for key, value in traj.items():
                if key == 'effective_gains':
                    json_traj[key] = [
                        {'Kp': gain['Kp'].tolist(), 'Kd': gain['Kd'].tolist()}
                        for gain in value
                    ]
                elif key == 'costs':
                    json_traj[key] = {k: v.tolist() for k, v in value.items()}
                elif isinstance(value, np.ndarray):
                    json_traj[key] = value.tolist()
                else:
                    json_traj[key] = value
            json_trajectories.append(json_traj)
        
        with open("analysis_results/trajectory_data.json", 'w') as f:
            json.dump(json_trajectories, f, indent=2)
    
    def plot_trajectory_analysis(self, trajectories):
        """Generate comprehensive trajectory plots."""
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Select first trajectory for detailed analysis
        traj = trajectories[0]
        time = np.arange(len(traj['joint_angles']))
        
        # 1. Joint angles over time
        axes[0,0].plot(time, traj['joint_angles'][:, 0], label='q1', alpha=0.7)
        axes[0,0].plot(time, traj['joint_angles'][:, 1], label='q2', alpha=0.7)
        axes[0,0].axhline(traj['joint_goals'][0, 0], color='r', linestyle='--', alpha=0.5, label='q1_goal')
        axes[0,0].axhline(traj['joint_goals'][0, 1], color='orange', linestyle='--', alpha=0.5, label='q2_goal')
        axes[0,0].set_title('Joint Angles')
        axes[0,0].set_ylabel('Angle (rad)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Joint velocities
        axes[0,1].plot(time, traj['joint_velocities'][:, 0], label='dq1', alpha=0.7)
        axes[0,1].plot(time, traj['joint_velocities'][:, 1], label='dq2', alpha=0.7)
        axes[0,1].set_title('Joint Velocities') 
        axes[0,1].set_ylabel('Velocity (rad/s)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. End-effector trajectory
        ee_traj = traj['ee_positions']
        axes[0,2].plot(ee_traj[:, 0], ee_traj[:, 1], 'b-', alpha=0.7, label='EE trajectory')
        axes[0,2].scatter(ee_traj[0, 0], ee_traj[0, 1], color='green', s=100, label='Start', zorder=5)
        axes[0,2].scatter(traj['ee_goals'][0, 0], traj['ee_goals'][0, 1], 
                         color='red', s=100, marker='x', label='Goal', zorder=5)
        axes[0,2].set_title('End-Effector Trajectory')
        axes[0,2].set_xlabel('X Position')
        axes[0,2].set_ylabel('Y Position') 
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].axis('equal')
        
        # 4. RL Actions over time
        axes[1,0].plot(time, traj['actions'][:, 0], label='ΔKp1', alpha=0.7)
        axes[1,0].plot(time, traj['actions'][:, 1], label='ΔKp2', alpha=0.7)
        axes[1,0].plot(time, traj['actions'][:, 2], label='ΔKd1', alpha=0.7)
        axes[1,0].plot(time, traj['actions'][:, 3], label='ΔKd2', alpha=0.7)
        axes[1,0].set_title('RL Actions (Gain Increments)')
        axes[1,0].set_ylabel('Action Value')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Effective gains evolution
        if traj['effective_gains']:
            gains_Kp = np.array([g['Kp'] for g in traj['effective_gains']])
            gains_Kd = np.array([g['Kd'] for g in traj['effective_gains']])
            
            axes[1,1].plot(time[:len(gains_Kp)], gains_Kp[:, 0], label='Kp1', alpha=0.7)
            axes[1,1].plot(time[:len(gains_Kp)], gains_Kp[:, 1], label='Kp2', alpha=0.7) 
            axes[1,1].plot(time[:len(gains_Kd)], gains_Kd[:, 0], label='Kd1', alpha=0.7)
            axes[1,1].plot(time[:len(gains_Kd)], gains_Kd[:, 1], label='Kd2', alpha=0.7)
            axes[1,1].set_title('Effective Gains Evolution')
            axes[1,1].set_ylabel('Gain Value')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Reward components
        axes[1,2].plot(time, traj['rewards'], label='Total Reward', alpha=0.7)
        axes[1,2].set_title('Reward Over Time')
        axes[1,2].set_ylabel('Reward')
        axes[1,2].grid(True, alpha=0.3)
        
        # 7-10. Individual cost components  
        cost_names = ['position', 'torque', 'jerk', 'gain_smooth']
        for i, cost_name in enumerate(cost_names):
            if cost_name in traj['costs'] and len(traj['costs'][cost_name]) > 0:
                row, col = 2, i % 3
                if i == 3:  # Last plot goes to next available spot
                    row, col = 2, 2
                    
                axes[row, col].plot(time[:len(traj['costs'][cost_name])], 
                                  traj['costs'][cost_name], alpha=0.7)
                axes[row, col].set_title(f'{cost_name.title()} Cost')
                axes[row, col].set_ylabel('Cost')
                axes[row, col].set_xlabel('Time Step')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_results/trajectory_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_animated_arm_visualization(self, trajectory, output_file="arm_animation.mp4"):
        """Create animated visualization of arm movement."""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set up the plot
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Planar Arm Motion')
        
        # Initialize plot elements
        arm_line, = ax.plot([], [], 'b-o', linewidth=4, markersize=8, label='Arm')
        ee_traj_line, = ax.plot([], [], 'r--', alpha=0.5, label='EE trajectory')
        goal_marker = Circle((0, 0), 0.1, color='green', alpha=0.3, label='Goal region')
        ax.add_patch(goal_marker)
        
        # Goal position (assuming constant)
        goal_pos = trajectory['ee_goals'][0]
        goal_marker.set_center(goal_pos)
        ax.scatter(goal_pos[0], goal_pos[1], color='red', s=200, marker='x', 
                  zorder=10, label='Goal')
        
        ax.legend()
        
        # Animation data
        joint_angles = trajectory['joint_angles']
        ee_positions = trajectory['ee_positions']
        
        def animate(frame):
            if frame >= len(joint_angles):
                return arm_line, ee_traj_line
                
            # Current joint configuration
            q = joint_angles[frame]
            q1, q2 = q
            
            # Forward kinematics for visualization
            x0, y0 = 0, 0
            x1 = np.cos(q1)
            y1 = np.sin(q1) 
            x2 = x1 + np.cos(q1 + q2)
            y2 = y1 + np.sin(q1 + q2)
            
            # Update arm configuration
            arm_line.set_data([x0, x1, x2], [y0, y1, y2])
            
            # Update end-effector trajectory
            if frame > 0:
                ee_traj_line.set_data(ee_positions[:frame, 0], ee_positions[:frame, 1])
            
            return arm_line, ee_traj_line
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(joint_angles), 
            interval=50, blit=True, repeat=True
        )
        
        # Save animation
        os.makedirs("analysis_results", exist_ok=True)
        anim.save(f"analysis_results/{output_file}", writer='ffmpeg', fps=20)
        plt.close()
        
        print(f"Animation saved to analysis_results/{output_file}")
    
    def compare_learning_progress(self, model_paths, labels=None):
        """Compare learning progress across different models."""
        
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(model_paths))]
        
        results = {}
        
        for path, label in zip(model_paths, labels):
            if os.path.exists(path + ".zip"):
                model = SAC.load(path)
                
                # Quick evaluation
                env = self.env_class()
                returns = []
                
                for _ in range(20):
                    obs, _ = env.reset()
                    ep_return = 0
                    
                    for _ in range(400):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        ep_return += reward
                        
                        if terminated or truncated:
                            break
                    
                    returns.append(ep_return)
                
                env.close()
                results[label] = {
                    'returns': returns,
                    'mean': np.mean(returns),
                    'std': np.std(returns)
                }
        
        # Plot comparison
        if results:
            labels = list(results.keys())
            means = [results[l]['mean'] for l in labels]
            stds = [results[l]['std'] for l in labels]
            
            plt.figure(figsize=(10, 6))
            plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
            plt.title('Model Performance Comparison')
            plt.ylabel('Mean Episode Return')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('analysis_results/model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return results


def main():
    """Run comprehensive analysis."""
    
    print("Starting Comprehensive Performance Analysis")
    print("="*50)
    
    os.makedirs("analysis_results", exist_ok=True)
    
    # Analyze trained model (if exists)
    model_path = "models/sac_planar_arm"
    analyzer = PerformanceAnalyzer(model_path)
    
    if analyzer.model:
        print("Analyzing trained model trajectories...")
        trajectories = analyzer.analyze_trajectory(episodes=3)
        
        print("Generating trajectory plots...")
        analyzer.plot_trajectory_analysis(trajectories)
        
        print("Creating animated visualization...")
        analyzer.create_animated_arm_visualization(trajectories[0])
        
        print("Analysis complete! Check analysis_results/ directory.")
    else:
        print("No trained model found. Train first with train_sac.py")


if __name__ == "__main__":
    main()