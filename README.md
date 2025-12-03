# ES-AM158 Project: RL-Tuned Gain Scheduling for Robotic Arm Control

**Course**: ES/AM 158 - Reinforcement Learning  
**Project**: Adaptive PD Gain Scheduling using Deep Reinforcement Learning  
**Task**: 2-DOF Planar Arm Reaching Control  

## Overview

This project implements an adaptive PD controller with reinforcement learning-based gain scheduling for a 2-DOF planar robotic arm. Instead of learning control torques directly, the RL agent learns to adaptively adjust PD gains based on the current state and task requirements.

### Key Features
- **Novel Approach**: RL learns gain increments (ΔK) rather than raw torques
- **Constrained Learning**: Magnitude and slew-rate limits on gain variations
- **Multi-Objective Reward**: Position tracking, control effort, smoothness, gain variation
- **Comprehensive Evaluation**: Multiple baselines, algorithms, and analysis tools

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n rlpj python==3.10 -y
conda activate rlpj

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install numpy "gymnasium[classic-control]" stable-baselines3 matplotlib imageio-ffmpeg
```

### 2. Basic Training & Evaluation

```bash
# Train the RL agent (basic)
python train_sac.py

# Evaluate trained model
python eval_sac.py

# Quick baseline comparison
python eval_pd_zero_dK.py
```

### 3. Run All Experiments

```bash
# Comprehensive experiment suite
./run_experiments.sh
```

This will run all experiments, comparisons, and generate a comprehensive report.

## Project Structure

```
.
├── envs/
│   ├── planar_arm_env.py           # Basic environment implementation
│   ├── enhanced_planar_arm_env.py  # Enhanced environment with realistic dynamics
│   └── __init__.py
├── control/
│   ├── gain_scheduler.py           # RL-tuned gain scheduler
│   └── pd_baseline.py              # Fixed PD controller baseline
├── experiments/
│   ├── baseline_comparison.py      # Compare RL vs fixed PD controllers
│   ├── algorithm_comparison.py     # Compare SAC vs PPO vs TD3
│   ├── hyperparameter_tuning.py    # Ablation studies
│   └── performance_analysis.py     # Detailed trajectory analysis
├── train_sac.py                    # Enhanced training script
├── eval_sac.py                     # Model evaluation
├── run_experiments.sh              # Automated experiment runner
└── README.md                       # This file
```

## Methodology

### Problem Formulation

**State Space**: `[q₁, q₂, q̇₁, q̇₂, q₁_goal, q₂_goal]`  
- Joint angles, velocities, and goal positions

**Action Space**: `[ΔKp₁, ΔKp₂, ΔKd₁, ΔKd₂]`  
- Gain increment commands (not raw torques)

**Control Law**: 
```
Kp_eff = Kp₀ + ΔKp_projected
Kd_eff = Kd₀ + ΔKd_projected  
τ = Kp_eff ⊙ (q_goal - q) + Kd_eff ⊙ (-q̇)
```

### Constraint Enforcement

1. **Magnitude Constraint**: `|ΔK| ≤ ε`
2. **Slew Rate Constraint**: `|ΔK_t - ΔK_{t-1}| ≤ ρ`  
3. **Positivity**: `Kp_eff ≥ 0, Kd_eff ≥ 0`

### Reward Function

```
R = -(w_e ||e||² + w_u ||u||² + w_ΔK ||ΔK_t - ΔK_{t-1}||² + w_j ||u_t - u_{t-1}||²)
```

Where:
- `e`: position error
- `u`: control torque
- `w_e, w_u, w_ΔK, w_j`: tunable weights

## Experiments

### 1. Baseline Comparison

**Purpose**: Establish performance baselines

**Methods Compared**:
- Fixed PD (Conservative): `Kp=[8,8], Kd=[2,2]`
- Fixed PD (Aggressive): `Kp=[15,15], Kd=[3,3]`  
- RL Gain Scheduler
- Random Policy

**Metrics**: Mean return, success rate, final positioning error

### 2. Algorithm Comparison

**Purpose**: Compare RL algorithms for this task

**Algorithms**: SAC, PPO, TD3

**Key Questions**:
- Which algorithm learns most efficiently?
- How do they compare in final performance?
- What are the trade-offs?

### 3. Hyperparameter Analysis

**Purpose**: Understand impact of design choices

**Studies**:
- Reward weight ablation (`w_e, w_u, w_ΔK, w_j`)
- Constraint parameter sensitivity (`ε, ρ`)
- Baseline gain selection (`Kp₀, Kd₀`)

### 4. Performance Analysis

**Purpose**: Understand learned behavior

**Analysis**:
- Trajectory visualization
- Gain evolution over time
- Phase-based gain adaptation
- Control effort vs accuracy trade-offs

## Key Results (Expected)

1. **RL vs Fixed PD**: RL should show better adaptation to different scenarios
2. **Algorithm Comparison**: SAC likely performs best for continuous control
3. **Reward Weights**: Balanced weights achieve best trade-off
4. **Gain Evolution**: Adaptive gains should show task-phase dependent behavior

## Advanced Usage

### Custom Training

```bash
# Custom hyperparameters
python train_sac.py --timesteps 500000 --lr 1e-4 --w_e 2.0 --w_u 0.05

# Custom experiment name
python train_sac.py --experiment_name "high_precision_control"
```

### Individual Experiments

```bash
# Run specific experiment
python experiments/baseline_comparison.py
python experiments/algorithm_comparison.py
python experiments/hyperparameter_tuning.py
python experiments/performance_analysis.py
```

### Analysis Tools

```bash
# Tensorboard monitoring
tensorboard --logdir experiments/

# Detailed trajectory analysis
python experiments/performance_analysis.py --model_path models/sac_planar_arm
```

## Implementation Details

### Environment Features
- Gymnasium-compatible interface
- Configurable reward weights
- Enhanced dynamics with mass/inertia (optional)
- Success condition tracking
- Comprehensive info logging

### Gain Scheduler Features
- Projection-based constraint enforcement
- Smooth gain transitions
- Configurable limits and baseline gains
- Real-time constraint monitoring

### Training Features
- Automatic checkpointing
- Evaluation callbacks
- Tensorboard logging
- Hyperparameter tracking
- Experiment organization

## Troubleshooting

**Training Issues**:
- Ensure CUDA is available if using GPU
- Check memory usage for large buffer sizes
- Verify environment parameters are reasonable

**Performance Issues**:
- Try different reward weight combinations
- Adjust constraint parameters (ε, ρ)
- Increase training time for complex scenarios

**Visualization Issues**:
- Ensure `imageio-ffmpeg` is installed for video generation
- Check matplotlib backend for headless servers

## Future Directions

1. **Extended Scenarios**: Obstacles, moving targets, tracking tasks
2. **Real Robot Validation**: Transfer learning to physical systems  
3. **Comparative Studies**: Classical adaptive control methods
4. **Multi-DOF Extension**: Higher dimensional manipulators
5. **Robustness Analysis**: Noise, disturbances, model uncertainties

## References

- Stable Baselines3 Documentation
- OpenAI Gymnasium Environment Design
- Classical PD Control Theory
- Reinforcement Learning for Robotics

---

**Note**: This project is designed for educational purposes as part of ES/AM 158. The implementation focuses on demonstrating key RL concepts in robotics control rather than achieving state-of-the-art performance.