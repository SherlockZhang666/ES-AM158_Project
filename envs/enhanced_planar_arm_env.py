"""
Enhanced planar arm environment with more realistic dynamics and configuration options.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from control.gain_scheduler import GainScheduler


class EnhancedPlanarArmEnv(gym.Env):
    """
    Enhanced 2-DOF planar arm with:
    - More realistic dynamics (mass, inertia, gravity, friction)
    - Configurable scheduler parameters
    - Success condition tracking
    - Multiple task types (reaching, tracking)
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        dt=0.02,
        max_torque=5.0,
        # Physical parameters
        link_lengths=(1.0, 1.0),
        link_masses=(1.0, 1.0), 
        link_inertias=(0.1, 0.1),
        gravity=9.81,
        friction_coeffs=(0.1, 0.1),
        # Scheduler parameters
        Kp0=(10.0, 10.0),
        Kd0=(1.0, 1.0),
        eps=2.0,
        rho=0.2,
        # Reward weights
        w_e=1.0,
        w_u=0.01, 
        w_dK=0.001,
        w_j=0.001,
        # Task configuration
        task_type="reaching",  # "reaching" or "tracking"
        success_threshold=0.1,
        initial_pos_range=0.5,
        goal_range=1.0,
    ):
        super().__init__()
        self.n_dof = 2
        self.dt = dt
        self.max_torque = max_torque
        self.render_mode = render_mode
        
        # Physical parameters
        self.link_lengths = np.array(link_lengths, dtype=np.float32)
        self.link_masses = np.array(link_masses, dtype=np.float32)
        self.link_inertias = np.array(link_inertias, dtype=np.float32)
        self.gravity = float(gravity)
        self.friction_coeffs = np.array(friction_coeffs, dtype=np.float32)
        
        # Task parameters
        self.task_type = task_type
        self.success_threshold = success_threshold
        self.initial_pos_range = initial_pos_range
        self.goal_range = goal_range
        
        # State space (add end-effector position for richer observation)
        # [q1, q2, dq1, dq2, q1_goal, q2_goal, ee_x, ee_y, ee_x_goal, ee_y_goal]
        high = np.array([
            np.pi, np.pi,           # Joint angles
            10.0, 10.0,             # Joint velocities  
            np.pi, np.pi,           # Goal joint angles
            3.0, 3.0,               # End-effector position
            3.0, 3.0,               # Goal end-effector position
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32,
        )

        # Action space remains the same
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_dof * 2,), dtype=np.float32,
        )

        # Enhanced gain scheduler with configurable parameters
        self.scheduler = GainScheduler(
            Kp0=Kp0, Kd0=Kd0, eps=eps, rho=rho, max_torque=max_torque,
        )

        # State variables
        self.q = None
        self.dq = None 
        self.q_goal = None
        self.ee_pos_goal = None

        self.step_count = 0
        self.max_steps = 400
        self.prev_u = np.zeros(self.n_dof, dtype=np.float32)

        # Reward weights
        self.w_e = w_e
        self.w_u = w_u
        self.w_dK = w_dK
        self.w_j = w_j
        
        # Performance tracking
        self.episode_success = False
        self.min_distance_to_goal = np.inf

    def _compute_forward_kinematics(self, q):
        """Compute end-effector position from joint angles."""
        q1, q2 = q
        l1, l2 = self.link_lengths
        
        ee_x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        ee_y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        
        return np.array([ee_x, ee_y], dtype=np.float32)

    def _compute_dynamics(self, q, dq, u):
        """
        More realistic 2-DOF arm dynamics with mass matrix, Coriolis, gravity.
        Simplified but more accurate than the previous linear model.
        """
        q1, q2 = q
        dq1, dq2 = dq
        m1, m2 = self.link_masses
        l1, l2 = self.link_lengths
        I1, I2 = self.link_inertias
        g = self.gravity
        
        # Mass matrix M(q)
        M11 = I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * l2/2 * np.cos(q2)
        M12 = I2 + m2 * l1 * l2/2 * np.cos(q2)
        M21 = M12  
        M22 = I2
        M = np.array([[M11, M12], [M21, M22]], dtype=np.float32)
        
        # Coriolis matrix C(q,dq)
        h = -m2 * l1 * l2/2 * np.sin(q2)
        C = np.array([[h * dq2, h * (dq1 + dq2)], 
                      [-h * dq1, 0]], dtype=np.float32)
        
        # Gravity vector G(q)
        G1 = (m1 * l1/2 + m2 * l1) * g * np.cos(q1) + m2 * l2/2 * g * np.cos(q1 + q2)
        G2 = m2 * l2/2 * g * np.cos(q1 + q2)
        G = np.array([G1, G2], dtype=np.float32)
        
        # Friction
        F = self.friction_coeffs * dq
        
        # Solve for acceleration: M(q) * ddq = u - C(q,dq)*dq - G(q) - F
        try:
            ddq = np.linalg.solve(M, u - C @ dq - G - F)
        except np.linalg.LinAlgError:
            # Fallback to simpler dynamics if matrix is singular
            ddq = (u - 0.5 * dq - 0.1 * np.array([np.cos(q1), np.cos(q1+q2)])) / 2.0
            
        return ddq

    def _get_obs(self):
        """Enhanced observation including end-effector positions."""
        ee_pos = self._compute_forward_kinematics(self.q)
        
        return np.concatenate([
            self.q,                    # Joint angles
            self.dq,                   # Joint velocities
            self.q_goal,               # Goal joint angles  
            ee_pos,                    # End-effector position
            self.ee_pos_goal,          # Goal end-effector position
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Random initial configuration
        self.q = self.np_random.uniform(
            low=-self.initial_pos_range, 
            high=self.initial_pos_range, 
            size=(self.n_dof,)
        ).astype(np.float32)
        
        self.dq = np.zeros(self.n_dof, dtype=np.float32)
        
        # Random goal configuration  
        self.q_goal = self.np_random.uniform(
            low=-self.goal_range, 
            high=self.goal_range, 
            size=(self.n_dof,)
        ).astype(np.float32)
        
        self.ee_pos_goal = self._compute_forward_kinematics(self.q_goal)

        self.step_count = 0
        self.prev_u = np.zeros(self.n_dof, dtype=np.float32)
        self.scheduler.reset()
        
        # Reset tracking variables
        self.episode_success = False
        self.min_distance_to_goal = np.inf

        obs = self._get_obs()
        info = {
            'initial_ee_pos': self._compute_forward_kinematics(self.q).copy(),
            'goal_ee_pos': self.ee_pos_goal.copy(),
        }
        
        return obs, info

    def step(self, action):
        # Update gain scheduling
        action = np.asarray(action, dtype=np.float32)
        scaled_action = 0.1 * action
        self.scheduler.update_from_action(scaled_action)

        # Compute control torque
        u = self.scheduler.compute_torque(self.q, self.dq, self.q_goal)

        # Enhanced dynamics
        ddq = self._compute_dynamics(self.q, self.dq, u)
        
        # Euler integration
        self.dq = self.dq + ddq * self.dt
        self.q = self.q + self.dq * self.dt

        self.step_count += 1
        obs = self._get_obs()

        # Enhanced reward computation
        ee_pos = self._compute_forward_kinematics(self.q)
        
        # Position error in both joint and task space
        joint_error = self.q_goal - self.q
        ee_error = self.ee_pos_goal - ee_pos
        
        # Primary cost: task-space error (more meaningful for manipulation)
        pos_cost = self.w_e * (0.5 * np.sum(joint_error**2) + np.sum(ee_error**2))
        
        # Control effort cost
        torque_cost = self.w_u * np.sum(u**2)
        
        # Smoothness costs
        jerk_cost = self.w_j * np.sum((u - self.prev_u)**2)
        gain_smooth_cost = self.w_dK * self.scheduler.gain_diff_norm_sq()
        
        self.prev_u = u.copy()
        
        total_cost = pos_cost + torque_cost + jerk_cost + gain_smooth_cost
        reward = -total_cost
        
        # Success tracking
        ee_distance = np.linalg.norm(ee_error)
        self.min_distance_to_goal = min(self.min_distance_to_goal, ee_distance)
        
        if ee_distance < self.success_threshold:
            self.episode_success = True
            reward += 10.0  # Success bonus
        
        # Termination conditions
        terminated = self.episode_success
        truncated = self.step_count >= self.max_steps

        info = {
            'pos_cost': pos_cost,
            'torque_cost': torque_cost, 
            'jerk_cost': jerk_cost,
            'gain_smooth_cost': gain_smooth_cost,
            'ee_error': ee_distance,
            'joint_error': np.linalg.norm(joint_error),
            'episode_success': self.episode_success,
            'min_distance_to_goal': self.min_distance_to_goal,
            'effective_gains': self.scheduler.get_effective_gains(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass