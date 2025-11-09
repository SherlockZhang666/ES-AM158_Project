# envs/planar_arm_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from control.gain_scheduler import GainScheduler


class PlanarArmEnv(gym.Env):
    """
    2-DOF planar arm with RL-tuned gain scheduling.
    Action = gain increment in ΔK-space, not raw torque.

    State: [q, dq, q_goal]
    Action: [ΔKp_increment (n), ΔKd_increment (n)]  -> length = 2 * n_dof
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        dt=0.02,
        max_torque=5.0,
        # reward weights
        w_e=1.0,
        w_u=0.01,
        w_dK=0.001,
        w_j=0.001,
    ):
        super().__init__()
        self.n_dof = 2
        self.dt = dt
        self.max_torque = max_torque
        self.render_mode = render_mode

        # State: [q1, q2, dq1, dq2, q1_goal, q2_goal]
        high = np.array(
            [np.pi, np.pi, 10.0, 10.0, np.pi, np.pi],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32,
        )

        # Action: gain increments in ΔK-space (ΔKp_inc + ΔKd_inc)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_dof * 2,),
            dtype=np.float32,
        )

        # Baseline gains K0 (you can tune these)
        self.scheduler = GainScheduler(
            Kp0=[10.0, 10.0],
            Kd0=[1.0, 1.0],
            eps=2.0,
            rho=0.2,
            max_torque=max_torque,
        )

        self.q = None
        self.dq = None
        self.q_goal = None

        self.step_count = 0
        self.max_steps = 400

        # To compute ||u_t - u_{t-1}||^2
        self.prev_u = np.zeros(self.n_dof, dtype=np.float32)

        # Reward weights
        self.w_e = w_e
        self.w_u = w_u
        self.w_dK = w_dK
        self.w_j = w_j

    def _get_obs(self):
        return np.concatenate([self.q, self.dq, self.q_goal]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.q = self.np_random.uniform(
            low=-0.5, high=0.5, size=(self.n_dof,)
        ).astype(np.float32)
        self.dq = np.zeros(self.n_dof, dtype=np.float32)
        self.q_goal = self.np_random.uniform(
            low=-1.0, high=1.0, size=(self.n_dof,)
        ).astype(np.float32)

        self.step_count = 0
        self.prev_u = np.zeros(self.n_dof, dtype=np.float32)
        self.scheduler.reset()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # 1) Update ΔK from RL action (scaled)
        action = np.asarray(action, dtype=np.float32)
        scaled_action = 0.1 * action  # small increments
        self.scheduler.update_from_action(scaled_action)

        # 2) Compute PD torque with scheduled gains
        u = self.scheduler.compute_torque(self.q, self.dq, self.q_goal)

        # 3) Simple dynamics: q̈ = u - 0.1 q̇
        ddq = u - 0.1 * self.dq
        self.dq = self.dq + ddq * self.dt
        self.q = self.q + self.dq * self.dt

        self.step_count += 1

        obs = self._get_obs()

        # 4) Reward terms as in proposal:
        #   we * ||e||^2 + wu * ||u||^2 + w_dK * ||ΔK_t - ΔK_{t-1}||^2 + w_j * ||u_t - u_{t-1}||^2
        e = self.q_goal - self.q
        pos_cost = self.w_e * np.sum(e ** 2)
        torque_cost = self.w_u * np.sum(u ** 2)

        jerk_cost = self.w_j * np.sum((u - self.prev_u) ** 2)
        self.prev_u = u.copy()

        dK_cost = self.w_dK * self.scheduler.gain_diff_norm_sq()

        cost = pos_cost + torque_cost + dK_cost + jerk_cost
        reward = -cost

        terminated = False  # You can add success condition (epsilon-ball) later
        truncated = self.step_count >= self.max_steps

        info = {
            "pos_cost": pos_cost,
            "torque_cost": torque_cost,
            "jerk_cost": jerk_cost,
            "dK_cost": dK_cost,
        }

        # We do NOT render in SSH/headless training; evaluation will handle video.
        return obs, reward, terminated, truncated, info

    def render(self):
        # For local GUI runs; leave empty for SSH usage
        pass

    def close(self):
        pass
