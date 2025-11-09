# control/gain_scheduler.py

import numpy as np


class GainScheduler:
    """
    Adaptive PD gain scheduler based on a baseline Kp0, Kd0.

    RL outputs an increment action:
        action[0:n]   -> ΔKp_increment
        action[n:2n]  -> ΔKd_increment

    Internally we keep ΔKp_t, ΔKd_t and update:
        ΔK_t <- Proj(ΔK_{t-1} + action)

    Projection Proj enforces:
        - magnitude: |ΔK_t| <= eps
        - slew rate: |ΔK_t - ΔK_{t-1}| <= rho
    """

    def __init__(self, Kp0, Kd0, eps=2.0, rho=0.2, max_torque=5.0):
        self.Kp0 = np.array(Kp0, dtype=np.float32)
        self.Kd0 = np.array(Kd0, dtype=np.float32)
        if self.Kp0.shape != self.Kd0.shape:
            raise ValueError(f"Kp0 and Kd0 shape mismatch: {self.Kp0.shape} vs {self.Kd0.shape}")

        self.n_dof = self.Kp0.shape[0]
        self.eps = float(eps)
        self.rho = float(rho)
        self.max_torque = float(max_torque)

        # Current gain offsets ΔK_t
        self.delta_Kp = np.zeros_like(self.Kp0)
        self.delta_Kd = np.zeros_like(self.Kd0)

        # For computing ||ΔK_t - ΔK_{t-1}||^2 in the reward
        self.prev_delta_Kp = np.zeros_like(self.Kp0)
        self.prev_delta_Kd = np.zeros_like(self.Kd0)

    def reset(self):
        """
        Reset ΔK_t and its previous value (called at env.reset()).
        """
        self.delta_Kp[...] = 0.0
        self.delta_Kd[...] = 0.0
        self.prev_delta_Kp[...] = 0.0
        self.prev_delta_Kd[...] = 0.0

    def _project_component(self, new_delta: np.ndarray, old_delta: np.ndarray) -> np.ndarray:
        """
        Apply both magnitude and slew constraints to a ΔK component.
        """
        # 1) global magnitude bound
        new_delta = np.clip(new_delta, -self.eps, self.eps)

        # 2) slew-rate bound per step
        diff = new_delta - old_delta
        diff = np.clip(diff, -self.rho, self.rho)
        projected = old_delta + diff
        return projected

    def update_from_action(self, action):
        """
        Update ΔK given RL increment action.
        action must have length 2 * n_dof.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2 * self.n_dof:
            raise ValueError(f"Expected action length {2 * self.n_dof}, got {action.shape[0]}")

        inc_Kp = action[: self.n_dof]
        inc_Kd = action[self.n_dof : 2 * self.n_dof]

        # Save previous ΔK for reward
        self.prev_delta_Kp = self.delta_Kp.copy()
        self.prev_delta_Kd = self.delta_Kd.copy()

        raw_new_delta_Kp = self.delta_Kp + inc_Kp
        raw_new_delta_Kd = self.delta_Kd + inc_Kd

        self.delta_Kp = self._project_component(raw_new_delta_Kp, self.delta_Kp)
        self.delta_Kd = self._project_component(raw_new_delta_Kd, self.delta_Kd)

    def get_effective_gains(self):
        """
        Effective gains:
            Kp_eff = Kp0 + delta_Kp
            Kd_eff = Kd0 + delta_Kd
        with positivity constraint.
        """
        Kp_eff = self.Kp0 + self.delta_Kp
        Kd_eff = self.Kd0 + self.delta_Kd

        # Enforce structure Kp >= 0, Kd >= 0
        Kp_eff = np.maximum(Kp_eff, 0.0)
        Kd_eff = np.maximum(Kd_eff, 0.0)
        return Kp_eff, Kd_eff

    def compute_torque(self, q, dq, q_goal):
        """
        PD torque with gain-scheduled effective gains.
        """
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        dq = np.asarray(dq, dtype=np.float32).reshape(-1)
        q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)

        if q.shape[0] != self.n_dof:
            raise ValueError(f"q dim {q.shape[0]} != expected {self.n_dof}")

        Kp_eff, Kd_eff = self.get_effective_gains()

        e = q_goal - q
        edot = -dq

        u = Kp_eff * e + Kd_eff * edot
        u = np.clip(u, -self.max_torque, self.max_torque)
        return u

    def gain_diff_norm_sq(self):
        """
        ||ΔK_t - ΔK_{t-1}||^2, used in the reward.
        """
        dKp = self.delta_Kp - self.prev_delta_Kp
        dKd = self.delta_Kd - self.prev_delta_Kd
        return float(np.sum(dKp ** 2) + np.sum(dKd ** 2))

    def __repr__(self):
        return (
            f"GainScheduler(Kp0={self.Kp0}, Kd0={self.Kd0}, "
            f"eps={self.eps}, rho={self.rho}, max_torque={self.max_torque})"
        )
