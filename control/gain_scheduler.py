import numpy as np


class GainScheduler:
    """
    Adaptive PD gain scheduler that builds on a baseline PD controller.

    The RL agent outputs an action representing the *increment* of PD gains:
        action[0:n]   -> ΔKp increment
        action[n:2n]  -> ΔKd increment

    The scheduler maintains internal gain offsets:
        delta_Kp, delta_Kd

    Each step:
        new_delta = old_delta + increment
        Then applies:
            1) magnitude constraint: |new_delta| <= eps
            2) rate constraint: |new_delta - old_delta| <= rho
    """

    def __init__(
        self,
        Kp0,
        Kd0,
        eps=2.0,
        rho=0.2,
        max_torque=5.0,
    ):
        """
        Args:
            Kp0, Kd0: baseline PD gains (scalars or length-n arrays)
            eps: absolute bound on ΔK magnitude
            rho: maximum per-step change in ΔK
            max_torque: torque saturation limit
        """
        self.Kp0 = np.array(Kp0, dtype=np.float32)
        self.Kd0 = np.array(Kd0, dtype=np.float32)
        if self.Kp0.shape != self.Kd0.shape:
            raise ValueError(f"Kp0 and Kd0 shape mismatch: {self.Kp0.shape} vs {self.Kd0.shape}")

        self.n_dof = self.Kp0.shape[0]
        self.eps = float(eps)
        self.rho = float(rho)
        self.max_torque = float(max_torque)

        # Current gain offsets ΔK
        self.delta_Kp = np.zeros_like(self.Kp0)
        self.delta_Kd = np.zeros_like(self.Kd0)

    def reset(self):
        """
        Reset ΔK to zero (e.g., called at env.reset()).
        """
        self.delta_Kp[...] = 0.0
        self.delta_Kd[...] = 0.0

    def _project_component(self, new_delta: np.ndarray, old_delta: np.ndarray) -> np.ndarray:
        """
        Project a ΔK component to satisfy:
            1) |new_delta| <= eps
            2) |new_delta - old_delta| <= rho
        """
        # Magnitude constraint
        new_delta = np.clip(new_delta, -self.eps, self.eps)

        # Rate constraint
        diff = new_delta - old_delta
        diff = np.clip(diff, -self.rho, self.rho)
        projected = old_delta + diff
        return projected

    def update_from_action(self, action):
        """
        Update ΔKp and ΔKd given RL agent's action increments.

        Args:
            action: numpy array of length 2 * n_dof:
                [ΔKp_increment, ΔKd_increment]
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2 * self.n_dof:
            raise ValueError(
                f"Expected action length {2 * self.n_dof}, got {action.shape[0]}"
            )

        inc_Kp = action[: self.n_dof]
        inc_Kd = action[self.n_dof : 2 * self.n_dof]

        raw_new_delta_Kp = self.delta_Kp + inc_Kp
        raw_new_delta_Kd = self.delta_Kd + inc_Kd

        # Apply projection constraints
        self.delta_Kp = self._project_component(raw_new_delta_Kp, self.delta_Kp)
        self.delta_Kd = self._project_component(raw_new_delta_Kd, self.delta_Kd)

    def get_effective_gains(self):
        """
        Compute effective PD gains:
            Kp_eff = Kp0 + ΔKp
            Kd_eff = Kd0 + ΔKd
        Ensure gains remain non-negative.
        """
        Kp_eff = self.Kp0 + self.delta_Kp
        Kd_eff = self.Kd0 + self.delta_Kd

        # Prevent negative gains for stability
        Kp_eff = np.maximum(Kp_eff, 0.0)
        Kd_eff = np.maximum(Kd_eff, 0.0)
        return Kp_eff, Kd_eff

    def compute_torque(self, q, dq, q_goal):
        """
        Compute PD torque using the current adaptive gains.
        Interface is the same as PDBaseline.
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

    def gains_norm(self):
        """
        Return the squared norm of ΔK for use as a regularization penalty:
            ||ΔKp||^2 + ||ΔKd||^2
        """
        return float(np.sum(self.delta_Kp ** 2) + np.sum(self.delta_Kd ** 2))

    def __repr__(self):
        return (
            f"GainScheduler(Kp0={self.Kp0}, Kd0={self.Kd0}, "
            f"eps={self.eps}, rho={self.rho}, max_torque={self.max_torque})"
        )


if __name__ == "__main__":
    # Simple self-test
    gs = GainScheduler(Kp0=[10.0, 10.0], Kd0=[1.0, 1.0], eps=1.0, rho=0.2, max_torque=5.0)
    print(gs)

    q = [0.0, 0.0]
    dq = [0.0, 0.0]
    q_goal = [1.0, 1.0]

    u0 = gs.compute_torque(q, dq, q_goal)
    print("u0 =", u0)

    # Simulate RL updating the gains
    action = np.array([0.5, -0.3, 0.4, -0.2], dtype=np.float32)  # 2 DOF -> 4D action
    gs.update_from_action(action)
    u1 = gs.compute_torque(q, dq, q_goal)
    print("u1 =", u1)
    print("delta_Kp =", gs.delta_Kp, "delta_Kd =", gs.delta_Kd)
