import numpy as np


class PDBaseline:
    """
    Basic diagonal PD controller:
        u = Kp * (q_goal - q) + Kd * (-dq)

    - Kp and Kd can be scalars, lists, or numpy arrays of length = n_dof.
    - This class is independent of Gymnasium, purely numerical.
    """

    def __init__(self, Kp, Kd, max_torque=5.0):
        self.Kp = np.array(Kp, dtype=np.float32)
        self.Kd = np.array(Kd, dtype=np.float32)
        if self.Kp.shape != self.Kd.shape:
            raise ValueError(f"Kp and Kd shape mismatch: {self.Kp.shape} vs {self.Kd.shape}")
        self.n_dof = self.Kp.shape[0]
        self.max_torque = float(max_torque)

    def compute_torque(self, q, dq, q_goal):
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        dq = np.asarray(dq, dtype=np.float32).reshape(-1)
        q_goal = np.asarray(q_goal, dtype=np.float32).reshape(-1)

        if q.shape[0] != self.n_dof:
            raise ValueError(f"q dim {q.shape[0]} != expected {self.n_dof}")
        if dq.shape[0] != self.n_dof or q_goal.shape[0] != self.n_dof:
            raise ValueError("dq or q_goal dimension mismatch")

        e = q_goal - q
        edot = -dq

        u = self.Kp * e + self.Kd * edot
        u = np.clip(u, -self.max_torque, self.max_torque)
        return u

    def set_gains(self, Kp=None, Kd=None):
        if Kp is not None:
            Kp = np.array(Kp, dtype=np.float32)
            if Kp.shape != self.Kp.shape:
                raise ValueError("New Kp shape mismatch")
            self.Kp = Kp
        if Kd is not None:
            Kd = np.array(Kd, dtype=np.float32)
            if Kd.shape != self.Kd.shape:
                raise ValueError("New Kd shape mismatch")
            self.Kd = Kd

    def __repr__(self):
        return f"PDBaseline(Kp={self.Kp}, Kd={self.Kd}, max_torque={self.max_torque})"
