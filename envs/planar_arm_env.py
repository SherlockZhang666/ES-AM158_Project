# envs/planar_arm_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PlanarArmEnv(gym.Env):
    """
    一个非常简化的 2-DOF 平面机械臂环境。
    现在先用 torque 作为 action，后面你可以再改成 ΔK 控制。
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, dt=0.02, max_torque=5.0):
        super().__init__()
        self.n_dof = 2
        self.dt = dt
        self.max_torque = max_torque
        self.render_mode = render_mode

        # 状态：[q1, q2, dq1, dq2, q1_goal, q2_goal]
        high = np.array(
            [np.pi, np.pi, 10.0, 10.0, np.pi, np.pi],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # 先让动作 = 各关节 torque，后面再改成 “ΔK 增量” 也可以
        self.action_space = spaces.Box(
            low=-max_torque,
            high=max_torque,
            shape=(self.n_dof,),
            dtype=np.float32
        )

        self.q = None
        self.dq = None
        self.q_goal = None

        self.step_count = 0
        self.max_steps = 400

        # 为了后面写 jerk 惩罚，记录上一时刻的 torque
        self.prev_u = np.zeros(self.n_dof, dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self.q, self.dq, self.q_goal]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 初始角度随机小偏移
        self.q = self.np_random.uniform(
            low=-0.5, high=0.5, size=(self.n_dof,)
        ).astype(np.float32)
        self.dq = np.zeros(self.n_dof, dtype=np.float32)

        # 随机目标角度
        self.q_goal = self.np_random.uniform(
            low=-1.0, high=1.0, size=(self.n_dof,)
        ).astype(np.float32)

        self.step_count = 0
        self.prev_u = np.zeros(self.n_dof, dtype=np.float32)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # 限幅后的 torque
        u = np.clip(action, -self.max_torque, self.max_torque)

        # 非严谨动力学：M=I，阻尼 0.1
        ddq = u - 0.1 * self.dq
        self.dq = self.dq + ddq * self.dt
        self.q = self.q + self.dq * self.dt

        self.step_count += 1

        obs = self._get_obs()

        # 误差
        e = self.q_goal - self.q

        # 简单的代价：位置误差 + 力矩大小 + 力矩变化
        we, wu, wdu = 1.0, 0.01, 0.001
        pos_cost = we * np.sum(e ** 2)
        torque_cost = wu * np.sum(u ** 2)
        jerk_cost = wdu * np.sum((u - self.prev_u) ** 2)

        cost = pos_cost + torque_cost + jerk_cost
        reward = -cost

        self.prev_u = u.copy()

        # 终止条件
        terminated = False   # 这里没有“任务完成就结束”，可以自己加
        truncated = self.step_count >= self.max_steps

        info = {
            "pos_cost": pos_cost,
            "torque_cost": torque_cost,
            "jerk_cost": jerk_cost,
        }

        # SSH 环境下不要尝试弹窗渲染
        # if self.render_mode == "human":
        #     self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        # 你如果以后要本地跑 GUI，可以在这里画图
        pass

    def close(self):
        pass


if __name__ == "__main__":
    # 小测试：随机动作跑一跑，确保 env 不报错
    env = PlanarArmEnv()
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
