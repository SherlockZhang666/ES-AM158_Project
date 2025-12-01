# eval_pd_zero_dK.py
import numpy as np
from envs.planar_arm_env import PlanarArmEnv

def eval_pd_like(episodes=10):
    env = PlanarArmEnv()
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for ep in range(episodes):
        obs, info = env.reset()
        ep_ret = 0.0
        for t in range(400):  # same horizon
            obs, r, term, trunc, inf = env.step(zero_action)
            ep_ret += float(r)
            if term or trunc:
                break
        print(f"[PD-like] ep={ep} return={ep_ret:.2f}")
    env.close()

if __name__ == "__main__":
    eval_pd_like()
