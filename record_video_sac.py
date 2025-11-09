import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from stable_baselines3 import SAC

from envs.planar_arm_env import PlanarArmEnv
from record_video_random import forward_kinematics


def main():
    env = PlanarArmEnv()
    model_path = os.path.join("models", "sac_planar_arm")
    model = SAC.load(model_path)
    print(f"Model loaded from: {model_path}")

    frames = []

    obs, info = env.reset()
    for t in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        q = env.q
        (x0, y0), (x1, y1), (x2, y2) = forward_kinematics(q)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([x0, x1, x2], [y0, y1, y2], "-o", linewidth=3, markersize=8)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"SAC policy - Step {t}, reward={reward:.2f}")
        ax.axis("off")

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

        if terminated or truncated:
            break

    env.close()

    output_path = "arm_sac.mp4"
    imageio.mimsave(output_path, frames, fps=15, codec="libx264")
    print(f"Video record at: {output_path}")


if __name__ == "__main__":
    main()
