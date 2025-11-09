import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from stable_baselines3 import SAC
from envs.planar_arm_env import PlanarArmEnv


def forward_kinematics(q, link_lengths=(1.0, 1.0)):
    """
    Simple 2-link planar arm forward kinematics for visualization only.
    """
    q1, q2 = q
    l1, l2 = link_lengths

    x0, y0 = 0.0, 0.0
    x1 = l1 * np.cos(q1)
    y1 = l1 * np.sin(q1)
    x2 = x1 + l2 * np.cos(q1 + q2)
    y2 = y1 + l2 * np.sin(q1 + q2)

    return (x0, y0), (x1, y1), (x2, y2)


def evaluate_and_record(
    model_path,
    out_video="eval_arm.mp4",
    episodes=3,
    max_steps=200,
):
    env = PlanarArmEnv()
    model = SAC.load(model_path)
    print(f"Loaded model from: {model_path}")

    frames = []

    for ep in range(episodes):
        obs, info = env.reset()
        ep_return = 0.0

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward

            # Draw current arm configuration
            q = env.q
            (x0, y0), (x1, y1), (x2, y2) = forward_kinematics(q)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot([x0, x1, x2], [y0, y1, y2], "-o", linewidth=3, markersize=8)
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2.2, 2.2)
            ax.set_aspect("equal")
            ax.set_title(f"Ep {ep}, Step {t}, R={ep_return:.2f}")
            ax.axis("off")

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(buf)
            plt.close(fig)

            if terminated or truncated:
                break

        print(f"Episode {ep} return: {ep_return:.2f}")

    env.close()

    imageio.mimsave(out_video, frames, fps=15, codec="libx264")
    print(f"Evaluation video saved to: {out_video}")


if __name__ == "__main__":
    model_path = os.path.join("models", "sac_planar_arm")
    evaluate_and_record(model_path, out_video="eval_arm.mp4", episodes=3, max_steps=200)
