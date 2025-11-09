import numpy as np
import matplotlib

# Use non-interactive backend for SSH/headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from envs.planar_arm_env import PlanarArmEnv


def forward_kinematics(q, link_lengths=(1.0, 1.0)):
    q1, q2 = q
    l1, l2 = link_lengths

    x0, y0 = 0.0, 0.0
    x1 = l1 * np.cos(q1)
    y1 = l1 * np.sin(q1)
    x2 = x1 + l2 * np.cos(q1 + q2)
    y2 = y1 + l2 * np.sin(q1 + q2)

    return (x0, y0), (x1, y1), (x2, y2)


def main():
    env = PlanarArmEnv()
    obs, info = env.reset()
    frames = []

    for t in range(200):  # record 200 frames
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        q = env.q
        (x0, y0), (x1, y1), (x2, y2) = forward_kinematics(q)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([x0, x1, x2], [y0, y1, y2], "-o", linewidth=3, markersize=8)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"Random policy - Step {t}")
        ax.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

        if terminated or truncated:
            break

    env.close()

    output_path = "arm_random.mp4"
    imageio.mimsave(output_path, frames, fps=15, codec="libx264")
    print(f"Video recorded at: {output_path}")


if __name__ == "__main__":
    main()
