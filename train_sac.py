import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.planar_arm_env import PlanarArmEnv


def make_env():
    """
    DummyVecEnv need a function that returns an env instance.
    """
    env = PlanarArmEnv()
    env = Monitor(env)
    return env


def main():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # only one env for now
    vec_env = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # train (try)
    model.learn(total_timesteps=100_000)
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "sac_planar_arm")
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
