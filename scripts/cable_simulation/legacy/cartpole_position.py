# SPDX-License-Identifier: BSD-3-Clause
"""Starter script: run cartpole and track the cart's slider position.

Usage:
    ./isaaclab.sh -p scripts/my_experiments/cartpole_position.py --num_envs 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Track cartpole cart position.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg


def main():
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Cartpole observation layout (per env):
    #   [0] cart position (slider_to_cart joint pos)
    #   [1] pole angle    (cart_to_pole joint pos)
    #   [2] cart velocity
    #   [3] pole angular velocity
    cart_pos_min = torch.full((args_cli.num_envs,), float("inf"), device=args_cli.device)
    cart_pos_max = torch.full((args_cli.num_envs,), float("-inf"), device=args_cli.device)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count = 0
                env.reset()
                cart_pos_min.fill_(float("inf"))
                cart_pos_max.fill_(float("-inf"))
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # push the cart with a position-proportional force (toy controller: bring cart back to x=0)
            obs_tensor = env.observation_manager.compute()["policy"]
            cart_pos = obs_tensor[:, 0]
            action = (-2.0 * cart_pos).unsqueeze(-1)  # 1-D action: effort on the slider

            obs, rew, terminated, truncated, info = env.step(action)

            cart_pos = obs["policy"][:, 0]
            cart_pos_min = torch.minimum(cart_pos_min, cart_pos)
            cart_pos_max = torch.maximum(cart_pos_max, cart_pos)

            if count % 30 == 0:
                print(
                    f"[step {count:3d}] env0 cart_x={cart_pos[0].item():+.3f} m  "
                    f"min={cart_pos_min[0].item():+.3f}  max={cart_pos_max[0].item():+.3f}"
                )

            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
