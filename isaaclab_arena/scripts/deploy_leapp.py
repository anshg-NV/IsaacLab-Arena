# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deploy a LEAPP-exported RobomimicStandPolicy in an Isaac Lab Arena simulation.

Arena environments are composed via ``ArenaEnvBuilder`` (they are not in the Isaac Lab gym
registry), so they cannot use Isaac Lab's ``scripts/reinforcement_learning/leapp/deploy.py``.
This driver mirrors that script for Arena: it composes the env cfg from the standard Arena CLI,
then runs the exported model through ``RobomimicStandLeappDeploymentEnv`` (which keeps the
observation + WBC action managers and swaps the LEAPP ``InferenceManager`` in for the neural net).

The env is specified exactly as for ``policy_runner`` (example-environment subcommand or
``--env_graph_spec_yaml``, ``--embodiment ...``, ``--enable_cameras``); add ``--leapp_model <path>``.
"""

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    # Base Arena CLI (app launcher + Isaac Lab + Arena args); parse once to launch the sim app.
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--leapp_model", type=str, required=True, help="Path to the LEAPP .yaml pipeline description."
    )
    args_cli, _ = parser.parse_known_args()

    sim_ctx = SimulationAppContext(args_cli)
    with sim_ctx:
        import torch

        from isaaclab_arena.evaluation.robomimic_stand_leapp_deployment_env import RobomimicStandLeappDeploymentEnv
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        # Add the example-environment args, then resolve the Arena env cfg (no gym env built yet).
        env_parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli, hydra_overrides = env_parser.parse_known_args()

        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        # build_registered returns (name, cfg, env_kwargs); the deployment env builds the env itself.
        _, env_cfg, _ = arena_builder.build_registered()

        if getattr(args_cli, "seed", None) is not None:
            env_cfg.seed = args_cli.seed
        if getattr(args_cli, "device", None) is not None:
            env_cfg.sim.device = args_cli.device

        env = RobomimicStandLeappDeploymentEnv(env_cfg, args_cli.leapp_model)

        print(f"[INFO]: Deploying Arena env with LEAPP model: {args_cli.leapp_model}")
        print(f"[INFO]: Num envs: {env.num_envs}, step_dt: {env.step_dt:.4f}s, device: {env.device}")

        env.reset()
        try:
            with torch.inference_mode():
                while sim_ctx.is_running():
                    env.step()
        except KeyboardInterrupt:
            pass
        env.close()


if __name__ == "__main__":
    main()
