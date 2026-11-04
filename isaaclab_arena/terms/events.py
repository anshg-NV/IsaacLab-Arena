# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Velocity | None = None,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # Convert the pose to the env frame
    pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_xyzw[:, :3] += env.scene.env_origins[env_ids]
    # Set the pose and velocity
    asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_ids)
    if velocity is not None:
        vel = velocity.to_tensor(device=env.device).unsqueeze(0).expand(num_envs, -1)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)
    else:
        asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


def set_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    if env_ids is None:
        return

    # Grab the object
    asset = env.scene[asset_cfg.name]

    # Set the objects pose in each environment independently
    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        # Convert the pose to the env frame
        pose = pose_list[cur_env]
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[cur_env, :]
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=torch.tensor([cur_env], device=env.device))
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
        )


def reset_all_articulation_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_root_pose_range: dict[str, tuple[float, float]] | None = None,
    robot_joint_names: tuple[str, ...] = (),
    robot_joint_position_range: tuple[float, float] | None = None,
) -> None:
    """Reset articulations, optionally offsetting the robot before controller resets.

    Args:
        env: Environment whose articulations are reset.
        env_ids: Selected environment indices.
        robot_root_pose_range: Independent uniform world X/Y offsets [m] and world-Z
            yaw offsets [rad], relative to the default robot pose. Yaw rotates about
            the robot root; height, nominal roll/pitch, velocities and joints retain
            their default reset values unless joint offsets are configured.
        robot_joint_names: Robot joint names or regex patterns to perturb.
        robot_joint_position_range: Independent additive uniform joint offsets [rad]
            around default positions, clamped to hard joint limits. Applied before
            controller resets; matching position targets are updated too.
            None preserves the original joint reset behavior.
    """
    if len(env_ids) == 0:
        return
    if robot_joint_position_range is not None:
        assert "robot" in env.scene.articulations, "Robot joint randomization requires a robot articulation."
        assert robot_joint_names, "Robot joint randomization requires joint names."
        assert len(robot_joint_position_range) == 2 and robot_joint_position_range[0] <= robot_joint_position_range[1]
    if robot_root_pose_range is not None:
        assert "robot" in env.scene.articulations, "Robot root randomization requires a robot articulation."
        assert set(robot_root_pose_range) == {"x", "y", "yaw"}
    for name, articulation_asset in env.scene.articulations.items():
        # obtain default and deal with the offset for env origins
        default_root_state = wp.to_torch(articulation_asset.data.default_root_state)[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        if name == "robot" and robot_root_pose_range is not None:
            bounds = torch.tensor(
                [robot_root_pose_range[key] for key in ("x", "y", "yaw")],
                dtype=default_root_state.dtype,
                device=env.device,
            )
            offsets = bounds[:, 0] + torch.rand((len(env_ids), 3), device=env.device) * (bounds[:, 1] - bounds[:, 0])
            default_root_state[:, :2] += offsets[:, :2]
            x, y, z, w = default_root_state[:, 3:7].clone().unbind(-1)
            s, c = torch.sin(offsets[:, 2] / 2), torch.cos(offsets[:, 2] / 2)
            default_root_state[:, 3:7] = torch.stack(
                (c * x - s * y, s * x + c * y, c * z + s * w, c * w - s * z), dim=-1
            )
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = wp.to_torch(articulation_asset.data.default_joint_pos)[env_ids].clone()
        default_joint_vel = wp.to_torch(articulation_asset.data.default_joint_vel)[env_ids].clone()
        if name == "robot" and robot_joint_position_range is not None:
            joint_ids, _ = articulation_asset.find_joints(robot_joint_names)
            assert joint_ids, "Robot joint randomization matched no joints."
            low, high = robot_joint_position_range
            shape = (len(env_ids), len(joint_ids))
            offsets = torch.full(shape, low, dtype=default_joint_pos.dtype, device=env.device)
            if low != high:
                offsets += torch.rand(shape, dtype=default_joint_pos.dtype, device=env.device) * (high - low)
            limits = articulation_asset.data.joint_pos_limits.torch[env_ids][:, joint_ids]
            default_joint_pos[:, joint_ids] = torch.clamp(default_joint_pos[:, joint_ids] + offsets, min=limits[..., 0], max=limits[..., 1])
            articulation_asset.set_joint_position_target(default_joint_pos[:, joint_ids], joint_ids=joint_ids, env_ids=env_ids)
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
