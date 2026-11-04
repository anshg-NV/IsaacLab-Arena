# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from enum import Enum

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.terminations import root_height_below_minimum
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_apply


class SuccessMode(str, Enum):
    """How `check_success` combines its results."""

    ALL = "ALL"
    """Success needs every predicate to be True."""

    ANY = "ANY"
    """Success needs at least one predicate to be True."""

    CHOOSE = "CHOOSE"
    """Success needs at least k predicates to be True."""


def check_success(
    env: ManagerBasedRLEnv,
    predicates: list[TerminationTermCfg],
    mode: SuccessMode | str = SuccessMode.ALL,
    k: int | None = None,
) -> torch.Tensor:
    """Compose multiple termination predicates into a single success signal.

    Each predicate is wrapped in a termination term and its `func` is evaluated with its
    `params` to produce a boolean tensor of shape (num_envs,). The results are then combined according to `mode`.

    - ALL: True where every predicate is True (logical AND).
    - ANY: True where at least one predicate is True (logical OR).
    - CHOOSE: True where at least k predicates are True.

    Args:
        env: The RL environment instance.
        predicates: Termination term configs to evaluate and combine.
        mode: How to combine the predicate results (SuccessMode or string).
        k: Number of predicates that must be True when `mode` is SuccessMode.CHOOSE.

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """

    if not predicates:
        raise ValueError("check_success requires at least one predicate.")
    valid_modes = [m.value for m in SuccessMode]
    if not (isinstance(mode, str) and mode.upper() in valid_modes):
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {valid_modes}.")
    mode = SuccessMode(mode.upper())

    results = torch.stack([predicate.func(env, **predicate.params) for predicate in predicates], dim=0)

    if mode is SuccessMode.ALL:
        return results.all(dim=0)
    if mode is SuccessMode.ANY:
        return results.any(dim=0)

    if k is None:
        raise ValueError("mode SuccessMode.CHOOSE requires 'k' to be provided.")
    if not (1 <= k <= len(predicates)):
        raise ValueError(f"'k' must be in [1, {len(predicates)}] for {len(predicates)} predicates, got {k}.")
    return results.sum(dim=0) >= k


# NOTE(alexmillane, 2025.09.15): The velocity threshold is set high because some stationary
# seem to generate a "small" velocity.
def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    object: RigidObject = env.unwrapped.scene[object_cfg.name]
    sensor: ContactSensor = env.unwrapped.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    assert sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(wp.to_torch(sensor.data.force_matrix_w), dim=-1).reshape(-1)
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = wp.to_torch(object.data.root_lin_vel_w)
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)

    return condition_met


def object_placed_on_target(
    env: ManagerBasedRLEnv,
    placement_radius_m: float,
    placement_z_m: float,
    min_object_height_above_target_m: float,
    max_object_height_m: float,
    min_object_height_m: float,
    min_target_height_m: float,
    min_contact_force: float,
    release_clearance_m: float,
    max_object_speed: float,
    task_link_names: tuple[str, str],
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("destination"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    object_center_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_center_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_target_height_m: float | None = None,
) -> torch.Tensor:
    """Check released, settled placement.

    Positions and height thresholds are relative to each environment origin [m].
    Contact is filtered apple-to-plate force [N]; speed is linear only [m/s].
    The circular footprint and both-hand clearance are independent of robot yaw.
    This is the instantaneous predicate; AutoData qualifies its final success hold
    separately before labeling/exporting generated episodes.
    Optional scaled local center offsets correct off-center asset roots for XY
    placement and hand clearance. Height limits still refer to rigid-body roots.

    Args:
        env: Environment being evaluated.
        placement_radius_m: Inclusive XY distance between the configured object/target centers [m].
        placement_z_m: Inclusive absolute object-to-target root-height limit [m].
        min_object_height_above_target_m: Exclusive lower object-to-target height delta [m].
        max_object_height_m: Exclusive upper supported object height [m].
        min_object_height_m: Exclusive lower supported object height [m].
        min_target_height_m: Exclusive lower supported target height [m].
        min_contact_force: Exclusive lower filtered contact-force magnitude [N].
        release_clearance_m: Exclusive lower distance from each task link to the object [m].
        max_object_speed: Exclusive upper object linear speed [m/s].
        task_link_names: Two distinct robot links, one near each gripper.
        object_cfg: Object being placed.
        robot_cfg: Robot providing the task links.
        target_cfg: Live placement target.
        contact_sensor_cfg: Object contact sensor filtered against the target.
        object_center_offset_m: Scaled object-local root-to-bounds-center offset [m].
        target_center_offset_m: Scaled target-local root-to-bounds-center offset [m].
        max_target_height_m: Optional exclusive upper supported target root height [m].

    Returns:
        Boolean success tensor with one entry per environment.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    robot = env.scene[robot_cfg.name]
    contact_sensor = env.scene[contact_sensor_cfg.name]

    object_pos = wp.to_torch(obj.data.root_pos_w) - env.scene.env_origins
    target_pos = wp.to_torch(target.data.root_pos_w) - env.scene.env_origins
    object_center = object_pos
    target_center = target_pos
    if any(object_center_offset_m):
        object_center = object_pos + quat_apply(
            wp.to_torch(obj.data.root_quat_w), object_pos.new_tensor(object_center_offset_m).expand_as(object_pos)
        )
    if any(target_center_offset_m):
        target_center = target_pos + quat_apply(
            wp.to_torch(target.data.root_quat_w), target_pos.new_tensor(target_center_offset_m).expand_as(target_pos)
        )
    on_target = torch.linalg.vector_norm(object_center[:, :2] - target_center[:, :2], dim=1) <= placement_radius_m
    target_height_delta = object_pos[:, 2] - target_pos[:, 2]
    within_target_height = target_height_delta.abs() <= placement_z_m
    above_target = target_height_delta > min_object_height_above_target_m
    supported = (object_pos[:, 2] > min_object_height_m) & (target_pos[:, 2] > min_target_height_m)
    if max_target_height_m is not None:
        supported &= target_pos[:, 2] < max_target_height_m
    resting = object_pos[:, 2] < max_object_height_m

    settled = torch.linalg.vector_norm(wp.to_torch(obj.data.root_lin_vel_w), dim=1) < max_object_speed
    contact_force = torch.linalg.vector_norm(wp.to_torch(contact_sensor.data.force_matrix_w), dim=-1).amax(dim=(1, 2))
    on_plate = contact_force > min_contact_force

    if not isinstance(task_link_names, list | tuple) or (
        len(task_link_names) != 2
        or any(not isinstance(name, str) or not name for name in task_link_names)
        or len(set(task_link_names)) != 2
    ):
        raise ValueError("task_link_names must contain two distinct non-empty robot links")
    link_indices = [robot.data.body_names.index(name) for name in task_link_names]
    link_pos = wp.to_torch(robot.data.body_pos_w)[:, link_indices, :] - env.scene.env_origins[:, None, :]
    released = (torch.linalg.vector_norm(link_pos - object_center[:, None, :], dim=2) > release_clearance_m).all(dim=1)

    return on_target & within_target_height & above_target & supported & resting & settled & on_plate & released


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Multi-object version of `object_on_destination`.

    Returns True only when ALL objects in the list satisfy the destination condition.
    See `object_on_destination` for details on the single-object logic.
    """
    condition_met = torch.ones((env.unwrapped.num_envs), device=env.unwrapped.device, dtype=torch.bool)
    for object_cfg, contact_sensor_cfg in zip(object_cfg_list, contact_sensor_cfg_list):
        single_condition = object_on_destination(
            env=env,
            object_cfg=object_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
        )
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met


def objects_in_proximity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns:
        Boolean tensor indicating when objects are within a certain proximity of each other.
    """
    # Get object entities from the scene
    object: RigidObject = env.scene[object_cfg.name]
    target_object: RigidObject = env.scene[target_object_cfg.name]

    # Get positions relative to environment origin
    object_pos = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    target_object_pos = wp.to_torch(target_object.data.root_pos_w) - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def lift_object_il_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_position: tuple[float, float, float] | None = None,
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        goal_position: Fixed goal position [x, y, z] to use if command goal not available.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """

    object_instance: RigidObject = env.scene[object_cfg.name]
    object_pos = wp.to_torch(object_instance.data.root_pos_w)

    goal_pos = torch.tensor([goal_position] * env.num_envs, device=env.device)

    # Check if object is within tolerance of goal
    distance = torch.norm(object_pos - goal_pos, dim=1)
    return distance < position_tolerance


def lift_object_rl_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rl_training: bool = False,
    command_name: str = "object_pose",
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Supports multiple modes:
    - RL training: Always returns False (no early termination)
    - RL evaluation: Uses goal from command manager

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        robot_cfg: The robot configuration (needed to transform the goal to world frame).
        rl_training: If True, always returns False (disables success termination for RL training).
        command_name: The name of the command that is used to control the object.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot: RigidObject = env.scene[robot_cfg.name]
    object_instance: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]

    # Transform goal from robot-base frame to world frame
    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)
    des_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, des_pos_b)

    object_pos_w = wp.to_torch(object_instance.data.root_pos_w)
    distance = torch.linalg.norm(des_pos_w - object_pos_w[:, :3], dim=1)
    return distance < position_tolerance


def goal_pose_task_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_x_range: tuple[float, float] | None = None,
    target_y_range: tuple[float, float] | None = None,
    target_z_range: tuple[float, float] | None = None,
    target_orientation_xyzw: tuple[float, float, float, float] | None = None,
    target_orientation_tolerance_rad: float = 0.1,
) -> torch.Tensor:
    """Terminate when the object's pose is within the thresholds (BBox + Orientation).

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        target_x_range: Success zone x-range [min, max] in meters.
        target_y_range: Success zone y-range [min, max] in meters.
        target_z_range: Success zone z-range [min, max] in meters.
        target_orientation_xyzw: Target quaternion [x, y, z, w].
        target_orientation_tolerance_rad: Angular tolerance in radians (default: 0.1).

    Returns:
        A boolean tensor of shape (num_envs, )
    """
    object_instance: RigidObject = env.scene[object_cfg.name]
    object_root_pos_w = wp.to_torch(object_instance.data.root_pos_w)
    object_root_quat_w = wp.to_torch(object_instance.data.root_quat_w)

    device = env.device
    num_envs = env.num_envs

    has_any_threshold = any([
        target_x_range is not None,
        target_y_range is not None,
        target_z_range is not None,
        target_orientation_xyzw is not None,
    ])

    if not has_any_threshold:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    success = torch.ones(num_envs, dtype=torch.bool, device=device)

    # Position range checks
    ranges = [target_x_range, target_y_range, target_z_range]
    for idx, range_val in enumerate(ranges):
        if range_val is not None:
            range_min, range_max = range_val
            in_range = (object_root_pos_w[:, idx] >= range_min) & (object_root_pos_w[:, idx] <= range_max)
            success &= in_range

    # Orientation check
    if target_orientation_xyzw is not None:
        target_quat = torch.tensor(target_orientation_xyzw, device=device, dtype=torch.float32).unsqueeze(0)

        # Formula: |<q1, q2>| > cos(tolerance / 2)
        quat_dot = torch.sum(object_root_quat_w * target_quat, dim=-1)
        abs_dot = torch.abs(quat_dot)
        min_cos = math.cos(target_orientation_tolerance_rad / 2.0)

        ori_success = abs_dot >= min_cos
        success &= ori_success

    return success


def root_height_below_minimum_multi_objects(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("robot")],
) -> torch.Tensor:
    """Terminate when any asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    outs = [
        root_height_below_minimum(env=env, minimum_height=minimum_height, asset_cfg=asset_cfg)
        for asset_cfg in asset_cfg_list
    ]
    outs_tensor = torch.stack(outs, dim=0)  # [X, N]
    terminated = outs_tensor.any(dim=0)  # [N], bool
    return terminated
