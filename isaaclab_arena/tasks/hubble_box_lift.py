# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch
import warp as wp

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mimic_env_cfg import (
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.hubble_pick_and_place import HubblePickAndPlaceMimicEnvCfg
from isaaclab_arena.tasks.task_base import TaskBase


class BoxLiftSuccess:
    """Require released apple placement before a sustained, two-handed loaded-box lift."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.placed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.placement_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.lift_steps = torch.zeros_like(self.placement_steps)
        self.last_step = torch.full_like(self.placement_steps, -1)
        self.contact_age = torch.full((env.num_envs, 3), float("inf"), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear phase and hold history for the reset environments."""
        ids = slice(None) if env_ids is None else env_ids
        self.placed[ids] = False
        self.placement_steps[ids] = 0
        self.lift_steps[ids] = 0
        self.last_step[ids] = -1
        self.contact_age[ids] = float("inf")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        box_cfg: SceneEntityCfg,
        support_surface_z: float,
        object_center_offset_m: tuple[float, float, float],
        object_radius_m: float,
        box_corners_m: tuple[tuple[float, float, float], ...],
    ) -> torch.Tensor:
        """Evaluate containment, contact, sequence, and hold duration in each environment.

        Args:
            env: The environment being evaluated.
            object_cfg: The apple scene entity.
            box_cfg: The purple box scene entity.
            support_surface_z: Tabletop height relative to each environment origin [m].
            object_center_offset_m: Apple bounds center relative to its root [m].
            object_radius_m: Conservative apple half-size for containment [m].
            box_corners_m: Eight local box bounds corners for ground clearance [m].

        Returns:
            Whether the full ordered task has succeeded in each environment.
        """
        obj, box = env.scene[object_cfg.name], env.scene[box_cfg.name]
        object_pos = wp.to_torch(obj.data.root_pos_w)
        box_pos = wp.to_torch(box.data.root_pos_w)
        box_quat = wp.to_torch(box.data.root_quat_w)
        object_center = object_pos + quat_apply(
            wp.to_torch(obj.data.root_quat_w), object_pos.new_tensor(object_center_offset_m).expand_as(object_pos)
        )
        local_center = quat_apply_inverse(box_quat, object_center - box_pos)
        # Native small_KLT inner wall faces and floor, measured from its collision cubes.
        inner_min = local_center.new_tensor((-0.0875, -0.1267, -0.0708))
        inner_max = local_center.new_tensor((0.0875, 0.1267, 0.0731))
        # The bounding sphere slightly overestimates the mesh at the walls.
        # Allow 5 mm horizontally and 8 mm at the non-spherical lower surface.
        lower = inner_min + object_radius_m
        upper = inner_max - object_radius_m
        lower[:2] -= 0.005
        upper[:2] += 0.005
        lower[2] -= 0.008
        contained = ((local_center >= lower) & (local_center <= upper)).all(dim=1)

        def contact_force(sensor_name: str) -> torch.Tensor:
            forces = wp.to_torch(env.scene[sensor_name].data.force_matrix_w)
            return torch.linalg.vector_norm(forces, dim=-1).sum(dim=(1, 2))

        supported = contact_force("pick_up_object_contact_sensor") > 0.5
        released = contact_force("apple_hand_contact_sensor") < 0.05
        left_contact = contact_force("box_left_hand_contact_sensor") > 0.5
        right_contact = contact_force("box_right_hand_contact_sensor") > 0.5
        # Contact impulses can disappear briefly as the apple rolls or fingers
        # shift. Require recent contact, with a grace shorter than the lift hold.
        fresh = env.episode_length_buf != self.last_step
        contacts = torch.stack((supported, left_contact, right_contact), dim=1)
        self.contact_age = torch.where(
            fresh[:, None], torch.where(contacts, 0.0, self.contact_age + env.step_dt), self.contact_age
        )
        supported, left_contact, right_contact = (self.contact_age <= 0.1).unbind(dim=1)
        corners = box_pos.new_tensor(box_corners_m).expand(env.num_envs, -1, -1)
        rotated_corners = quat_apply(box_quat[:, None, :].expand(-1, 8, -1), corners)
        bottom_z = box_pos[:, 2] - env.scene.env_origins[:, 2] + rotated_corners[:, :, 2].amin(dim=1)
        up = quat_apply(box_quat, box_pos.new_tensor((0.0, 0.0, 1.0)).expand_as(box_pos))
        loaded = contained & supported & released
        on_table = torch.abs(bottom_z - support_surface_z) < 0.015
        placement = loaded & on_table & (up[:, 2] > math.cos(math.radians(20.0)))
        lifting = (
            self.placed & loaded & left_contact & right_contact
            & (bottom_z >= support_surface_z + 0.05)
            & (up[:, 2] > math.cos(math.radians(40.0)))
        )
        # Recording and datagen can query success repeatedly between simulation steps.
        self.placement_steps = torch.where(
            fresh, torch.where(placement, self.placement_steps + 1, 0), self.placement_steps
        )
        self.placed |= self.placement_steps >= math.ceil(0.2 / env.step_dt)
        self.lift_steps = torch.where(fresh, torch.where(lifting, self.lift_steps + 1, 0), self.lift_steps)
        self.last_step.copy_(env.episode_length_buf)
        return lifting & (self.lift_steps >= math.ceil(0.4 / env.step_dt))


def box_lift_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    support_surface_z: float,
    object_center_offset_m: tuple[float, float, float],
    object_radius_m: float,
    box_corners_m: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    """Check ordered placement/lift success, including when called directly by recorders.

    Arguments match ``BoxLiftSuccess.__call__``. State belongs to the live environment,
    because recording and Mimic remove the success term from the termination manager.
    """
    if not hasattr(env, "_hubble_box_lift_success"):
        env._hubble_box_lift_success = BoxLiftSuccess(env)
    return env._hubble_box_lift_success(
        env, object_cfg, box_cfg, support_surface_z, object_center_offset_m, object_radius_m, box_corners_m
    )


def reset_box_lift_success(env: ManagerBasedRLEnv, env_ids: Sequence[int] | None) -> None:
    """Clear per-episode task history even when the success termination is disabled."""
    if hasattr(env, "_hubble_box_lift_success"):
        env._hubble_box_lift_success.reset(env_ids)


class HubbleBoxLiftTask(TaskBase):
    """Apple placement followed by a two-handed KLT lift on the Hubble table."""

    def __init__(self, pick_up_object: Asset, destination_location: Asset, *, support_surface_z: float):
        """Configure the task for the calibrated apple and native-size KLT asset.

        Args:
            pick_up_object: The calibrated Hubble apple.
            destination_location: The dynamic purple small KLT box at unit scale.
            support_surface_z: Tabletop height relative to each environment origin [m].
        """
        super().__init__(
            episode_length_s=30.0,
            task_description="Pick up the apple, place it inside the purple box, then lift the box with both hands.",
        )
        self.pick_up_object = pick_up_object
        self.destination_location = destination_location
        apple_sensor = pick_up_object.get_contact_sensor_cfg(contact_against_object=destination_location)
        box_sensor = destination_location.get_contact_sensor_cfg()
        # PhysX needs one rigid body per filter per environment; a hand wildcard
        # matches several bodies and cannot initialize a filtered contact view.
        hand_links = (
            "palm_link", "index_0_link", "index_1_link", "middle_0_link", "middle_1_link",
            "thumb_0_link", "thumb_1_link", "thumb_2_link",
        )
        left_hand = [f"{{ENV_REGEX_NS}}/Robot/left_hand_{link}" for link in hand_links]
        right_hand = [f"{{ENV_REGEX_NS}}/Robot/right_hand_{link}" for link in hand_links]
        self.scene_cfg = SceneCfg(
            pick_up_object_contact_sensor=apple_sensor,
            apple_hand_contact_sensor=ContactSensorCfg(
                prim_path=apple_sensor.prim_path,
                filter_prim_paths_expr=left_hand + right_hand,
            ),
            box_left_hand_contact_sensor=ContactSensorCfg(
                prim_path=box_sensor.prim_path, filter_prim_paths_expr=left_hand,
            ),
            box_right_hand_contact_sensor=ContactSensorCfg(
                prim_path=box_sensor.prim_path, filter_prim_paths_expr=right_hand,
            ),
        )
        apple_bounds = pick_up_object.get_bounding_box()
        self.terminations_cfg = TerminationsCfg(
            success=TerminationTermCfg(func=box_lift_success, params={
                "object_cfg": SceneEntityCfg(pick_up_object.name),
                "box_cfg": SceneEntityCfg(destination_location.name),
                "support_surface_z": support_surface_z,
                "object_center_offset_m": tuple(apple_bounds.center[0].tolist()),
                "object_radius_m": float(apple_bounds.size.max().item() / 2),
                "box_corners_m": tuple(
                    tuple(row) for row in destination_location.get_bounding_box().get_corners_at()[0].tolist()
                ),
            }),
            object_dropped=TerminationTermCfg(func=mdp.root_height_below_minimum, params={
                "minimum_height": support_surface_z - 0.15, "asset_cfg": SceneEntityCfg(pick_up_object.name),
            }),
            box_dropped=TerminationTermCfg(func=mdp.root_height_below_minimum, params={
                "minimum_height": support_surface_z - 0.15, "asset_cfg": SceneEntityCfg(destination_location.name),
            }),
        )

    def get_scene_cfg(self) -> SceneCfg:
        return self.scene_cfg

    def get_termination_cfg(self) -> TerminationsCfg:
        return self.terminations_cfg

    def get_events_cfg(self) -> EventsCfg:
        return EventsCfg()

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> HubbleBoxLiftMimicEnvCfg:
        return HubbleBoxLiftMimicEnvCfg(
            arm_mode=arm_mode, pick_up_object_name=self.pick_up_object.name,
            destination_location_name=self.destination_location.name,
        )


@configclass
class SceneCfg:
    pick_up_object_contact_sensor: ContactSensorCfg = MISSING
    apple_hand_contact_sensor: ContactSensorCfg = MISSING
    box_left_hand_contact_sensor: ContactSensorCfg = MISSING
    box_right_hand_contact_sensor: ContactSensorCfg = MISSING


@configclass
class EventsCfg:
    reset_box_lift_success: EventTermCfg = EventTermCfg(func=reset_box_lift_success, mode="reset")


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING
    box_dropped: TerminationTermCfg = MISSING


@configclass
class HubbleBoxLiftMimicEnvCfg(HubblePickAndPlaceMimicEnvCfg):
    """Left-hand apple placement and coordinated bimanual box lifting."""

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "hubble_box_lift_D0"

        def subtask(object_ref: str, signal: str | None = None) -> SubTaskConfig:
            return SubTaskConfig(
                object_ref=object_ref, subtask_term_signal=signal, subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object", selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005, num_interpolation_steps=0, num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )

        box = self.destination_location_name
        self.subtask_configs["left"] = [
            subtask(self.pick_up_object_name, "grasp_apple"), subtask(box, "place_apple"),
            subtask(box, "grasp_box_left"), subtask(box),
        ]
        self.subtask_configs["right"] = [
            subtask(box, "grasp_box_right"), subtask(box),
        ]
        self.subtask_configs["body"] = [subtask(box)]
        self.subtask_configs["body"][0].action_noise = 0.0
        self.task_constraint_configs = [SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=(("left", 3), ("right", 1)),
            constraint_type=SubTaskConstraintType.COORDINATION, coordination_synchronize_start=True,
            coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
        )]
