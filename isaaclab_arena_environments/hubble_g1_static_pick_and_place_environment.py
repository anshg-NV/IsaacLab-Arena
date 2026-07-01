# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Datagen stuff copied from galileo_g1_locomanip_pick_and_place_environment.py
_LEGACY_DATAGEN_NAME = "locomanip_pick_and_place_D0"


def _is_legacy_pair(pick_up_object_name: str, destination_name: str) -> bool:
    return pick_up_object_name == "apple_01_objaverse_robolab" and destination_name == "clay_plates_hot3d_robolab"


def _apply_legacy_datagen_name_override(
    env_cfg: Any,
    pick_up_object_name: str,
    destination_name: str,
) -> Any:
    """Rewrite the Mimic ``datagen_config.name`` to the legacy value for the v0.2 workflow.

    Only applies to Mimic configs (where ``datagen_config`` exists) and only to the exact
    ``(apple_01_objaverse_robolab, clay_plates_hot3d_robolab)`` pair that was SQA'd against this datagen key. All other
    pairs keep the templated name produced by ``G1PickAndPlaceMimicEnvCfg``.
    """
    if not _is_legacy_pair(pick_up_object_name, destination_name):
        return env_cfg

    datagen_config = getattr(env_cfg, "datagen_config", None)
    if datagen_config is None:
        return env_cfg

    print(
        f"Overriding Mimic datagen_config.name from {datagen_config.name} to the legacy {_LEGACY_DATAGEN_NAME}"
        "This preserves identical behavior with existing Mimic datasets"
        "Remove this in the future when checkpoints are retrained."
    )
    datagen_config.name = _LEGACY_DATAGEN_NAME
    return env_cfg


@register_environment
class HubbleG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """G1 (WBC-balanced, no nav) pick-and-place on a table.

    Defaults to the apple-to-plate pairing so this env composes cleanly into the existing
    apple-to-plate workflow (record_demos -> replay -> eval) without requiring locomotion.
    """

    name: str = "hubble_g1_static_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.hubble_pick_and_place import HubblePickAndPlaceMimicEnvCfg, HubblePickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange
        from isaaclab_arena_environments.mdp.galileo_g1_static_pick_and_place.robot_configs import (
            G1_STATIC_FINGER_DYNAMIC_FRICTION,
            G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            G1_STATIC_FINGER_PRIM_NAME_MARKERS,
            G1_STATIC_FINGER_STATIC_FRICTION,
            G1_STATIC_OPEN_ARM_JOINT_POS,
        )

        background = self.asset_registry.get_asset_by_name("hubble_background")()
        table = self.asset_registry.get_asset_by_name("hubble_table")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)(scale=(0.009, 0.009, 0.009))
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)(scale=(0.5, 0.5, 0.5))

        # Add ground plane and light to the scene
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        assets = [background, table, light, ground_plane, pick_up_object, destination]

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            lock_waist=True,
        )
        embodiment.set_finger_contact_friction(
            material_path=G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            static_friction=G1_STATIC_FINGER_STATIC_FRICTION,
            dynamic_friction=G1_STATIC_FINGER_DYNAMIC_FRICTION,
            prim_name_markers=G1_STATIC_FINGER_PRIM_NAME_MARKERS,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set all positions
        pick_up_object.set_initial_pose(PoseRange(
            position_xyz_min=(0.10, 0.15, 0.775),
            position_xyz_max=(0.20, 0.25, 0.775),
        ))
        destination.set_initial_pose(Pose(position_xyz=(-0.10, 0.25, 0.755)))

        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.55, 0.78), rotation_xyzw=(0.0, 0.0, -0.7071068, 0.7071068)))
        embodiment.set_joint_initial_pos(G1_STATIC_OPEN_ARM_JOINT_POS)

        task_description = f"Pick up the {args_cli.object.replace("_", " ")} from the table and place it onto the {args_cli.destination.replace("_", " ")}."

        def env_cfg_callback(env_cfg):
            return _apply_legacy_datagen_name_override(
                env_cfg,
                pick_up_object_name=pick_up_object.name,
                destination_name=destination.name,
            )

        def _build_hubble_pick_and_place_mimic_cfg(arm_mode):
            return HubblePickAndPlaceMimicEnvCfg(
                pick_up_object_name=pick_up_object.name,
                destination_location_name=destination.name,
                arm_mode=arm_mode,
            )

        scene = Scene(assets=assets)

        task = HubblePickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination,
            background_scene=background,
            episode_length_s=6.0,
            task_description=task_description,
            # Mirror the locomanip env's success thresholds so metrics are comparable.
            force_threshold=0.5,
            velocity_threshold=0.1,
            mimic_env_cfg_factory=_build_hubble_pick_and_place_mimic_cfg,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="apple_01_objaverse_robolab")
        parser.add_argument("--destination", type=str, default="clay_plates_hot3d_robolab")
        # Default embodiment is g1_wbc_agile_pink: AGILE end-to-end velocity policy for
        # whole-body balance + PinkIK upper body. The static task never walks, so AGILE's
        # single-policy backend is a better fit than HOMIE's stand+walk split (which
        # ``g1_wbc_pink`` ships). Same 23-D action layout and OpenXR retargeter as the
        # locomanip env -- the only knob that flips is which lower-body ONNX policy gets
        # loaded by the WBC factory. ``g1_wbc_pink`` is still accepted as an override
        # for users who specifically want HOMIE.
        parser.add_argument("--embodiment", type=str, default="g1_wbc_agile_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
