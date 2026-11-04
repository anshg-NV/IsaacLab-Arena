# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.hubble_g1_static_pick_and_place_environment import (
    HubbleG1StaticPickAndPlaceEnvironment,
    _configure_minimal_teleop_scene,
)

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.tasks.task_base import TaskBase


KLT_USD_PATH = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/"
    "assets/object_library/KLT_Bin/small_KLT.usd"
)


def _configure_box_lift_teleop_scene(env: Any, env_ids: Any) -> None:
    """Keep only the task assets visible, with a matte table that contrasts with G1's hands."""
    from pxr import Gf, Sdf, Usd, UsdShade

    _configure_minimal_teleop_scene(env, env_ids)
    stage = env.scene.stage
    selected_ids = range(len(env.scene.env_prim_paths)) if env_ids is None else env_ids.tolist()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for env_id in selected_ids:
            root = env.scene.env_prim_paths[int(env_id)]
            material = UsdShade.Material.Define(stage, f"{root}/TeleopLooks/Tablecloth")
            shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.58, 0.60))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for name in ("TableclothTop", "DrapePerimeter"):
                prim = stage.GetPrimAtPath(f"{root}/hubble_lab_visuals/{name}")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


@register_environment
class HubbleBoxLiftEnvironment(HubbleG1StaticPickAndPlaceEnvironment):
    """Place the apple in the purple KLT box, then lift it with both G1 hands."""

    name: str = "hubble_box_lift"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        env = super().get_env(args_cli)
        # Keep the inherited Hubble PnP reset posture. The hand material's "max"
        # combine rule controls box contact, so halve its coefficients here.
        finger_material = env.embodiment.event_config.apply_high_friction_to_g1_fingers.params
        finger_material["static_friction"] *= 0.5
        finger_material["dynamic_friction"] *= 0.5
        hubble_callback = env.env_cfg_callback

        def env_cfg_callback(env_cfg):
            env_cfg = hubble_callback(env_cfg)
            env_cfg.episode_length_s = env.task.episode_length_s
            if args_cli.minimal_teleop_scene:
                env_cfg.events.minimal_teleop_scene.func = _configure_box_lift_teleop_scene
            if args_cli.teleop_device == "openxr":
                # Minimal ignores the lab's room lights. Apply its shading and ambient
                # fill after the base config, whose zero ambient overrides Kit CLI flags.
                env_cfg.sim.render.carb_settings["/rtx/minimal/mode"] = 3
                env_cfg.sim.render.carb_settings["/rtx/sceneDb/ambientLightIntensity"] = 3.0
            return env_cfg

        env.env_cfg_callback = env_cfg_callback
        return env

    def _make_destination(self, args_cli: argparse.Namespace) -> Asset:
        from isaaclab.sim import MassPropertiesCfg

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType

        assert args_cli.object == "apple_01_objaverse_robolab", "Hubble box lift uses the calibrated Hubble apple"
        assert args_cli.destination == "purple_box", "Hubble box lift uses the native-size KLT box"
        return Object(
            name="purple_box", usd_path=KLT_USD_PATH, object_type=ObjectType.RIGID,
            spawn_cfg_addon={"mass_props": MassPropertiesCfg(mass=1.12)},
        )

    def _make_task(self, pick_up_object: Asset, destination: Asset, background: Asset, table: Asset) -> TaskBase:
        from isaaclab_arena.tasks.hubble_box_lift import HubbleBoxLiftTask
        from isaaclab_arena.utils.pose import PoseRange

        table_top_z = float(table.get_world_bounding_box().top_surface_z.item())
        # The long side runs across G1; shift both regions 5 cm ego-right (world -X).
        for obj, xy_min, xy_max, yaw in (
            (pick_up_object, (0.19, 0.20), (0.23, 0.28), 0.0),
            (destination, (-0.06, 0.10), (-0.02, 0.14), math.pi / 2),
        ):
            z = float(table_top_z + 0.002 - obj.get_bounding_box().min_point[0, 2])
            obj.set_initial_pose(
                PoseRange(
                    position_xyz_min=(*xy_min, z),
                    position_xyz_max=(*xy_max, z),
                    rpy_min=(0.0, 0.0, yaw),
                    rpy_max=(0.0, 0.0, yaw),
                )
            )
        return HubbleBoxLiftTask(pick_up_object, destination, support_surface_z=table_top_z)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        HubbleG1StaticPickAndPlaceEnvironment.add_cli_args(parser)
        parser.set_defaults(destination="purple_box")
