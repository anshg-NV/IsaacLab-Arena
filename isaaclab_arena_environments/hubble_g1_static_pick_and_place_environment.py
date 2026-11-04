# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.tasks.task_base import TaskBase


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


def _configure_recording_camera(env: Any, env_ids: Any) -> None:
    """Record from the calibrated tripod camera in the selected environment."""
    recorder = env.cfg.video_recorder
    viewer = env.cfg.viewer
    if recorder is None:
        return
    assert 0 <= viewer.env_index < env.num_envs, "Recording viewer env_index is out of range"
    # The composed USD transform already includes the cloned environment's origin.
    recorder.camera_prim_path = f"/World/envs/env_{viewer.env_index}/hubble_lab_visuals/SceneCamera"
    recorder.backend_source = "renderer"
    recorder.window_width = 960
    recorder.window_height = 720  # Match the calibrated 4:3 apertures; do not crop to 16:9.
    print(f"[Hubble recording camera] camera={recorder.camera_prim_path} resolution=960x720")


def _configure_ego_intrinsics(env: Any, env_ids: Any) -> None:
    """Apply the recorded D435 pinhole calibration, including its principal point [pixel]."""
    import torch
    from pxr import Gf

    from isaaclab.sim.utils import find_matching_prims
    from isaaclab_arena_environments.hubble_depth import DEPTH_INTRINSICS

    # Isaac Sim 6's depth-only sensor initialization disables color globally.
    # This environment also records RGB, so keep the shared color renderer enabled.
    env.sim.set_setting("/rtx/sdg/force/disableColorRender", False)
    # The ordinary Isaac Lab aperture conversion centers the principal point. RTX's
    # native OpenCV lens schema preserves the actual CameraInfo calibration instead.
    calibrations = {
        "robot_head_cam": (607.8684692382812, 607.80810546875, 319.8163146972656, 253.15415954589844),
        "robot_head_depth_cam": DEPTH_INTRINSICS,
    }
    for sensor_name, (native_fx, native_fy, native_cx, native_cy) in calibrations.items():
        sensor = env.scene[sensor_name]
        height, width = sensor.image_shape
        sx, sy = width / 640.0, height / 480.0
        fx, fy = native_fx * sx, native_fy * sy
        # Match the pixel-center convention of the deployment stack's OpenCV resize.
        cx, cy = (native_cx + 0.5) * sx - 0.5, (native_cy + 0.5) * sy - 0.5
        prefix = "omni:lensdistortion:opencvPinhole:"
        for prim in find_matching_prims(sensor.cfg.prim_path, stage=env.scene.stage):
            assert prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI"), "OpenCV lens schema is unavailable"
            prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
            prim.GetAttribute(prefix + "imageSize").Set(Gf.Vec2i(width, height))
            for name, value in zip(("fx", "fy", "cx", "cy"), (fx, fy, cx, cy)):
                prim.GetAttribute(prefix + name).Set(value)
            for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"):
                prim.GetAttribute(prefix + name).Set(0.0)
            # Keep the nominal calibration immutable so episodic focal DR never accumulates.
            prim.SetCustomDataByKey("isaaclab_arena:nominal_opencv_intrinsics", Gf.Vec4d(fx, fy, cx, cy))
        sensor._data.intrinsic_matrices.torch[:] = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=env.device
        )


def _align_table_visuals(env: Any, env_ids: Any, height_offset_m: float) -> None:
    """Move the separate cloth and clamp visuals with the physical tabletop."""
    from pxr import UsdGeom

    selected_ids = range(len(env.scene.env_prim_paths)) if env_ids is None else env_ids.tolist()
    for env_id in selected_ids:
        root = f"{env.scene.env_prim_paths[int(env_id)]}/hubble_lab_visuals"
        for name in ("TableclothTop", "DrapePerimeter", "Clamps"):
            prim = env.scene.stage.GetPrimAtPath(f"{root}/{name}")
            assert prim.IsValid(), f"Missing table visual: {prim.GetPath()}"
            if name == "Clamps":
                # The asset authors this grouping prim without a transformable type.
                prim.SetTypeName("Xform")
            UsdGeom.Xformable(prim).AddTranslateOp(opSuffix="table_height").Set((0.0, 0.0, height_offset_m))


def _apply_calibrated_materials(env: Any, env_ids: Any) -> None:
    """Apply calibrated apple/table materials and a smooth plate while preserving all meshes."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    settings = {
        "hubble_table": ("sm_tabletop_a01_top_01", {
            "diffuse_tint": (0.755, 0.657, 0.461), "specular_level": 0.0,
            "reflection_roughness_constant": 1.0, "reflection_roughness_texture_influence": 0.0,
        }),
    }
    apple_override_path = str(
        Path(__file__).resolve().parents[3] / "agentic_design/lane29/assets/apple/apple_visual_override.usdc"
    )
    apple_override = Usd.Stage.Open(apple_override_path)
    apple_colors = UsdGeom.PrimvarsAPI(apple_override.GetPrimAtPath("/Apple/mesh")).GetPrimvar("displayColor").Get()
    material_env_ids = range(len(env.scene.env_prim_paths)) if env_ids is None else env_ids.tolist()
    for env_id in material_env_ids:
        apple_path = f"{env.scene.env_prim_paths[int(env_id)]}/apple_01_objaverse_robolab"
        apple_mesh = env.scene.stage.GetPrimAtPath(apple_path + "/mesh")
        # Change only shading attributes, never reload a live rigid-body reference during reset.
        colors = UsdGeom.PrimvarsAPI(apple_mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex")
        colors.Set(apple_colors)
        colors.BlockIndices()
        apple_skin = UsdShade.Material.Define(env.scene.stage, apple_path + "/Looks/AppleSkin")
        apple_skin.GetPrim().GetReferences().SetReferences([Sdf.Reference(apple_override_path, "/Apple/Looks/AppleSkin")])
        UsdShade.MaterialBindingAPI.Apply(apple_mesh).Bind(apple_skin)
        assert len(UsdGeom.PrimvarsAPI(apple_mesh).GetPrimvar("displayColor").Get()) == 75405
        background_path = f"{env.scene.env_prim_paths[int(env_id)]}/hubble_background"
        concrete = UsdShade.Material.Define(env.scene.stage, background_path + "/Looks/NeutralConcrete")
        concrete.GetPrim().GetReferences().SetReferences([Sdf.Reference(str(
            Path(__file__).resolve().parents[3] / "agentic_design/lane29/assets/floor/floor_material.usda"
        ), "/FloorMaterial")])
        UsdShade.MaterialBindingAPI.Apply(env.scene.stage.GetPrimAtPath(background_path + "/floor")).Bind(concrete)
        robot_path = f"{env.scene.env_prim_paths[int(env_id)]}/Robot"
        chest = UsdGeom.Xform.Define(env.scene.stage, robot_path + "/torso_link/SceneChestAccessory")
        chest.GetPrim().GetReferences().SetReferences([Sdf.Reference(str(
            Path(__file__).resolve().parents[3] / "agentic_design/lane29/assets/robot_chest/robot_chest_visuals.usdc"
        ))])
        # The recorded robot has the ID strip, not the model's raised front wordmark.
        # Cut out only front-facing logo faces; keep rear lettering and all collision meshes.
        logo = UsdGeom.Mesh(env.scene.stage.GetPrimAtPath(robot_path + "/logo_link/visuals"))
        logo_points = logo.GetPointsAttr().Get()
        logo_indices = logo.GetFaceVertexIndicesAttr().Get()
        assert all(count == 3 for count in logo.GetFaceVertexCountsAttr().Get())
        front_faces = [i // 3 for i in range(0, len(logo_indices), 3)
                       if all(logo_points[index][0] > 0 for index in logo_indices[i:i + 3])]
        assert len(front_faces) == 2598, "Unexpected front/rear logo geometry"
        cutout = UsdShade.Material.Define(env.scene.stage, robot_path + "/Looks/HiddenOriginalFrontWordmark")
        cutout_shader = UsdShade.Shader.Define(env.scene.stage, str(cutout.GetPath()) + "/Shader")
        cutout_shader.CreateIdAttr("UsdPreviewSurface")
        cutout_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.0)
        cutout_shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
        cutout.CreateSurfaceOutput().ConnectToSource(cutout_shader.ConnectableAPI(), "surface")
        front_subset = UsdGeom.Subset.Define(env.scene.stage, str(logo.GetPath()) + "/ReferenceFrontWordmark")
        front_subset.CreateElementTypeAttr().Set("face")
        front_subset.CreateFamilyNameAttr().Set("materialBind")
        front_subset.CreateIndicesAttr().Set(front_faces)
        UsdGeom.Subset.SetFamilyType(logo, "materialBind", "nonOverlapping")
        UsdShade.MaterialBindingAPI.Apply(front_subset.GetPrim()).Bind(cutout)
        silver = UsdShade.Shader(env.scene.stage.GetPrimAtPath(robot_path + "/Looks/Aluminum_Anodized_Silver/Shader"))
        if silver:
            silver.GetInput("anodization_roughness").Set(0.32)
            silver.GetInput("enable_anodization_bump").Set(False)
        black = UsdShade.Shader(env.scene.stage.GetPrimAtPath(robot_path + "/Looks/Aluminum_Anodized/Shader"))
        assert black and black.GetInput("color_1"), "Missing G1 black anodized material"
        black.GetInput("color_1").Set(Gf.Vec3f(0.035, 0.035, 0.036))
        plate_path = f"{env.scene.env_prim_paths[int(env_id)]}/clay_plates_hot3d_robolab"
        if env.scene.stage.GetPrimAtPath(plate_path).IsValid():
            material = UsdShade.Material.Define(env.scene.stage, plate_path + "/Looks/CalibratedPlate")
            shader = UsdShade.Shader.Define(env.scene.stage, str(material.GetPath()) + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            # Oct 20 real ego reference: muted pink ceramic with a visible glaze highlight.
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.44, 0.24, 0.29))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.20)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for prim in Usd.PrimRange(env.scene.stage.GetPrimAtPath(plate_path)):
                if "obj_000003_Mesh" in prim.GetName():
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        for asset_name, (marker, inputs) in settings.items():
            asset = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[int(env_id)]}/{asset_name}")
            found = set()
            for prim in Usd.PrimRange(asset):
                if marker not in prim.GetName():
                    continue
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if not material:
                    continue
                for material_prim in Usd.PrimRange(material.GetPrim()):
                    if material_prim.GetTypeName() != "Shader":
                        continue
                    for name, value in inputs.items():
                        shader_input = UsdShade.Shader(material_prim).GetInput(name)
                        if shader_input:
                            shader_input.Set(Gf.Vec3f(*value) if isinstance(value, tuple) else value)
                            found.add(name)
            assert found == set(inputs), f"Missing Hubble shader inputs for {asset_name}: {set(inputs) - found}"
    # Visibility does not disable the existing ground-plane collision shape.
    UsdGeom.Imageable(env.scene.stage.GetPrimAtPath("/World/GroundPlane")).MakeInvisible()


def _configure_minimal_teleop_scene(env: Any, env_ids: Any) -> None:
    """Temporarily simplify the lab, robot shading, and lighting for readable teleoperation."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    stage = env.scene.stage
    selected_ids = range(len(env.scene.env_prim_paths)) if env_ids is None else env_ids.tolist()
    # Session opinions disappear on exit; the calibrated scene files stay untouched.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for env_id in selected_ids:
            root = env.scene.env_prim_paths[int(env_id)]
            # Keep the floor's collision shape for the balancing controller.
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{root}/hubble_background")).MakeInvisible()
            lab = stage.GetPrimAtPath(f"{root}/hubble_lab_visuals")
            for prim in lab.GetChildren():
                if prim.GetName() not in {"Looks", "TableclothTop", "DrapePerimeter", "SceneCamera"}:
                    prim.SetActive(False)
            # Metallic MDL surfaces have almost no diffuse response in RTX Minimal.
            # Keep the robot's light/dark parts distinct with session-only matte materials.
            robot_materials = {}
            for name, color in (
                ("Aluminum_Anodized", (0.12, 0.14, 0.16)),
                ("Aluminum_Anodized_Silver", (0.42, 0.44, 0.46)),
            ):
                material = UsdShade.Material.Define(stage, f"{root}/TeleopLooks/{name}")
                shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                robot_materials[name] = material
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{root}/Robot")):
                if prim.IsA(UsdGeom.Mesh):
                    UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
                    binding = UsdShade.MaterialBindingAPI.Apply(prim)
                    material, _ = binding.ComputeBoundMaterial()
                    if material and material.GetPrim().GetName() in robot_materials:
                        binding.Bind(robot_materials[material.GetPrim().GetName()])

        # RTX Minimal uses only the first distant light (or a camera light if none exists).
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdLux.DistantLight):
                prim.SetActive(False)
        light = UsdLux.DistantLight.Define(stage, "/World/TeleopFrontLight")
        light.CreateIntensityAttr(500.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        # The robot faces -Y; illuminate toward +Y and down, away from the tabletop.
        UsdGeom.Xformable(light).AddRotateXOp().Set(55.0)
        dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
        dome.CreateIntensityAttr(300.0)
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    print("[Hubble teleop] Minimal scene enabled: lab props and robot cast shadows disabled, cloth and front light enabled.")


@register_environment
class HubbleG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """G1 (WBC-balanced, no nav) pick-and-place on a table.

    Defaults to the apple-to-plate pairing so this env composes cleanly into the existing
    apple-to-plate workflow (record_demos -> replay -> eval) without requiring locomotion.
    """

    name: str = "hubble_g1_static_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp.galileo_g1_static_pick_and_place.robot_configs import (
            G1_STATIC_FINGER_DYNAMIC_FRICTION,
            G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            G1_STATIC_FINGER_PRIM_NAME_MARKERS,
            G1_STATIC_FINGER_STATIC_FRICTION,
        )

        background = self.asset_registry.get_asset_by_name("hubble_background")()
        # Measured tabletop height: 29.5 inches (0.7493 m); retain the table footprint.
        table = self.asset_registry.get_asset_by_name("hubble_table")(scale=(0.98, 0.90, 0.988438579682))
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)(scale=(0.0117551, 0.0121355, 0.0118187))
        destination = self._make_destination(args_cli)

        # Add ground plane and light to the scene
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=500.0)
        )
        lab_visuals = Background(
            name="hubble_lab_visuals",
            usd_path=str(Path(__file__).resolve().parents[1] / "isaaclab_arena/environments/hubble_lab_visuals.usda"),
            object_min_z=-0.05,
        )

        assets = [background, table, light, ground_plane, pick_up_object, destination, lab_visuals]

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            lock_waist=True,
        )
        if embodiment.action_config.g1_action.wbc_version == "agile":
            # Match the tested deployment export; retain the existing armature and upper-body gains.
            embodiment.action_config.g1_action.wbc_version = "agile_deployment"
            actuators = embodiment.scene_config.robot.actuators
            actuators["legs"].stiffness = {".*_hip_.*_joint": 100.0, ".*_knee_joint": 200.0}
            actuators["legs"].damping = {".*_hip_.*_joint": 2.5, ".*_knee_joint": 5.0}
            actuators["feet"].stiffness = 20.0
            actuators["feet"].damping = {".*_ankle_pitch_joint": 0.2, ".*_ankle_roll_joint": 0.1}
        embodiment.scene_config.robot.spawn.usd_path = str(Path(__file__).resolve().parents[1] / "isaaclab_arena/embodiments/g1/assets/g1_sim2real.usd")
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

        table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
        table_top_z = float(table.get_world_bounding_box().top_surface_z.item())
        task = self._make_task(pick_up_object, destination, background, table)

        # The near table edge is 7.5 inches ahead of the rearmost heel in the recorded posture.
        # Lowest-motion tested reset: 0.730 m and bag roll/pitch (0.30079/0.91772 deg), yaw -90 deg.
        embodiment.set_initial_pose(Pose(
            position_xyz=(0.0, 0.507906193, 0.73),
            rotation_xyzw=(0.0075188823154495165, 0.0038068535141696338, -0.7070965336262947, 0.7070668047707561),
        ))

        def env_cfg_callback(env_cfg):
            # Render once per control step (200 Hz physics / 4 = 50 Hz).
            env_cfg.sim.render_interval = 4
            # Datagen only: let AGILE and contact dynamics settle before capturing the initial state.
            env_cfg.datagen_settle_time_s = 2.0
            # Reference AutoData final qualification: initial success + 20 follow-through
            # frames, retrying within 5 seconds. The live termination remains instantaneous.
            env_cfg.final_success_hold_frames = 20
            env_cfg.final_success_settle_time_s = 5.0
            # Refresh camera observations after joint/object teleports on reset.
            env_cfg.num_rerenders_on_reset = 2
            if args_cli.enable_cameras:
                # rosbag fit: wrist motion + plate shape, with a held-out wrist pose.
                # Episode camera DR is composed around this nominal ROS optical pose.
                env_cfg.scene.robot_head_cam.offset.pos = (0.06666881967709855, -0.0028225634532942946, 0.4686949455532736,)
                env_cfg.scene.robot_head_cam.offset.rot = (0.6786696624183769, -0.6610453565935731, 0.1718243625557676, -0.27000539674478047,)
                env_cfg.scene.robot_head_cam.offset.convention = "ros"
                from isaaclab_arena_environments.hubble_depth import configure_depth_camera

                configure_depth_camera(env_cfg)
            env_cfg = _apply_legacy_datagen_name_override(
                env_cfg,
                pick_up_object_name=pick_up_object.name,
                destination_name=destination.name,
            )
            from isaaclab.managers import EventTermCfg

            if args_cli.enable_cameras:
                env_cfg.events.calibrate_ego_camera = EventTermCfg(func=_configure_ego_intrinsics, mode="startup")

            # Author static USD topology before PhysX/Fabric initializes, not during live resets.
            env_cfg.events.align_table_visuals = EventTermCfg(
                func=_align_table_visuals,
                mode="prestartup",
                # The lab's cloth/clamps were authored for the previous 0.816753149 m tabletop.
                params={"height_offset_m": table_top_z - 0.8167531490325928},
            )
            env_cfg.events.apply_calibrated_materials = EventTermCfg(func=_apply_calibrated_materials, mode="prestartup")

            env_cfg.viewer.eye = (-1.35, -1.30, 2.275)
            env_cfg.viewer.lookat = (0.15, 0.20, 0.775)
            env_cfg.viewer.origin_type = "env"
            # Keep the interactive free view separate from the calibrated tripod recording.
            env_cfg.events.recording_camera = EventTermCfg(
                func=_configure_recording_camera, mode="prestartup"
            )
            if getattr(args_cli, "minimal_teleop_scene", False):
                env_cfg.events.minimal_teleop_scene = EventTermCfg(
                    func=_configure_minimal_teleop_scene, mode="prestartup"
                )
                env_cfg.sim.render.carb_settings["/rtx/sceneDb/ambientLightIntensity"] = 3.0
                # Retain the cloth's sheen; texture-only shading leaves the dark cloth nearly black.
                env_cfg.sim.render.carb_settings["/rtx/minimal/mode"] = 3
            deployment_pd_profile = getattr(args_cli, "deployment_pd_profile", "current")
            if deployment_pd_profile != "current":
                from agentic_design.lane29.residual_recipe_20261014.data.diagnostic_profiles import apply_profile

                env_cfg.deployment_pd_profile = apply_profile(env_cfg, deployment_pd_profile)
            hand_profile = getattr(args_cli, "deployment_hand_target_profile", "current")
            if hand_profile == "episode_alternating_abrupt_ramp075":
                assert deployment_pd_profile == "current", "Mixed hand execution requires current actuator gains"
                hands = env_cfg.scene.robot.actuators["hands"]
                assert hands.stiffness == 4.0 and hands.damping == 0.5, "Mixed hand execution requires hand PD4/0.5"
            if hand_profile != "current":
                assert hasattr(env_cfg.actions.g1_action, "hand_target_profile")
                env_cfg.actions.g1_action.hand_target_profile = hand_profile
            arm_control_mode = getattr(args_cli, "arm_control_mode", "legacy_action_step")
            rnea_urdf_sha256 = getattr(args_cli, "rnea_urdf_sha256", "")
            assert arm_control_mode in ("legacy_action_step", "physics_step_rnea")
            if arm_control_mode == "physics_step_rnea":
                from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_cumotion_action_cfg import (
                    G1DecoupledWBCCuMotionActionCfg,
                )

                assert isinstance(env_cfg.actions.g1_action, G1DecoupledWBCCuMotionActionCfg)
                assert type(rnea_urdf_sha256) is str and len(rnea_urdf_sha256) == 64
                assert all(c in "0123456789abcdef" for c in rnea_urdf_sha256)
                env_cfg.actions.g1_action.arm_control_mode = arm_control_mode
                env_cfg.actions.g1_action.rnea_urdf_sha256 = rnea_urdf_sha256
            else:
                assert rnea_urdf_sha256 == "", "URDF pin requires explicit physics_step_rnea mode"
            return env_cfg

        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )

        return isaaclab_arena_environment

    def _make_destination(self, args_cli: argparse.Namespace) -> Asset:
        return self.asset_registry.get_asset_by_name(args_cli.destination)(scale=(0.7618191, 0.7583220, 0.5307171))

    def _make_task(self, pick_up_object: Asset, destination: Asset, background: Asset, table: Asset) -> TaskBase:
        from isaaclab_arena.tasks.hubble_pick_and_place import HubblePickAndPlaceMimicEnvCfg, HubblePickAndPlaceTask
        from isaaclab_arena.utils.pose import PoseRange

        table_top_z = float(table.get_world_bounding_box().top_surface_z.item())
        # Uniform object-root XY samples; keep each object's bottom 2 mm above the table.
        for obj, xy_min, xy_max in (
            (pick_up_object, (0.09, 0.15), (0.22, 0.30)),
            (destination, (-0.12, 0.185), (-0.08, 0.225)),
        ):
            z = float(table_top_z + 0.002 - obj.get_bounding_box().min_point[0, 2])
            obj.set_initial_pose(PoseRange(position_xyz_min=(*xy_min, z), position_xyz_max=(*xy_max, z)))

        def _build_hubble_pick_and_place_mimic_cfg(arm_mode):
            return HubblePickAndPlaceMimicEnvCfg(
                pick_up_object_name=pick_up_object.name,
                destination_location_name=destination.name,
                arm_mode=arm_mode,
            )

        return HubblePickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination,
            background_scene=background,
            episode_length_s=6.0,
            task_description=(
                f"Pick up the {pick_up_object.name.replace('_', ' ')} from the table and place it onto the "
                f"{destination.name.replace('_', ' ')}."
            ),
            # Reference-derived placement with bounds for this lane's verified apple/plate/table.
            support_surface_z=float(table.get_world_bounding_box().top_surface_z.item()),
            force_threshold=0.5,
            velocity_threshold=0.10,
            mimic_env_cfg_factory=_build_hubble_pick_and_place_mimic_cfg,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="apple_01_objaverse_robolab")
        parser.add_argument("--destination", type=str, default="clay_plates_hot3d_robolab")
        parser.add_argument("--arm_control_mode", choices=("legacy_action_step", "physics_step_rnea"),
                            default="legacy_action_step", help="Opt-in 5 ms arm RNEA; no complete hardware-parity claim.")
        parser.add_argument("--rnea_urdf_sha256", type=str, default="",
                            help="Required independent URDF identity for physics_step_rnea.")
        parser.add_argument(
            "--minimal_teleop_scene", action="store_true",
            help="Temporarily disable lab props and robot cast shadows, with matte robot shading and frontal lighting.",
        )
        parser.add_argument(
            "--deployment_pd_profile",
            choices=("current", "recorded_arm_pd", "recorded_hand_pd", "recorded_upperbody_pd"),
            default="current",
            help="Opt-in recorded upper-body gain hypothesis; hardware torque equivalence is unvalidated.",
        )
        parser.add_argument(
            "--deployment_hand_target_profile",
            choices=("current", "full_blend_ramp075", "episode_alternating_abrupt_ramp075"), default="current",
            help="Opt-in 0.75 rad/s hand-target ramp at physics dt; full-blend hypothesis, not identified hardware dynamics.",
        )
        # AGILE deployment balance with cuMotion RMPFlow arms; retain the 23-D wrist-pose
        # action interface used by teleop, Mimic, and end-effector policies.
        parser.add_argument("--embodiment", type=str, default="g1_wbc_agile_cumotion")
        parser.add_argument("--teleop_device", type=str, default=None)
