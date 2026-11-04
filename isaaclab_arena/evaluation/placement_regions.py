# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Export default object-reset regions for an idle deployment camera overlay."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera


def project_corners(
    corners: np.ndarray, camera_position: np.ndarray, camera_quaternion_xyzw: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Project environment-frame corners [m] into ROS optical camera pixels [px]."""
    quaternion = np.asarray(camera_quaternion_xyzw, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    optical = (np.asarray(corners) - camera_position) @ rotation
    assert np.isfinite(optical).all() and (optical[:, 2] > 0).all(), "Placement region is behind the camera"
    pixels = optical @ intrinsics.T
    return pixels[:, :2] / pixels[:, 2:3]


def _rendered_camera_pose(camera: Camera, use_fabric: bool) -> tuple[np.ndarray, np.ndarray]:
    """Read the renderer's existing transform; sensor pose caches may precede reset."""
    prim = camera._sensor_prims[0].GetPrim()
    if use_fabric:
        import usdrt

        from isaaclab.sim.utils.stage import get_current_stage_id

        stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
        fabric_prim = stage.GetPrimAtPath(str(prim.GetPath()))
        attribute = usdrt.Rt.Xformable(fabric_prim).GetFabricHierarchyWorldMatrixAttr()
        assert attribute, "Render the default reset scene before exporting placement regions"
        matrix = attribute.Get()
    else:
        from pxr import UsdGeom

        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    assert matrix is not None, "Camera render transform is unavailable"
    rotation = matrix.ExtractRotationQuat()
    x, y, z = rotation.GetImaginary()
    w = rotation.GetReal()
    # Native USD camera axes are OpenGL. ROS optical axes require a local 180-degree X rotation.
    return np.array(matrix.ExtractTranslation()), np.array([w, z, -y, -x])


def collect_placement_regions(env: ManagerBasedRLEnv, environment_name: str) -> dict | None:
    """Capture default reset bounds and the camera pose before any policy actions.

    Call after reset in a default environment, without a domain-randomization profile.
    Bounds come from native PoseRange reset events, not sampled object positions.
    """
    if "robot_head_cam" not in env.scene.sensors:
        return None
    regions = []
    for event in vars(env.cfg.events).values():
        if event is None or getattr(event, "mode", None) != "reset":
            continue
        params = event.params
        if "pose_range" not in params or "asset_cfgs" not in params:
            continue
        bounds = params["pose_range"]
        for asset in params["asset_cfgs"]:
            if asset.name not in env.scene.rigid_objects:
                continue
            low = [float(bounds[axis][0]) for axis in ("x", "y", "z")]
            high = [float(bounds[axis][1]) for axis in ("x", "y", "z")]
            assert low[2] == high[2], "Placement overlay requires a fixed reset height"
            corners = [
                [low[0], low[1], low[2]],
                [high[0], low[1], low[2]],
                [high[0], high[1], low[2]],
                [low[0], high[1], low[2]],
            ]
            label = "Apple" if "apple" in asset.name else "Plate" if "plate" in asset.name else asset.name
            regions.append(
                {
                    "asset_name": asset.name,
                    "label": label,
                    "position_xyz_min_m": low,
                    "position_xyz_max_m": high,
                    "rpy_min_rad": [float(bounds[axis][0]) for axis in ("roll", "pitch", "yaw")],
                    "rpy_max_rad": [float(bounds[axis][1]) for axis in ("roll", "pitch", "yaw")],
                    "corners_xyz_m": corners,
                }
            )
    if not regions:
        return None

    camera = env.scene.sensors["robot_head_cam"]
    data = camera.data
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    position, quaternion = _rendered_camera_pose(camera, env.cfg.sim.use_fabric)
    position -= origin
    intrinsics = data.intrinsic_matrices.torch[0].detach().cpu().numpy()
    robot = env.scene["robot"]
    root = robot.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
    root[:3] -= origin
    for region in regions:
        region["corners_uv_px"] = project_corners(
            np.array(region["corners_xyz_m"]), position, quaternion, intrinsics
        ).tolist()
    return {
        "schema_version": 1,
        "environment": environment_name,
        "configuration": "default_environment_without_domain_randomization",
        "coordinate_frame": "environment_local",
        "region_semantics": "object_root_position",
        "camera": {
            "name": "robot_head_cam",
            "image_width": camera.cfg.width,
            "image_height": camera.cfg.height,
            "intrinsic_matrix": intrinsics.tolist(),
            "position_xyz_m": position.tolist(),
            "orientation_xyzw_ros": quaternion.tolist(),
            "pose_source": "rendered_camera_transform",
            "projection": "pinhole_ros_optical_x_right_y_down_z_forward",
            "image_mapping": "full_frame_resize",
        },
        "robot_reference": {
            "state": "reset_before_policy_execution",
            "root_pose_xyz_xyzw": root.tolist(),
            "joint_names": list(robot.joint_names),
            "joint_positions_rad": robot.data.joint_pos.torch[0].detach().cpu().tolist(),
        },
        "regions": regions,
    }


def write_placement_regions(metadata: dict | None, export_directory: str | Path) -> None:
    """Save the optional sidecar next to the LEAPP graph YAML and ONNX files."""
    path = Path(export_directory) / "placement_regions.yaml"
    if metadata is None:
        path.unlink(missing_ok=True)
        return
    assert path.parent.is_dir(), "LEAPP must finish creating its export directory first"
    path.write_text(
        "# Display-only placement guide at the recorded reference robot/camera pose.\n"
        "# Coordinates describe object roots, not silhouettes; no DR was applied.\n"
        + yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    print(f"LEAPP_PLACEMENT_REGIONS: {path.resolve()}", flush=True)
