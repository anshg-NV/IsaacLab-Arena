# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""D435 depth observation calibrated from the October 21 frame_test recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Native 640 x 480 CameraInfo; resize using OpenCV's pixel-center convention.
DEPTH_INTRINSICS = (389.74505615234375, 389.74505615234375, 320.1612548828125, 242.17100524902344)
DEPTH_SCALE_M = 5.0
DEPTH_NEAR_M = 0.1
DEPTH_OBSERVATION_SCHEMA = "rgbd_aligned_to_color_div5_v2"
DEPTH_TO_COLOR = (
    (0.9999615046334971, 0.0003965565145417532, 0.008765386132035355, 0.01469204457654864),
    (-0.00040423102789719753, 0.9999995365411206, 0.0008737933510446459, 0.00024249750510583755),
    (-0.008765035561193598, -0.0008773029550954178, 0.9999612014929059, 0.0002879710112181979),
)


def normalize_depth(depth_m: torch.Tensor) -> torch.Tensor:
    """Encode optical-axis depth [m] as float32 depth/5; invalid/out-of-range pixels are zero."""
    valid = torch.isfinite(depth_m) & (depth_m >= DEPTH_NEAR_M) & (depth_m <= DEPTH_SCALE_M)
    return torch.where(valid, depth_m / DEPTH_SCALE_M, 0.0).float()


def align_depth_to_rgb(
    depth_m: torch.Tensor,
    depth_intrinsics: torch.Tensor,
    rgb_intrinsics: torch.Tensor,
    output_shape: tuple[int, int],
) -> torch.Tensor:
    """Register physical depth [m] onto RGB pixels using the recorded D435 transform.

    Project each depth pixel's footprint and retain the nearest source depth on
    overlap, following librealsense's align_z_to_other. Values remain the physical
    depth camera's optical Z, as in aligned_depth_to_color; unobserved pixels are zero.
    Intrinsics are batched (N, 3, 3), and input/output depths are (N, H, W, 1).
    """
    batch, height, width, channels = depth_m.shape
    assert channels == 1
    target_height, target_width = output_shape
    z = depth_m[..., 0].reshape(batch, -1)
    valid = torch.isfinite(z) & (z >= DEPTH_NEAR_M) & (z <= DEPTH_SCALE_M)
    safe_z = torch.where(valid, z, 1.0)
    y, x = torch.meshgrid(
        torch.arange(height, device=z.device, dtype=z.dtype),
        torch.arange(width, device=z.device, dtype=z.dtype), indexing="ij",
    )
    transform = torch.as_tensor(DEPTH_TO_COLOR, dtype=z.dtype, device=z.device)
    corners = []
    for delta in (-0.5, 0.5):
        points = torch.stack((
            (x.flatten() + delta - depth_intrinsics[:, 0, 2, None]) / depth_intrinsics[:, 0, 0, None] * safe_z,
            (y.flatten() + delta - depth_intrinsics[:, 1, 2, None]) / depth_intrinsics[:, 1, 1, None] * safe_z,
            safe_z,
        ), dim=-1)
        points = points @ transform[:, :3].T + transform[:, 3]
        projected = points @ rgb_intrinsics.transpose(1, 2)
        pixels = projected[..., :2] / projected[..., 2:]
        # C++ static_cast<int>(pixel + 0.5), including truncation near image borders.
        corners.append((pixels + 0.5).to(torch.long))
        valid &= points[..., 2] > 0
    lo, hi = corners
    valid &= (lo[..., 0] >= 0) & (lo[..., 1] >= 0)
    valid &= (hi[..., 0] < target_width) & (hi[..., 1] < target_height)
    valid &= (hi >= lo).all(dim=-1)
    span = torch.where(valid[..., None], hi - lo, 0).amax(dim=(0, 1)).tolist()
    aligned = torch.full((batch, target_height * target_width), float("inf"), device=z.device, dtype=z.dtype)
    for dy in range(span[1] + 1):
        for dx in range(span[0] + 1):
            px, py = lo[..., 0] + dx, lo[..., 1] + dy
            covered = valid & (px <= hi[..., 0]) & (py <= hi[..., 1])
            indices = torch.where(covered, py * target_width + px, 0)
            values = torch.where(covered, z, float("inf"))
            aligned.scatter_reduce_(1, indices, values, reduce="amin", include_self=True)
    aligned = torch.where(torch.isfinite(aligned), aligned, 0.0)
    return aligned.reshape(batch, target_height, target_width, 1)


def head_depth_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Return RGB-registered depth/5 with shape (environments, height, width, 1)."""
    depth = env.scene["robot_head_depth_cam"]
    rgb = env.scene["robot_head_cam"]
    aligned = align_depth_to_rgb(
        depth.data.output["distance_to_image_plane"],
        depth.data.intrinsic_matrices.torch,
        rgb.data.intrinsic_matrices.torch,
        rgb.image_shape,
    )
    return normalize_depth(aligned)


def configure_depth_camera(env_cfg: Any) -> None:
    """Add the physical depth camera as a rigid child of the calibrated RGB camera."""
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sim import PinholeCameraCfg

    rgb = env_cfg.scene.robot_head_cam
    depth = rgb.copy()
    depth.prim_path = rgb.prim_path + "/RobotHeadDepthCam"
    depth.data_types = ["distance_to_image_plane"]
    # TF and depth_to_color extrinsics agree to 1.4e-8. Convert both optical
    # frames from ROS to USD/OpenGL: T_GL = diag(1,-1,-1) @ T_ROS @ diag(1,-1,-1).
    # Parenting under RGB preserves this baseline during mount-pose DR.
    depth.offset.pos = (0.01469204457654864, -0.00024249750510583755, -0.0002879710112181979)
    depth.offset.rot = (-0.0004377783316151053, -0.004382648021391674, 0.000200198831485082, 0.999990280286204)
    depth.offset.convention = "opengl"
    fx, fy, cx, cy = DEPTH_INTRINSICS
    sx, sy = depth.width / 640.0, depth.height / 480.0
    depth.spawn = PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[fx * sx, 0.0, (cx + 0.5) * sx - 0.5,
                          0.0, fy * sy, (cy + 0.5) * sy - 0.5, 0.0, 0.0, 1.0],
        width=depth.width, height=depth.height, focal_length=15,
        clipping_range=(DEPTH_NEAR_M, DEPTH_SCALE_M),
    )
    env_cfg.scene.robot_head_depth_cam = depth
    env_cfg.observations.camera_obs.robot_head_cam_depth = ObservationTermCfg(func=head_depth_observation)
