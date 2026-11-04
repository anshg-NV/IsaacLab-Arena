# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Geometry and metadata checks without starting a simulator."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

SPEC = importlib.util.spec_from_file_location(
    "placement_regions", Path(__file__).parents[1] / "evaluation/placement_regions.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_project_camera_rotation_translation_and_depth():
    # Camera optical +Z faces environment +X; optical +X faces environment -Z.
    quaternion = np.array([0, np.sqrt(0.5), 0, np.sqrt(0.5)])
    position = np.array([2, 3, 4])
    intrinsics = np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1]])
    points = np.array([[4, 3, 4], [4, 4, 3]])
    np.testing.assert_allclose(
        MODULE.project_corners(points, position, quaternion, intrinsics), [[160, 120], [210, 170]]
    )
    with pytest.raises(AssertionError, match="behind"):
        MODULE.project_corners(np.array([[1, 3, 4]]), position, quaternion, intrinsics)


def test_export_uses_pose_bounds_and_subtracts_parallel_environment_origin(tmp_path, monkeypatch):
    def proxy(values):
        return SimpleNamespace(torch=torch.tensor(values, dtype=torch.float32))

    bounds = {"x": [0, 1], "y": [0, 1], "z": [2, 2], "roll": [0, 0], "pitch": [0, 0], "yaw": [0, 0]}
    event = SimpleNamespace(mode="reset", params={"pose_range": bounds, "asset_cfgs": [SimpleNamespace(name="apple")]})
    camera = SimpleNamespace(
        cfg=SimpleNamespace(width=320, height=240),
        data=SimpleNamespace(
            pos_w=proxy([[0, 0, 0]]),  # Stale sensor pose must not be used for projection.
            quat_w_ros=proxy([[0, 0, 0, 1]]),
            intrinsic_matrices=proxy([[[100, 0, 160], [0, 100, 120], [0, 0, 1]]]),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_rendered_camera_pose",
        lambda camera, use_fabric: (np.array([10.0, 20.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
    )
    robot = SimpleNamespace(
        joint_names=["arm"],
        data=SimpleNamespace(root_pose_w=proxy([[10, 20, 0, 0, 0, 0, 1]]), joint_pos=proxy([[0.25]])),
    )

    class Scene(dict):
        sensors = {"robot_head_cam": camera}
        rigid_objects = {"apple": object()}
        env_origins = torch.tensor([[10, 20, 0]])

    env = SimpleNamespace(
        cfg=SimpleNamespace(events=SimpleNamespace(apple=event), sim=SimpleNamespace(use_fabric=True)),
        scene=Scene(robot=robot),
    )
    metadata = MODULE.collect_placement_regions(env, "test_environment")
    assert metadata["robot_reference"]["root_pose_xyz_xyzw"] == [0, 0, 0, 0, 0, 0, 1]
    region = metadata["regions"][0]
    assert region["position_xyz_max_m"] == [1, 1, 2]
    np.testing.assert_allclose(region["corners_uv_px"], [[160, 120], [210, 120], [210, 170], [160, 170]])
    MODULE.write_placement_regions(metadata, tmp_path)
    assert yaml.safe_load((tmp_path / "placement_regions.yaml").read_text()) == metadata
    MODULE.write_placement_regions(None, tmp_path)
    assert not (tmp_path / "placement_regions.yaml").exists()
