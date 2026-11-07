# SPDX-FileCopyrightText: 2026 The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0

"""Observation contracts for visual and blind EEF or joint residuals."""

import torch


BLIND_OBSERVATION_SHAPES = {
    "left_eef_pos": (3,),
    "left_eef_quat": (4,),
    "left_wrist_pose_pelvis_frame": (4, 4),
    "right_eef_pos": (3,),
    "right_eef_quat": (4,),
    "right_wrist_pose_pelvis_frame": (4, 4),
}
RESIDUAL_FEATURE_DIMS = {"vision": 220, "blind": 92}


def residual_features(algo, obs: dict[str, torch.Tensor], mode: str, *, action_space: str = "eef") -> torch.Tensor:
    """Return chunk-cached features; blind mode never reads RGB or an image encoder.

    Blind features concatenate prepared wrist observations (EEF) or arm angles (joint)
    within each chronological history frame. The frozen vision base's
    action chunk is a separate input, so the combined policy is not camera-free.
    """
    assert mode in RESIDUAL_FEATURE_DIMS, f"Unsupported residual input mode: {mode}"
    assert action_space in ("eef", "joint")
    if mode == "vision":
        from robomimic.models.residual_policy_nets import encode_residual_observation

        return encode_residual_observation(algo, obs)
    values = []
    shapes = (
        {"left_arm_joint_pos": (7,), "right_arm_joint_pos": (7,)}
        if action_space == "joint"
        else BLIND_OBSERVATION_SHAPES
    )
    for key, shape in shapes.items():
        value = obs[key]
        assert tuple(value.shape[2:]) == shape, (key, value.shape)
        if action_space == "eef":
            assert value.shape[1] == 2, (key, value.shape)
        values.append(value.flatten(start_dim=2))
    return torch.cat(values, dim=-1).flatten(start_dim=1)


def residual_mode_from_config(config: dict) -> str:
    """Identify the supported compact residual input contract without runtime overrides."""
    assert config["horizon"] == 6 and config["max_translation"] == 0.04
    modes = [mode for mode, size in RESIDUAL_FEATURE_DIMS.items() if size == config["feature_dim"]]
    assert len(modes) == 1, f"Unsupported residual feature dimension: {config['feature_dim']}"
    return modes[0]
