# SPDX-FileCopyrightText: 2026 The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit observation contracts for visual and blind wrist residuals."""

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


def residual_features(algo, obs: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    """Return chunk-cached features; blind mode never reads RGB or an image encoder.

    Blind features concatenate the six prepared wrist observations in the order above,
    within each of the two chronological history frames. The frozen vision base's
    action chunk is a separate input, so the combined policy is not camera-free.
    """
    assert mode in RESIDUAL_FEATURE_DIMS, f"Unsupported residual input mode: {mode}"
    if mode == "vision":
        from robomimic.models.residual_policy_nets import encode_residual_observation

        return encode_residual_observation(algo, obs)
    values = []
    for key, shape in BLIND_OBSERVATION_SHAPES.items():
        value = obs[key]
        assert value.shape[1] == 2 and tuple(value.shape[2:]) == shape, (key, value.shape)
        values.append(value.flatten(start_dim=2))
    return torch.cat(values, dim=-1).flatten(start_dim=1)


def residual_mode_from_config(config: dict) -> str:
    """Identify the supported compact residual input contract without runtime overrides."""
    assert config["horizon"] == 6 and config["max_translation"] == 0.04
    modes = [mode for mode, size in RESIDUAL_FEATURE_DIMS.items() if size == config["feature_dim"]]
    assert len(modes) == 1, f"Unsupported residual feature dimension: {config['feature_dim']}"
    return modes[0]
