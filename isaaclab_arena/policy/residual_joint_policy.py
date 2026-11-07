# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded arm-angle and gripper corrections for a frozen 16-D joint policy."""

import torch
from torch import nn

from robomimic.models.residual_policy_nets import ResidualWristPolicy

MAX_JOINT_DELTA = 0.1
MAX_GRIPPER_DELTA = 0.5


def joint_residual_network(input_dim: int, output_dim: int) -> nn.Sequential:
    """Build the shared compact/RSL actor or critic architecture."""
    network = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, output_dim),
    )
    if output_dim == 16:
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
    return network


def apply_joint_residual(
    base: torch.Tensor,
    raw: torch.Tensor,
    max_joint_delta: float = MAX_JOINT_DELTA,
    max_gripper_delta: float = MAX_GRIPPER_DELTA,
) -> torch.Tensor:
    """Correct one action or a chunk, retaining continuous grippers until consumption."""
    assert base.shape[-1] == raw.shape[-1] == 16
    delta = torch.cat((max_gripper_delta * raw[..., :2].tanh(), max_joint_delta * raw[..., 2:].tanh()), dim=-1)
    if base.ndim == raw.ndim + 1:
        delta = delta.unsqueeze(-2)
    return base + delta


class ResidualJointPolicy(ResidualWristPolicy):
    """Predict two gripper and fourteen joint corrections shared over each action chunk.

    Uses the existing compact checkpoint, distribution, and frozen-base identity
    handling, with a distinct joint-action configuration and 16-output actor.
    """

    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        input_mode: str = "vision",
        max_joint_delta: float = MAX_JOINT_DELTA,
        max_gripper_delta: float = MAX_GRIPPER_DELTA,
    ):
        nn.Module.__init__(self)
        assert feature_dim > 0 and horizon > 0
        assert input_mode in ("vision", "blind")
        assert 0 < max_joint_delta <= 0.5 and 0 < max_gripper_delta <= 1.0
        self.config = dict(
            feature_dim=feature_dim,
            horizon=horizon,
            input_mode=input_mode,
            max_joint_delta=max_joint_delta,
            max_gripper_delta=max_gripper_delta,
        )
        input_dim = feature_dim + horizon * 16
        self.actor = joint_residual_network(input_dim, 16)
        self.critic = joint_residual_network(input_dim, 1)
        self.log_std = nn.Parameter(torch.full((16,), -1.0))

    def combine(self, base: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Apply the same bounded correction used by native residual training."""
        assert base.shape[-2:] == (self.config["horizon"], 16)
        return apply_joint_residual(base, raw, self.config["max_joint_delta"], self.config["max_gripper_delta"])
