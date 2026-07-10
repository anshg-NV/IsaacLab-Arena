# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class StandPolicyCfg(PolicyCfg):
    """Configuration for StandPolicy."""

    base_height: float = 0.78


@register_policy
class StandPolicy(PolicyBase[StandPolicyCfg]):

    name = "stand"

    def __init__(self, config: StandPolicyCfg):
        """
        Initialize StandPolicy.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)
        self._base_height = config.base_height

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        action = torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))
        action[..., -4] = self._base_height
        return action
