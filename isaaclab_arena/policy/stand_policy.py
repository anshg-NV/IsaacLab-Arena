# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import gymnasium as gym
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class StandPolicyArgs:
    """
    Configuration dataclass for StandPolicy.
    """

    base_height: float = 0.78

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "StandPolicyArgs":
        return cls(base_height=getattr(args, "base_height", 0.78))


@register_policy
class StandPolicy(PolicyBase):

    name = "stand"
    # enable from_dict() from policy_base.PolicyBase
    config_class = StandPolicyArgs

    def __init__(self, config: StandPolicyArgs):
        """
        Initialize StandPolicy.

        Args:
            config: StandPolicyArgs configuration dataclass (optional, not used)
        """
        super().__init__(config)
        self._base_height = config.base_height

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        action = torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))
        action[..., -4] = self._base_height
        return action

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Args:
            parser: The argument parser to add arguments to

        Returns:
            The updated argument parser (unchanged)
        """
        parser.add_argument(
            "--base_height",
            type=float,
            default=0.78,
            help="Target pelvis height in meters written into the WBC height command channel.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "StandPolicy":
        """
        Create a StandPolicy instance from parsed CLI arguments.

        Path: CLI args → ConfigDataclass → init cls

        Args:
            args: Parsed command line arguments

        Returns:
            StandPolicy instance
        """
        config = StandPolicyArgs.from_cli_args(args)
        return StandPolicy(config)
