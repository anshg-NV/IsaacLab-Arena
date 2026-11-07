# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Joint-space variant of the robomimic stand policy.

Same diffusion policy, same frame stacking, same chunked rollout as
``RobomimicStandPolicy`` -- only the action space differs. The network predicts upper-body
arm joint angles instead of end-effector poses, so the assembled action flows through the
decoupled-WBC *joint* action term rather than the PINK-IK one.
"""

import torch
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.robomimic_stand_policy import RobomimicStandPolicy, RobomimicStandPolicyCfg
from isaaclab_arena_g1.g1_env.g1_arm_joints import (
    G1_JOINT_ACTION_DIM,
    G1_JOINT_ACTION_ELEMENT_NAMES,
    G1_LEFT_ARM_JOINT_NAMES,
    G1_RIGHT_ARM_JOINT_NAMES,
    joint_action_to_wbc,
)


@dataclass
class RobomimicStandJointPolicyCfg(RobomimicStandPolicyCfg):
    """Configuration for RobomimicStandJointPolicy."""


@register_policy
class RobomimicStandJointPolicy(RobomimicStandPolicy, PolicyBase[RobomimicStandJointPolicyCfg]):
    """WBC lower body (fixed stand height) + robomimic diffusion policy upper-body joint angles.

    The policy emits a 16-dim action -- two gripper scalars followed by the 14 arm joint
    angles in G1_ARM_JOINT_NAMES order -- which is expanded here into the 50-dim action the
    decoupled-WBC joint action term consumes: 43 full-body joint targets followed by the
    navigate / base-height / torso commands. Only the arm and finger slots of the joint
    block are filled; the WBC lower-body policy overwrites the leg slots and holds the
    waist, so leaving those at zero is what the action term already expects.
    """

    # ``PolicyBase[...]`` is re-declared even though RobomimicStandPolicy already provides it:
    # the policy registry reads the *direct* bases to find the config type, so a subclass that
    # only inherits another policy has no PolicyBase[Cfg] for it to resolve.
    name = "robomimic_stand_joint"

    action_space = "joint"
    action_kind = "command/g1_joint_action"
    action_element_names = list(G1_JOINT_ACTION_ELEMENT_NAMES)
    obs_element_names = {
        "left_arm_joint_pos": list(G1_LEFT_ARM_JOINT_NAMES),
        "right_arm_joint_pos": list(G1_RIGHT_ARM_JOINT_NAMES),
    }

    def __init__(self, config: RobomimicStandJointPolicyCfg):
        """Initialize the joint-space stand policy.

        Args:
            config: Typed policy configuration.
        """
        assert not config.record_residual_metrics, "Wrist-translation metrics do not apply to joint actions"
        assert config.feedback_contract is None, "Causal20 wrist feedback requires an end-effector policy"
        super().__init__(config)
        self._joint_names: list[str] | None = None
        self.raw_action: torch.Tensor | None = None
        assert (
            self._policy.policy.ac_dim == G1_JOINT_ACTION_DIM
        ), f"checkpoint predicts {self._policy.policy.ac_dim}-dim actions, expected {G1_JOINT_ACTION_DIM}"
        if self._residual is not None:
            assert (
                self._residual.config["horizon"] == self._policy.policy.algo_config.horizon.action_horizon
            ), "Joint residual/base action horizons differ"
        self._action_scale = self._action_offset = None
        stats = self._policy.action_normalization_stats
        if stats is not None:
            assert list(self._policy.policy.global_config.train["action_keys"]) == ["actions"]
            self._action_scale = torch.as_tensor(stats["actions"]["scale"], device=self._device, dtype=torch.float32)
            self._action_offset = torch.as_tensor(stats["actions"]["offset"], device=self._device, dtype=torch.float32)
            assert self._action_scale.shape == self._action_offset.shape == (1, G1_JOINT_ACTION_DIM)
            if getattr(self._policy.policy, "rel_joint_mode", False):
                assert torch.all(self._action_scale == 1) and torch.all(
                    self._action_offset == 0
                ), "Joint-relative checkpoints must train on unnormalized action labels"

    def _ensure_joint_slots(self, env) -> None:
        """Validate the simulator action layout and cache its articulation order."""
        assert env.unwrapped.action_manager.total_action_dim == 50, "Use a G1 WBC joint embodiment"
        if self._joint_names is None:
            self._joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
            assert len(self._joint_names) == len(set(self._joint_names)) == 43

    def _prepare_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        """Restore physical units once, before caching, residual correction, or export."""
        assert actions.shape[-1] == G1_JOINT_ACTION_DIM
        if self._action_scale is not None:
            actions = actions * self._action_scale.to(actions.device) + self._action_offset.to(actions.device)
        return actions

    def _assemble_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Expand corrected arm angles and continuous grippers for the simulator."""
        self.raw_action = actions
        return joint_action_to_wbc(actions, self._joint_names, self._base_height)

    def _export_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        """Export two continuous gripper scalars and fourteen absolute arm angles."""
        assert actions.shape[-1] == G1_JOINT_ACTION_DIM
        return actions

    def get_action(self, env, observation) -> torch.Tensor:
        """Resolve articulation joint slots on the first call, then defer to the base policy."""
        self._ensure_joint_slots(env)
        return super().get_action(env, observation)
