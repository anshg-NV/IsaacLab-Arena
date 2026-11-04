# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import os
import torch
import yaml
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.config.configs import (
    AgileConfig,
    AgileDeploymentConfig,
    HomieV2Config,
)
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    G1_NUM_JOINTS,
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.wbc_policy_factory import get_wbc_policy
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.run_policy import (
    convert_sim_joint_to_wbc_joint,
    postprocess_actions,
    prepare_observations,
)
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 import instantiate_g1_robot_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_joint_action_cfg import G1DecoupledWBCJointActionCfg


_HAND_TARGET_JOINT_NAMES = tuple(
    f"{side}_hand_{part}_joint" for side in ("left", "right")
    for part in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
)


class G1DecoupledWBCJointAction(ActionTerm):
    """Action term for the G1 decoupled WBC policy. Upper body direct joint position control, lower body RL-based policy."""

    cfg: G1DecoupledWBCJointActionCfg

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: G1DecoupledWBCJointActionCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._processed_actions = torch.zeros([self.num_envs, self._num_joints], device=self.device)
        self._initialize_hand_target_profile()

        self._wbc_version = self.cfg.wbc_version

        if self._wbc_version == "homie_v2":
            wbc_config = HomieV2Config()
        elif self._wbc_version == "agile":
            wbc_config = AgileConfig()
        elif self._wbc_version == "agile_deployment":
            wbc_config = AgileDeploymentConfig()
        else:
            raise ValueError(f"Invalid WBC version: {self._wbc_version}")

        waist_location = "lower_and_upper_body" if wbc_config.enable_waist else "lower_body"
        self.robot_model = instantiate_g1_robot_model(waist_location=waist_location)

        self.wbc_policy = get_wbc_policy("g1", self.robot_model, wbc_config, self.num_envs)

        self._wbc_goal = {
            # lin_vel_cmd_x, lin_vel_cmd_y, ang_vel_cmd
            "navigate_cmd": np.tile(np.array([[0.0, 0.0, 0.0]]), (self.num_envs, 1)),
            # Pelvis standing height [m], matching the initial root and stand policies.
            "base_height_command": np.tile(np.array([0.72]), (self.num_envs, 1)),
            # toggle_policy_action: 0 for disable policy, 1 for enable policy
            "toggle_policy_action": np.tile(np.array([0]), (self.num_envs, 1)),
            # roll pitch yaw command
            "torso_orientation_rpy_cmd": np.tile(np.array([[0.0, 0.0, 0.0]]), (self.num_envs, 1)),
        }
        wbc_g1_joints_order_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config/loco_manip_g1_joints_order_43dof.yaml",
        )
        try:
            with open(wbc_g1_joints_order_path) as f:
                self.wbc_g1_joints_order = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {wbc_g1_joints_order_path}")

    # Properties.
    # """
    @property
    def num_joints(self) -> int:
        """Get the number of joints."""
        return G1_NUM_JOINTS

    @property
    def navigate_cmd_dim(self) -> int:
        """Dimension of navigation command."""
        return NUM_NAVIGATE_CMD

    @property
    def base_height_cmd_dim(self) -> int:
        """Dimension of base height command."""
        return NUM_BASE_HEIGHT_CMD

    @property
    def torso_orientation_rpy_cmd_dim(self) -> int:
        """Dimension of torso orientation command."""
        return NUM_TORSO_ORIENTATION_RPY_CMD

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        return G1_NUM_JOINTS + NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def get_wbc_version(self):
        return self._wbc_version

    @property
    def get_wbc_policy(self):
        return self.wbc_policy

    @property
    def get_wbc_goal(self):
        return self._wbc_goal

    def set_wbc_goal(
        self, navigate_cmd: torch.Tensor, base_height_cmd: torch.Tensor, torso_orientation_rpy_cmd: torch.Tensor = None
    ):
        self._wbc_goal["navigate_cmd"] = navigate_cmd.cpu().numpy()
        self._wbc_goal["base_height_command"] = base_height_cmd.cpu().numpy()
        if self._wbc_version == "homie_v2" and torso_orientation_rpy_cmd is not None:
            self._wbc_goal["torso_orientation_rpy_cmd"] = torso_orientation_rpy_cmd.cpu().numpy()
        assert self._wbc_goal["navigate_cmd"].shape == (self.num_envs, NUM_NAVIGATE_CMD)
        assert self._wbc_goal["base_height_command"].shape == (self.num_envs, NUM_BASE_HEIGHT_CMD)
        assert self._wbc_goal["torso_orientation_rpy_cmd"].shape == (self.num_envs, NUM_TORSO_ORIENTATION_RPY_CMD)

    def get_navigation_cmd_from_actions(self, actions: torch.Tensor):
        """Get the navigation command from the actions."""
        return actions[
            :,
            -NUM_NAVIGATE_CMD
            - NUM_BASE_HEIGHT_CMD
            - NUM_TORSO_ORIENTATION_RPY_CMD : -NUM_BASE_HEIGHT_CMD
            - NUM_TORSO_ORIENTATION_RPY_CMD,
        ]

    def get_base_height_cmd_from_actions(self, actions: torch.Tensor):
        """Get the base height command from the actions."""
        return actions[:, -NUM_BASE_HEIGHT_CMD - NUM_TORSO_ORIENTATION_RPY_CMD : -NUM_TORSO_ORIENTATION_RPY_CMD]

    def get_torso_orientation_rpy_cmd_from_actions(self, actions: torch.Tensor):
        """Get the torso orientation command from the actions."""
        return actions[:, -NUM_TORSO_ORIENTATION_RPY_CMD:]

    # """
    # Operations.
    # """

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """

        # Store the raw actions
        self._raw_actions[:] = actions[:, : self.action_dim]

        # Make a copy of actions before modifying so that raw actions are not modified
        actions_clone = actions.clone()
        # NOTE (xinjie.yao, 9.22.2025): for multi-env, wbc policy supports multi-env inference,
        # we expect actions containing multiple sets of commands for each env
        """
        **************************************************
        WBC closedloop
        **************************************************
        """
        # extract navigate_cmd  base_height_cmd, and torso_orientation_rpy_cmd from actions
        navigate_cmd = self.get_navigation_cmd_from_actions(actions_clone)
        base_height_cmd = self.get_base_height_cmd_from_actions(actions_clone)
        torso_orientation_rpy_cmd = self.get_torso_orientation_rpy_cmd_from_actions(actions_clone)

        self.set_wbc_goal(navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        """
        **************************************************
        Prepare WBC policy input
        **************************************************
        """
        wbc_obs = prepare_observations(self.num_envs, self._asset.data, self.wbc_g1_joints_order)
        sim_target_full_body_joints = actions_clone[:, : self._num_joints]
        wbc_target_full_body_joints = convert_sim_joint_to_wbc_joint(
            sim_target_full_body_joints, self._asset.data.joint_names, self.wbc_g1_joints_order
        )
        wbc_target_upper_body_joints = wbc_target_full_body_joints[
            :, self.robot_model.get_joint_group_indices("upper_body")
        ]

        self.wbc_policy.set_observation(wbc_obs)

        wbc_action = self.wbc_policy.get_action(wbc_target_upper_body_joints)
        self._processed_actions = postprocess_actions(
            wbc_action, self._asset.data, self.wbc_g1_joints_order, self.device
        )

    def _initialize_hand_target_profile(self) -> None:
        self._hand_target_profile = getattr(self.cfg, "hand_target_profile", "current")
        assert self._hand_target_profile in ("current", "full_blend_ramp075", "episode_alternating_abrupt_ramp075")
        self._hand_target_reference = None
        if self._hand_target_profile == "current":
            return
        assert not self._env._physics_handles_decimation, "Hand ramp requires one apply per physics step"
        assert math.isfinite(self._env.physics_dt) and self._env.physics_dt > 0
        names = list(self._asset.data.joint_names)
        assert len(set(names)) == len(names), "Duplicate articulation joint names"
        # WBC postprocess produces full articulation-order targets; do not silently support a reordered subset.
        assert self._num_joints == len(names) and list(self._joint_names) == names, (
            "Hand ramp requires full articulation-order action targets"
        )
        assert all(name in names for name in _HAND_TARGET_JOINT_NAMES), "Missing hand joint"
        self._hand_target_indices = [names.index(name) for name in _HAND_TARGET_JOINT_NAMES]
        self._hand_target_reference = self._processed_actions.new_empty((self.num_envs, 14))
        self._hand_target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hand_target_pending_reset = True
        if self._hand_target_profile == "episode_alternating_abrupt_ramp075":
            self._hand_target_episode_ramp = torch.arange(self.num_envs, device=self.device) % 2 == 1
            self._hand_target_episode_used = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._hand_target_used_reset_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def hand_target_contract(self) -> dict:
        """Describe the actual target filter, independently of the actuator-gain profile."""
        if self._hand_target_profile == "current":
            return {"profile": "current"}
        contract = {
            "profile": self._hand_target_profile, "rate_rad_s": 0.75,
            "physics_dt_s": self._env.physics_dt, "joint_names": list(_HAND_TARGET_JOINT_NAMES),
            "initialization": "measured simulated joint q on first apply after selected-env reset",
            "full_blend": 1.0,
            "limitation": "Current deployment execution hypothesis; not identified Oct6 dynamics or a physical velocity bound",
        }
        if self._hand_target_profile == "episode_alternating_abrupt_ramp075":
            contract.update({
                "domains": ["current", "full_blend_ramp075"],
                "schedule": "(environment slot + used-episode reset count) % 2; odd selects ramp",
                "switch": "selected reset after at least one apply; no mid-episode switch or RNG",
                "duplicate_reset": "resets without intervening apply do not advance the domain",
                "used_reset": "includes manually aborted episodes; not proof of natural termination",
                "weighting": "alternating episodes per slot, not equal transition counts",
            })
        return contract

    def hand_target_episode_domain(self, env_id: int) -> dict:
        """Report the active hand domain before reset, without adding it to actor observations."""
        if self._hand_target_profile != "episode_alternating_abrupt_ramp075":
            return {"profile": self._hand_target_profile}
        return {
            "profile": "full_blend_ramp075" if self._hand_target_episode_ramp[env_id].item() else "current",
            "used_episode_reset_count": int(self._hand_target_used_reset_count[env_id].item()),
        }

    def _apply_mixed_hand_targets(self) -> None:
        """Apply the fixed per-episode domain; only ramp rows read lazy reset q or integrate references."""
        ramp = self._hand_target_episode_ramp
        self._hand_target_episode_used[:] = True
        if self._hand_target_pending_reset:
            pending = ~self._hand_target_initialized & ramp
            measured = self._asset.data.joint_pos.torch
            self._hand_target_reference[pending] = measured[pending][:, self._hand_target_indices]
            self._hand_target_initialized[pending] = True
            self._hand_target_pending_reset = False
        desired = self._processed_actions[:, self._hand_target_indices]
        delta = 0.75 * self._env.physics_dt
        self._hand_target_reference[ramp] += (desired[ramp] - self._hand_target_reference[ramp]).clamp(-delta, delta)
        applied = self._processed_actions.clone()
        hand_targets = desired.clone()
        hand_targets[ramp] = self._hand_target_reference[ramp]
        applied[:, self._hand_target_indices] = hand_targets
        self._asset.set_joint_position_target(applied, self._joint_ids)

    def apply_actions(self):
        """Apply WBC targets, optionally slewing only hand references at the physics cadence."""
        if self._hand_target_profile == "episode_alternating_abrupt_ramp075":
            self._apply_mixed_hand_targets()
            return
        if self._hand_target_reference is None:
            self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)
            return
        if self._hand_target_pending_reset:
            pending = ~self._hand_target_initialized
            measured = self._asset.data.joint_pos.torch
            self._hand_target_reference[pending] = measured[pending][:, self._hand_target_indices]
            self._hand_target_initialized[pending] = True
            self._hand_target_pending_reset = False
        desired = self._processed_actions[:, self._hand_target_indices]
        delta = 0.75 * self._env.physics_dt
        self._hand_target_reference += (desired - self._hand_target_reference).clamp(-delta, delta)
        # Keep the held 20 ms desired target intact while applying four independent 5 ms increments.
        applied = self._processed_actions.clone()
        applied[:, self._hand_target_indices] = self._hand_target_reference
        self._asset.set_joint_position_target(applied, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
        if self._hand_target_profile == "episode_alternating_abrupt_ramp075":
            selected = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
            if selected.numel() == 0:
                return
            used = selected[self._hand_target_episode_used[selected]]
            self._hand_target_episode_ramp[used] = ~self._hand_target_episode_ramp[used]
            self._hand_target_used_reset_count[used] += 1
            self._hand_target_episode_used[selected] = False
        if self._hand_target_reference is not None:
            # reset_to installs its supplied state after action.reset; sample q lazily on the next apply.
            self._hand_target_initialized[env_ids] = False
            self._hand_target_pending_reset = True
