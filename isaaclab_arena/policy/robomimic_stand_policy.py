# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import gymnasium as gym
import torch
from collections import deque
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

import robomimic.utils.file_utils as FileUtils

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

import leapp
from leapp import annotate

# G1 WBC + PINK IK action vector layout (23 dims total):
#   [0]     left gripper
#   [1]     right gripper
#   [2:5]   left eef pos
#   [5:9]   left eef quat
#   [9:12]  right eef pos
#   [12:16] right eef quat
#   [16:19] navigate cmd
#   [19]    base height cmd  (== index -4)
#   [20:23] torso orientation rpy cmd

# Must match the low_dim / rgb obs keys in the robomimic training config.
_LOW_DIM_OBS_KEYS = [
    "left_eef_pos",
    "left_eef_quat",
    "left_wrist_pose_pelvis_frame",
    "right_eef_pos",
    "right_eef_quat",
    "right_wrist_pose_pelvis_frame",
]
_RGB_OBS_KEYS = ["robot_head_cam_rgb"]


@dataclass
class RobomimicStandPolicyCfg(PolicyCfg):
    """Configuration for RobomimicStandPolicy."""

    robomimic_checkpoint: str
    base_height: float = 0.78
    device: str = "cuda:0"


@register_policy
class RobomimicStandPolicy(PolicyBase[RobomimicStandPolicyCfg]):
    """Agile WBC lower body (fixed stand height) + robomimic diffusion policy upper body.

    Lower body: zeros with base_height written to index -4 (navigate and torso commands stay zero).
    Upper body: arm dims [0:16] are overwritten by the diffusion policy output each step.

    Frame stacking is managed internally using the frame_stack value from the checkpoint config.
    """

    name = "robomimic_stand"

    def __init__(self, config: RobomimicStandPolicyCfg):
        """
        Initialize StandPolicy.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)
        self._base_height = config.base_height
        self._device = config.device
        self._checkpoint_path = config.robomimic_checkpoint
        self._obs_history: dict | None = None
        # LEAPP export runs once, on a step where the diffusion policy actually denoises.
        self._exported = False

        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=self._checkpoint_path,
            device=self._device,
            verbose=True,
        )

        try:
            self._frame_stack = max(1, int(self._policy.policy.global_config.train.frame_stack))
        except AttributeError:
            self._frame_stack = 1

    def _env_graph_name(self, env: gym.Env) -> str:
        """Gym spec id of the environment, used as the LEAPP graph and node name so the export
        carries the source environment's identity (mirrors upstream ``ensure_env_spec_id``).
        Falls back to the policy name when no spec id is available."""
        spec = getattr(env, "spec", None) or getattr(getattr(env, "unwrapped", env), "spec", None)
        spec_id = getattr(spec, "id", None)
        if not spec_id:
            return self.name
        # strip the gym namespace and avoid path separators (leapp.start treats name as a path)
        return spec_id.split(":")[-1].replace("/", "_")

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        env_device = torch.device(env.unwrapped.device)

        # ---- Runtime glue: select keys + frame-stack (NOT traced) ----
        # Frame-stacking is cross-call state (the deque holds frames from prior steps), so it
        # cannot be a pure function of this step's inputs and must stay outside the traced graph.
        # The stacked obs is annotated as the graph input below; deployment supplies the history.
        policy_obs = observation.get("policy", observation)
        camera_obs = observation.get("camera_obs", {})

        obs = {}
        for k in _LOW_DIM_OBS_KEYS:
            obs[k] = torch.squeeze(copy.deepcopy(policy_obs[k])).float()
        for k in _RGB_OBS_KEYS:
            obs[k] = torch.squeeze(copy.deepcopy(camera_obs[k]))

        if self._obs_history is None:
            self._obs_history = {
                k: deque([v.unsqueeze(0).clone() for _ in range(self._frame_stack)], maxlen=self._frame_stack)
                for k, v in obs.items()
            }
        else:
            for k, v in obs.items():
                self._obs_history[k].append(v.unsqueeze(0).clone())

        stacked_obs = {k: torch.cat(list(q), dim=0) for k, q in self._obs_history.items()}

        # Export single-shot, and only on a step where the diffusion policy actually denoises
        # (action queue empty). On queue-non-empty steps get_action returns a cached action from
        # a prior step that isn't connected to this step's annotated input -> "non-traced tensors".
        exporting = not self._exported and len(self._policy.policy.action_queue) == 0
        if exporting:
            graph_name = self._env_graph_name(env)
            leapp.start(name=graph_name)
            stacked_obs = dict(zip(stacked_obs.keys(), annotate.input_tensors(graph_name, stacked_obs)))

        model_obs = self._policy._prepare_observation(stacked_obs)
        dp_actions = self._policy.policy.get_action(obs_dict=model_obs, goal_dict=None).to(device=env_device)

        # snap continuous gripper output back to the binary {open, closed} values the
        # WBC controller expects (it treats any non-zero hand_state as closed)
        grippers = dp_actions[:, :2]
        grippers = torch.where(
            grippers < -0.25,
            torch.full_like(grippers, -0.5),
            torch.full_like(grippers, 0.0),
        )

        # Assemble the 23-dim action functionally (torch.cat, no in-place setitem) so the
        # traced graph is exportable to ONNX/TorchScript. Layout: arm[0:16] from the policy
        # (grippers binarized) + lower-body stand commands (navigate=0, base_height, torso=0).
        arm = torch.cat([grippers, dp_actions[:, 2:16]], dim=1)
        navigate_cmd = torch.zeros_like(dp_actions[:, :3])
        base_height_cmd = torch.full_like(dp_actions[:, :1], self._base_height)
        torso_rpy_cmd = torch.zeros_like(dp_actions[:, :3])
        action = torch.cat([arm, navigate_cmd, base_height_cmd, torso_rpy_cmd], dim=1)

        if exporting:
            annotate.output_tensors(graph_name, {"action": action}, export_with="onnx")
            leapp.stop()
            # Relax atol: validation runs the ONNX graph on CPU (ORT) vs the CUDA/PyTorch reference,
            # which drifts ~1e-4 across the deep UNet x denoising steps -- behaviorally negligible.
            leapp.compile_graph(atol=1e-3)
            self._exported = True

        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._policy is not None:
            self._policy.start_episode()
        self._obs_history = None
