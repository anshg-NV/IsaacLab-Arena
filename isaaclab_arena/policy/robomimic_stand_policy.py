# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
from collections import deque
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

import robomimic.utils.file_utils as FileUtils

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


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
    "robot_joint_pos",
]
_RGB_OBS_KEYS = ["robot_head_cam_rgb"]

# Per-element labels only where the layout is unambiguous (xyz positions). Quaternions,
# 4x4 pose matrices, and the image tensor are left unlabeled.
_OBS_ELEMENT_NAMES = {
    "left_eef_pos": ["x", "y", "z"],
    "right_eef_pos": ["x", "y", "z"],
}
# Per-dimension labels for the assembled 23-D action output (matches the layout comment above).
_ACTION_ELEMENT_NAMES = [
    "left_gripper",
    "right_gripper",
    "left_eef_pos_x",
    "left_eef_pos_y",
    "left_eef_pos_z",
    "left_eef_quat_x",
    "left_eef_quat_y",
    "left_eef_quat_z",
    "left_eef_quat_w",
    "right_eef_pos_x",
    "right_eef_pos_y",
    "right_eef_pos_z",
    "right_eef_quat_x",
    "right_eef_quat_y",
    "right_eef_quat_z",
    "right_eef_quat_w",
    "navigate_x",
    "navigate_y",
    "navigate_yaw",
    "base_height",
    "torso_roll",
    "torso_pitch",
    "torso_yaw",
]


@dataclass
class RobomimicStandPolicyCfg(PolicyCfg):
    """Configuration for RobomimicStandPolicy."""

    robomimic_checkpoint: str
    base_height: float = 0.78
    device: str = "cuda:0"
    leapp_export: bool = False


@register_policy
class RobomimicStandPolicy(PolicyBase[RobomimicStandPolicyCfg]):
    """Agile WBC lower body (fixed stand height) + robomimic diffusion policy upper body.

    Lower body: zeros with base_height written to index -4 (navigate and torso commands stay zero).
    Upper body: arm dims [0:16] are overwritten by the diffusion policy output each step.

    Frame stacking is managed internally using the frame_stack value from the checkpoint config.
    Multi-env capable: observations keep their (num_envs, ...) batch dim, the diffusion UNet
    denoises batched, and action chunks are queued per env (robomimic's own queued
    ``get_action`` only serves batch index 0, so it is bypassed in favor of
    ``_get_action_trajectory``).
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
        self._leapp_export = config.leapp_export
        self._obs_history: dict[str, torch.Tensor] | None = None
        self._action_queues: list[deque] | None = None
        self._pending_history_reset: list[int] = []
        # LEAPP export runs once, on a step where the diffusion policy actually denoises.
        self._exported = False

        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=self._checkpoint_path,
            device=self._device,
            verbose=True,
        )

        algo_horizon = self._policy.policy.global_config.algo.get("horizon", {})
        self._frame_stack = max(1, int(algo_horizon.get("observation_horizon", 1)))

        train_frame_stack = int(self._policy.policy.global_config.train.get("frame_stack", self._frame_stack))
        if train_frame_stack != self._frame_stack:
            print(f"[RobomimicStandPolicy] WARNING: checkpoint has train.frame_stack={train_frame_stack} but algo.horizon.observation_horizon={self._frame_stack}. Rollouts feed the last {self._frame_stack} observations, which is not the window this policy was trained on.")

    def _env_graph_name(self, env: gym.Env) -> str:
        """Name the exported LEAPP graph after the environment's registered task id.

        Only affects the export artifact filenames (``<name>.yaml`` / ``.onnx``) and the graph node
        label, so a reader can tell which task an export came from -- deployment reads whatever node
        name the YAML contains, so the exact name is not load-bearing. Falls back to the policy name
        if the env has no spec. Reads ``env.unwrapped.spec`` (a plain attribute) rather than
        ``env.spec`` (a gymnasium wrapper property that deep-copies the USD-laden cfg and warns).
        """
        spec = env.unwrapped.spec
        return spec.id if spec is not None else self.name

    def _assemble_action(self, dp: torch.Tensor) -> torch.Tensor:
        """Turn a diffusion-policy output ``(..., ac_dim)`` into the 23-dim WBC action ``(..., 23)``.

        Layout: arm ``[0:16]`` from the policy (grippers binarized to the {open, closed} values the
        WBC controller expects) + fixed lower-body stand commands (navigate=0, base_height, torso=0).
        Operates on the last dim, so it works for a single action ``(N, 23)`` or a chunk ``(N, Ta, 23)``.
        """
        grippers = dp[..., :2]
        grippers = torch.where(
            grippers < -0.25,
            torch.full_like(grippers, -0.5),
            torch.full_like(grippers, 0.0),
        )
        arm = torch.cat([grippers, dp[..., 2:16]], dim=-1)
        navigate_cmd = torch.zeros_like(dp[..., :3])
        base_height_cmd = torch.full_like(dp[..., :1], self._base_height)
        torso_rpy_cmd = torch.zeros_like(dp[..., :3])
        return torch.cat([arm, navigate_cmd, base_height_cmd, torso_rpy_cmd], dim=-1)

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
            obs[k] = policy_obs[k].clone().float()
        for k in _RGB_OBS_KEYS:
            obs[k] = camera_obs[k].clone()

        num_envs = env.unwrapped.num_envs

        if self._obs_history is None:
            self._obs_history = {
                k: v.unsqueeze(1).repeat_interleave(self._frame_stack, dim=1) for k, v in obs.items()
            }
            self._action_queues = [deque() for _ in range(num_envs)]
            self._pending_history_reset = []
        else:
            for k, v in obs.items():
                self._obs_history[k] = torch.roll(self._obs_history[k], shifts=-1, dims=1)
                self._obs_history[k][:, -1] = v

            if self._pending_history_reset:
                for k, v in obs.items():
                    self._obs_history[k][self._pending_history_reset] = v[self._pending_history_reset].unsqueeze(1)
                self._pending_history_reset = []

        stacked_obs = self._obs_history

        # Export single-shot, and only on a step where the diffusion policy actually denoises
        # (action queues empty). On queue-non-empty steps get_action returns a cached action from
        # a prior step that isn't connected to this step's annotated input -> "non-traced tensors".
        exporting = self._leapp_export and not self._exported and len(self._action_queues[0]) == 0

        noise = None
        if exporting:
            import leapp
            from leapp import TensorSemantics, annotate
            from leapp.utils.enums import InputKindEnum

            # LEAPP semantic kinds for observations fed to the exported graph.
            obs_input_kind = {
                "left_eef_pos": InputKindEnum.BODY_POSITION,
                "left_eef_quat": InputKindEnum.BODY_ROTATION,
                "left_wrist_pose_pelvis_frame": InputKindEnum.BODY_POSE,
                "right_eef_pos": InputKindEnum.BODY_POSITION,
                "right_eef_quat": InputKindEnum.BODY_ROTATION,
                "right_wrist_pose_pelvis_frame": InputKindEnum.BODY_POSE,
                "robot_head_cam_rgb": "observation/image/rgb",
            }
            Tp = self._policy.policy.algo_config.horizon.prediction_horizon
            action_dim = self._policy.policy.ac_dim
            noise = torch.randn((num_envs, Tp, action_dim), device=self._device)

            graph_name = self._env_graph_name(env)
            leapp.start(name=graph_name)
            input_semantics = [
                TensorSemantics(k, v, kind=obs_input_kind.get(k), element_names=_OBS_ELEMENT_NAMES.get(k))
                for k, v in stacked_obs.items()
            ]
            input_semantics.append(TensorSemantics("diffusion_noise", noise, kind="noise/gaussian"))
            *traced_obs, noise = annotate.input_tensors(graph_name, input_semantics)
            stacked_obs = dict(zip(stacked_obs.keys(), traced_obs))

        model_obs = self._policy._prepare_observation(stacked_obs, batched_ob=True)

        needs_trajectory = [env_id for env_id in range(num_envs) if len(self._action_queues[env_id]) == 0]
        if needs_trajectory:
            sub_obs = {k: v[needs_trajectory] for k, v in model_obs.items()}
            sub_noise = noise[needs_trajectory] if noise is not None else None
            action_sequences = self._policy.policy._get_action_trajectory(obs_dict=sub_obs, noise=sub_noise)  # (n, Ta, Da)
            if exporting:
                # On the export step every env needs a trajectory, so action_sequences is the full
                # (num_envs, Ta, Da) chunk. Export the assembled *chunk* so deployment denoises once
                # per chunk and replays it, instead of re-running the diffusion model every step.
                action_chunk = self._assemble_action(action_sequences)  # (num_envs, Ta, 23)
                annotate.output_tensors(
                    graph_name,
                    [TensorSemantics("action", action_chunk, kind="command/g1_wbc_pink_action", element_names=_ACTION_ELEMENT_NAMES)],
                    export_with="onnx",
                )
                leapp.stop()
                # Relax atol: validation runs the ONNX graph on CPU (ORT) vs the CUDA/PyTorch
                # reference, which drifts ~1e-4 across the deep UNet x denoising -- negligible.
                leapp.compile_graph(atol=1e-3)
                self._exported = True
            for action_sequence, env_id in zip(action_sequences, needs_trajectory):
                self._action_queues[env_id].extend(action_sequence)

        dp_actions = torch.stack([self._action_queues[env_id].popleft() for env_id in range(num_envs)]).to(device=env_device)
        return self._assemble_action(dp_actions)  # (num_envs, 23)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            if self._policy is not None:
                self._policy.start_episode()
            self._obs_history = None
            self._action_queues = None
            self._pending_history_reset = []
            return

        if self._action_queues is None:
            return

        env_id_list = env_ids.flatten().tolist()
        for env_id in env_id_list:
            self._action_queues[env_id].clear()
        self._pending_history_reset = list(set(self._pending_history_reset).union(env_id_list))
