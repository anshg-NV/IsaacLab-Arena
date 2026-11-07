# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch
from collections import deque
from dataclasses import dataclass
from importlib.metadata import version
from gymnasium.spaces.dict import Dict as GymSpacesDict

import robomimic.utils.file_utils as FileUtils

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.policy.deployment_timing import DeploymentTiming, TIMING_PROFILES
from isaaclab_arena.policy.residual_observation import residual_features, residual_mode_from_config


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

# Per-element labels only where the layout is unambiguous (xyz positions). Quaternions,
# 4x4 pose matrices, and the image tensor are left unlabeled.
_OBS_ELEMENT_NAMES = {
    "left_eef_pos": ["x", "y", "z"],
    "right_eef_pos": ["x", "y", "z"],
}
# Per-dimension labels for the 23-D output with continuous grippers and fixed standing commands.
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
    base_height: float = 0.72
    device: str = "cuda:0"
    leapp_export: bool = False
    residual_model_path: str | None = None
    cache_residual_inputs: bool = False
    record_residual_metrics: bool = False
    residual_input_mode: str = "vision"
    timing_profile: str = "instant"
    variance_noise_seed: int | None = None
    feedback_enabled: bool | None = None
    feedback_contract: dict | None = None
    feedback_instance_id: str | None = None
    feedback_availability_hypothesis: str | None = None
    residual_binding_path: str | None = None
    residual_binding_sha256: str | None = None

    def __post_init__(self) -> None:
        """Resolve CLI-compatible independently pinned compact context before policy logging."""
        if self.residual_binding_path is None:
            return
        assert self.residual_model_path is not None and self.residual_binding_sha256 is not None
        from robomimic.models.feedback_residual_policy_nets import load_export_context, _same_record

        context = load_export_context(self.residual_model_path, self.residual_binding_path,
                                      self.residual_binding_sha256)
        for name in ("feedback_contract", "feedback_enabled", "feedback_availability_hypothesis"):
            supplied = getattr(self, name)
            if supplied is not None:
                _same_record(supplied, context[name])
            setattr(self, name, context[name])


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
    action_space = "eef"

    action_element_names = _ACTION_ELEMENT_NAMES
    action_kind = "command/g1_wbc_pink_action"
    obs_element_names = _OBS_ELEMENT_NAMES

    def __init__(self, config: RobomimicStandPolicyCfg):
        """
        Initialize StandPolicy.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)
        self._native_feedback, self._feedback_tick = None, 0
        if config.feedback_contract is None:
            assert all(value is None for value in (config.feedback_enabled, config.feedback_instance_id,
                config.feedback_availability_hypothesis, config.residual_binding_path, config.residual_binding_sha256))
        else:
            from robomimic.models.feedback_residual_policy_nets import feedback_contract, _same_record
            from isaaclab_arena.policy.native_feedback import ZERO_DELAY_HYPOTHESIS

            assert type(config.feedback_enabled) is bool and type(config.feedback_instance_id) is str and config.feedback_instance_id
            assert config.feedback_availability_hypothesis == ZERO_DELAY_HYPOTHESIS
            assert config.timing_profile == "fresh240" and config.residual_input_mode == "vision"
            assert not config.leapp_export, "Native feedback is not the separate explicit20-input export path"
            assert config.cache_residual_inputs or config.residual_model_path, "Feedback requires its actor or training cache"
            _same_record(config.feedback_contract, feedback_contract(**config.feedback_contract["age_limits_ns"]))
            if config.residual_model_path is None:
                assert config.residual_binding_path is None and config.residual_binding_sha256 is None
        self._base_height = config.base_height
        self._device = config.device
        self._checkpoint_path = config.robomimic_checkpoint
        self._leapp_export = config.leapp_export
        self._variance_noise_seed = config.variance_noise_seed
        assert self._variance_noise_seed is None or (
            type(self._variance_noise_seed) is int and self._variance_noise_seed >= 0
        ), "variance_noise_seed must be a nonnegative int"
        if config.residual_model_path:
            if config.feedback_contract is not None:
                from robomimic.models.feedback_residual_policy_nets import load_export_context

                context = load_export_context(config.residual_model_path, config.residual_binding_path,
                                              config.residual_binding_sha256)
                assert context["feedback_availability_hypothesis"] == config.feedback_availability_hypothesis
                _same_record(context["feedback_contract"], config.feedback_contract)
                assert context["feedback_enabled"] is config.feedback_enabled
                assert type(context["variance_noise_seed"]) is int and context["variance_noise_seed"] >= 0
                metadata = {"training_variance_noise_seed": context["variance_noise_seed"],
                            "training_diffusion_contract": context["diffusion_contract"]}
            else:
                metadata = torch.load(config.residual_model_path, map_location="cpu", weights_only=True).get("metadata", {})
            trained_seed = metadata.get("training_variance_noise_seed")
            if trained_seed is not None:
                assert self._variance_noise_seed in (None, trained_seed), "Residual diffusion-noise contract mismatch"
                self._variance_noise_seed = trained_seed
        assert self._variance_noise_seed is None or (
            type(self._variance_noise_seed) is int and self._variance_noise_seed >= 0
        ), "Checkpoint variance_noise_seed must be a nonnegative int"
        assert config.timing_profile in TIMING_PROFILES
        assert not config.leapp_export or config.timing_profile == "instant", "Export the stateless graph without timing glue"
        self._timing: DeploymentTiming | None = None
        self._obs_history: dict[str, torch.Tensor] | None = None
        self._action_queues: list[deque] | None = None
        self._pending_history_reset: list[int] = []
        # LEAPP export runs once, on a step where the diffusion policy actually denoises.
        self._exported = False
        self._cache_residual_inputs = config.cache_residual_inputs
        assert config.residual_input_mode in ("vision", "blind")
        self._residual_input_mode = config.residual_input_mode
        self._residual_inputs: torch.Tensor | None = None
        self._record_residual_metrics = config.record_residual_metrics
        self._correction_queues: list[deque] | None = None
        self._evaluation_corrections: list[torch.Tensor] = []
        self._evaluation_camera_frames: list[torch.Tensor] = []
        self._evaluation_pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self._evaluation_num_envs = 0

        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=self._checkpoint_path,
            device=self._device,
            verbose=True,
        )
        obs_modalities = self._policy.policy.global_config.observation.modalities.obs
        self._low_dim_obs_keys = list(obs_modalities.get("low_dim", []))
        self._camera_obs_keys = list(obs_modalities.get("rgb", [])) + list(obs_modalities.get("depth", []))

        self._residual = None
        if config.residual_model_path:
            from robomimic.models.residual_policy_nets import ResidualWristPolicy, checkpoint_sha256, freeze_base

            freeze_base(self._policy.policy)
            if config.feedback_contract is not None:
                from robomimic.models.feedback_residual_policy_nets import FeedbackResidualPolicy

                base_sha = checkpoint_sha256(self._checkpoint_path)
                prior = context["actor_initialization"]
                assert context["base_sha256"] == prior["base_sha256"] == base_sha
                self._residual = FeedbackResidualPolicy.load(
                    config.residual_model_path, golden_path=prior["checkpoint"], golden_sha256=prior["sha256"],
                    base_sha256=base_sha, contract=config.feedback_contract,
                    feedback_enabled=config.feedback_enabled,
                ).to(self._device)
            else:
                residual_type = ResidualWristPolicy
                if self.action_space == "joint":
                    from isaaclab_arena.policy.residual_joint_policy import ResidualJointPolicy

                    residual_type = ResidualJointPolicy
                self._residual = residual_type.load(
                    config.residual_model_path, self._device, checkpoint_sha256(self._checkpoint_path)
                )
            # Evaluation/export use the checkpoint's contract, never a separate blind flag.
            if self.action_space == "joint":
                self._residual_input_mode = self._residual.config["input_mode"]
            elif config.feedback_contract is not None:
                self._residual_input_mode = "vision"
            else:
                self._residual_input_mode = residual_mode_from_config(self._residual.config)
            if self._record_residual_metrics:
                assert self._residual.config["max_translation"] == 0.04, "Evaluation metrics require the 40 mm bound"

        algo_horizon = self._policy.policy.global_config.algo.get("horizon", {})
        self._frame_stack = max(1, int(algo_horizon.get("observation_horizon", 1)))
        self.diffusion_contract = {
            "variance_noise_seed": self._variance_noise_seed,
            "variance_mode": "legacy_resampled" if self._variance_noise_seed is None else "fixed_per_chunk_per_environment",
            "initial_noise": "fresh or explicit graph input; not fixed by variance_noise_seed",
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "diffusers_version": version("diffusers"),
            "generator_device_type": torch.device(self._device).type,
        }
        if config.residual_model_path and trained_seed is not None:
            trained_contract = metadata.get("training_diffusion_contract")
            assert trained_contract == self.diffusion_contract, "Seeded residual requires its training RNG/runtime contract"

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

    def _prepare_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        """Restore physical action units before residual inputs and corrections are formed."""
        return actions

    def _assemble_action(self, dp: torch.Tensor) -> torch.Tensor:
        """Turn a diffusion-policy output ``(..., ac_dim)`` into the 23-dim WBC action ``(..., 23)``.

        Layout: arm ``[0:16]`` from the policy (grippers binarized to the {open, closed} values the
        WBC controller expects) + fixed lower-body stand commands (navigate=0, base_height, torso=0).
        Operates on the last dim, so it works for a single action ``(N, 23)`` or a chunk ``(N, Ta, 23)``.
        """
        assert dp.shape[-1] == 23, "End-effector policies require 23-D absolute actions"
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

    def _export_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        """Export fixed standing commands with continuous grippers for downstream corrections."""
        assembled = self._assemble_action(actions)
        return torch.cat([actions[..., :2], assembled[..., 2:]], dim=-1)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        if self._native_feedback is not None:
            self._native_feedback._check(self._feedback_tick, "ready")
        env_device = torch.device(env.unwrapped.device)

        # ---- Runtime glue: select keys + frame-stack (NOT traced) ----
        # Frame-stacking is cross-call state (the deque holds frames from prior steps), so it
        # cannot be a pure function of this step's inputs and must stay outside the traced graph.
        # The stacked obs is annotated as the graph input below; deployment supplies the history.
        policy_obs = observation.get("policy", observation)
        camera_obs = observation.get("camera_obs", {})

        obs = {}
        for k in self._low_dim_obs_keys:
            obs[k] = policy_obs[k].clone().float()
        for k in self._camera_obs_keys:
            obs[k] = camera_obs[k].clone()

        num_envs = env.unwrapped.num_envs
        if self.config.timing_profile != "instant":
            if self._timing is None:
                self._timing = DeploymentTiming(
                    self.config.timing_profile, num_envs, env.unwrapped.step_dt,
                    **({"track_consumption": True} if self.config.feedback_contract is not None else {}),
                )
                if self.config.feedback_contract is not None:
                    from isaaclab_arena.policy.native_feedback import NativeFeedback

                    names, positions = self._native_joint_state(env)
                    self._native_feedback = NativeFeedback(
                        self._timing, names, instance_id=self.config.feedback_instance_id,
                        timing_profile=self.config.timing_profile,
                        joint_delay_hypothesis=self.config.feedback_availability_hypothesis,
                        **self.config.feedback_contract["age_limits_ns"],
                    )
                    self._native_feedback.initialize(positions, global_tick=self._feedback_tick)
            obs = self._timing.observe(obs)
        if self._record_residual_metrics:
            assert self._evaluation_pending is None, "Previous evaluation action was not recorded as consumed"
            pixels = obs["robot_head_cam_rgb"].float().flatten(start_dim=1)
            variance, mean = torch.var_mean(pixels, dim=1, correction=0)
            camera_stats = torch.stack((mean, variance.sqrt()), dim=-1)
            assert torch.isfinite(camera_stats).all() and (camera_stats > 2.0).all(), (
                "Invalid actor camera frame: each environment requires finite RGB mean/std greater than 2 DN"
            )
            self._evaluation_num_envs = num_envs

        if self._obs_history is None:
            self._obs_history = {
                k: v.unsqueeze(1).repeat_interleave(self._frame_stack, dim=1) for k, v in obs.items()
            }
            self._action_queues = [deque() for _ in range(num_envs)]
            if self._record_residual_metrics:
                self._correction_queues = [deque() for _ in range(num_envs)]
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
        if self._timing is not None:
            stacked_obs = self._timing.source(stacked_obs)
            for env_id, inputs in self._timing.deliver(self._action_queues, self._correction_queues):
                if self._cache_residual_inputs:
                    self._residual_inputs[env_id] = inputs

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
                "robot_joint_pos": InputKindEnum.JOINT_POSITION,
                "left_arm_joint_pos": InputKindEnum.JOINT_POSITION,
                "right_arm_joint_pos": InputKindEnum.JOINT_POSITION,
                "robot_head_cam_rgb": "observation/image/rgb",
                "robot_head_cam_depth": "observation/image/depth",
            }
            Tp = self._policy.policy.algo_config.horizon.prediction_horizon
            action_dim = self._policy.policy.ac_dim
            noise = torch.randn((num_envs, Tp, action_dim), device=self._device)

            graph_name = self._env_graph_name(env)
            from isaaclab_arena.evaluation.placement_regions import collect_placement_regions, write_placement_regions

            placement_regions = collect_placement_regions(env.unwrapped, graph_name)
            leapp.start(name=graph_name)
            input_semantics = [
                TensorSemantics(k, v, kind=obs_input_kind.get(k), element_names=self.obs_element_names.get(k))
                for k, v in stacked_obs.items()
            ]
            input_semantics.append(TensorSemantics("diffusion_noise", noise, kind="noise/gaussian"))
            *traced_obs, noise = annotate.input_tensors(graph_name, input_semantics)
            stacked_obs = dict(zip(stacked_obs.keys(), traced_obs))

        model_obs = self._policy._prepare_observation(stacked_obs, batched_ob=True)

        needs_trajectory = [env_id for env_id in range(num_envs) if len(self._action_queues[env_id]) == 0]
        if self._timing is not None:
            needs_trajectory = self._timing.needs_inference()
        if needs_trajectory:
            sub_obs = {k: v[needs_trajectory] for k, v in model_obs.items()}
            sub_noise = noise[needs_trajectory] if noise is not None else None
            variance_args = {} if self._variance_noise_seed is None else {"variance_noise_seed": self._variance_noise_seed}
            feedback = None
            if self._native_feedback is not None:
                packets = self._native_feedback.submission_packets(needs_trajectory, global_tick=self._feedback_tick)
                feedback = torch.tensor([packet.features for packet in packets], device=self._device, dtype=torch.float32)
            action_sequences = self._policy.policy._get_action_trajectory(
                obs_dict=sub_obs, noise=sub_noise, **variance_args
            )  # (n, Ta, Da)
            action_sequences = self._prepare_action_chunk(action_sequences)
            inputs = None
            if self._residual is not None or self._cache_residual_inputs:
                features = residual_features(
                    self._policy.policy, sub_obs, self._residual_input_mode, action_space=self.action_space
                )
                if self._cache_residual_inputs:
                    inputs = torch.cat((features, action_sequences.flatten(start_dim=1)), dim=-1)
                    if feedback is not None:
                        assert inputs.shape[-1] == 358
                        inputs = torch.cat((inputs, feedback), dim=-1)
                    if self._residual_inputs is None:
                        self._residual_inputs = inputs.new_zeros((num_envs, inputs.shape[-1]))
                    if self._timing is None:
                        self._residual_inputs[needs_trajectory] = inputs
            base_sequences = action_sequences
            if self._residual is not None:
                action_sequences = (self._residual(features, action_sequences, feedback) if feedback is not None
                                    else self._residual(features, action_sequences))
            corrections = None
            if self._record_residual_metrics:
                axes = [2, 3, 4, 9, 10, 11]
                corrections = action_sequences[..., axes] - base_sequences[..., axes]
                if self._timing is None:
                    for correction_sequence, env_id in zip(corrections, needs_trajectory):
                        self._correction_queues[env_id].extend(correction_sequence)
            if exporting:
                assert action_sequences.dtype == torch.float32
                action_chunk = self._export_action_chunk(action_sequences)
                annotate.output_tensors(
                    graph_name,
                    [TensorSemantics("action", action_chunk, kind=self.action_kind, element_names=self.action_element_names)],
                    export_with="onnx",
                )
                leapp.stop()
                # Relax atol: validation runs the ONNX graph on CPU (ORT) vs the CUDA/PyTorch
                # reference, which drifts ~1e-4 across the deep UNet x denoising -- negligible.
                validation_results = leapp.compile_graph(atol=1e-3)
                assert validation_results == {graph_name: True}, "LEAPP numerical validation did not pass"
                # LEAPP hides its informational validation logs when verbose=False.
                print(f"LEAPP_VALIDATION_PASSED: {graph_name}", flush=True)
                write_placement_regions(placement_regions, graph_name)
                self._exported = True
            for row, (action_sequence, env_id) in enumerate(zip(action_sequences, needs_trajectory)):
                if self._timing is not None:
                    prefilled = self._timing.submit(
                        env_id, action_sequence, inputs[row] if inputs is not None else None,
                        corrections[row] if corrections is not None else None,
                    )
                    if not prefilled:
                        continue
                    if self._cache_residual_inputs:
                        self._residual_inputs[env_id] = inputs[row]
                    if self._record_residual_metrics:
                        self._correction_queues[env_id].extend(corrections[row])
                self._action_queues[env_id].extend(action_sequence)

        dp_actions = (
            self._timing.consume(self._action_queues) if self._timing is not None
            else torch.stack([self._action_queues[env_id].popleft() for env_id in range(num_envs)])
        ).to(device=env_device)
        if self._record_residual_metrics:
            corrections = (
                self._timing.consume(self._correction_queues, corrections=True) if self._timing is not None
                else torch.stack([self._correction_queues[env_id].popleft() for env_id in range(num_envs)])
            )
            assert torch.isfinite(corrections).all(), "Nonfinite deterministic wrist correction"
            self._evaluation_pending = (corrections.detach(), camera_stats.detach())
        if self._timing is not None:
            self._timing.advance()
        assembled = self._assemble_action(dp_actions)
        if self._native_feedback is not None:
            self._native_feedback.prepare_step(assembled[:, :2], global_tick=self._feedback_tick)
        return assembled

    @staticmethod
    def _native_joint_state(env: gym.Env) -> tuple[list[str], torch.Tensor]:
        data = env.unwrapped.scene["robot"].data
        positions = data.joint_pos
        if not isinstance(positions, torch.Tensor):
            import warp as wp

            positions = wp.to_torch(positions)
        return list(data.joint_names), positions

    def record_native_step(self, env: gym.Env, done_ids: torch.Tensor) -> None:
        """Accept feedback only after successful env.step, before partial policy.reset."""
        if self.config.feedback_contract is None:
            return
        assert self._native_feedback is not None
        names, positions = self._native_joint_state(env)
        assert tuple(names) == self._native_feedback._names, "Native joint order changed"
        self._native_feedback.accept_step(done_ids.flatten().tolist(), positions, global_tick=self._feedback_tick)
        self._feedback_tick += 1

    def record_evaluation_step(self) -> None:
        """Count the last returned action only after its native environment step succeeds."""
        if not self._record_residual_metrics:
            return
        assert self._evaluation_pending is not None, "No unrecorded evaluation action"
        corrections, camera_stats = self._evaluation_pending
        self._evaluation_corrections.append(corrections)
        self._evaluation_camera_frames.append(camera_stats)
        self._evaluation_pending = None

    def evaluation_metrics(self) -> dict | None:
        """Summarize consumed deterministic wrist-target offsets [mm] and actor camera validity."""
        if not self._record_residual_metrics:
            return None
        assert self._evaluation_pending is None and self._evaluation_corrections
        corrections = torch.cat(self._evaluation_corrections).double().cpu() * 1000.0
        camera = torch.cat(self._evaluation_camera_frames).double().cpu()
        absolute = corrections.abs().flatten()
        saturation_threshold = 38.0
        saturated = int((absolute >= saturation_threshold).sum())
        sum_squared = float(corrections.square().sum())
        return {
            "format": 1,
            "timing_profile": self.config.timing_profile,
            "diffusion_contract": self.diffusion_contract,
            "timing_parameters": TIMING_PROFILES[self.config.timing_profile],
            "timing_startup": "instant" if self._timing is None else "one immediate prefilled chunk per reset",
            "num_envs": self._evaluation_num_envs,
            "control_steps": corrections.shape[0],
            "component_count": corrections.numel(),
            "sum_squared_mm2": sum_squared,
            "rms_mm": (sum_squared / corrections.numel()) ** 0.5,
            "per_axis_rms_mm": corrections.square().mean(dim=0).sqrt().tolist(),
            "p95_abs_mm": float(torch.quantile(absolute, 0.95)),
            "saturation_count": saturated,
            "saturation_threshold_mm": saturation_threshold,
            "saturation_frequency": saturated / corrections.numel(),
            "residual_model_path": self.config.residual_model_path,
            "definition": (
                "Consumed control steps pooled over left xyz/right xyz. Applied deterministic corrected-minus-base "
                "float32 targets, converted to mm; includes float32 rounding. No sampled residual noise. p95 uses "
                "linear interpolation; saturation is absolute component >=38 mm (95% of 40 mm bound). "
                "Unconsumed chunk tails are excluded."
            ),
            "camera_validity": {
                "frame_count": camera.shape[0],
                "all_finite": True,
                "all_nonblank": True,
                "min_mean": float(camera[:, 0].min()),
                "max_mean": float(camera[:, 0].max()),
                "min_std": float(camera[:, 1].min()),
                "max_std": float(camera[:, 1].max()),
                "minimum_mean_std_threshold_dn": 2.0,
            },
        }

    @property
    def residual_inputs(self) -> torch.Tensor:
        """Return frozen features and the complete base chunk cached at its denoising boundary."""
        assert self._cache_residual_inputs and self._residual_inputs is not None
        return self._residual_inputs

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._native_feedback is not None:
            assert env_ids is not None, "Full feedback reset requires a new policy instance"
            assert self._native_feedback._phase == "ready", "Accept the native step before scheduler reset"
        if env_ids is None:
            if self._policy is not None:
                self._policy.start_episode()
            self._obs_history = None
            self._action_queues = None
            self._pending_history_reset = []
            self._residual_inputs = None
            self._correction_queues = None
            self._evaluation_corrections = []
            self._evaluation_camera_frames = []
            self._evaluation_pending = None
            self._evaluation_num_envs = 0
            self._timing = None
            return

        if self._action_queues is None:
            return

        env_id_list = env_ids.flatten().tolist()
        if self._timing is not None:
            self._timing.reset(env_id_list)
        for env_id in env_id_list:
            self._action_queues[env_id].clear()
            if self._correction_queues is not None:
                self._correction_queues[env_id].clear()
        self._pending_history_reset = list(set(self._pending_history_reset).union(env_id_list))
