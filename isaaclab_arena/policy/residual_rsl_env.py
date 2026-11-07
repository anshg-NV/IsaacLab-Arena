# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen-base residual adapter for the existing IsaacLab RSL-RL training entrypoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
import warp as wp
from tensordict import TensorDict

from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from isaaclab_arena.policy.robomimic_stand_policy import RobomimicStandPolicy, RobomimicStandPolicyCfg
from isaaclab_arena.policy.residual_observation import RESIDUAL_FEATURE_DIMS
from isaaclab_arena.policy.residual_joint_policy import MAX_JOINT_DELTA, MAX_GRIPPER_DELTA, apply_joint_residual
from isaaclab_arena.policy.robomimic_stand_joint_policy import RobomimicStandJointPolicy, RobomimicStandJointPolicyCfg
from isaaclab_arena.policy.deployment_timing import TIMING_PROFILES
from robomimic.models.residual_policy_nets import checkpoint_sha256, freeze_base
from robomimic.models.feedback_residual_policy_nets import MODEL_TYPE, _configuration, _same_record, feedback_contract

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


GAMMA = 0.98 ** (1 / 6)


@configclass
class ResidualRslActorCfg(RslRlMLPModelCfg):
    """Optional actor-only initialization; critic and optimizer always start fresh."""

    actor_init_checkpoint: str | None = None
    actor_init_sha256: str | None = None
    actor_init_base_sha256: str | None = None
    feedback_enabled: bool | None = None
    feedback_contract: dict | None = None
    feedback_auxiliary_seed: int = 29451


@configclass
class ResidualRslRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO at the native control rate, with a chunk-conditioned deterministic residual."""

    env_wrapper = "isaaclab_arena.policy.residual_rsl_env:ResidualRslEnv"
    init_at_random_ep_len = False
    num_steps_per_env = 120
    max_iterations = 40
    save_interval = 1
    experiment_name = "hubble_residual"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = ResidualRslActorCfg(
        class_name="isaaclab_arena.policy.residual_rsl_model:ResidualRslModel",
        hidden_dims=[128, 128], activation="tanh", obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=math.exp(-1), std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        class_name="isaaclab_arena.policy.residual_rsl_model:ResidualRslModel",
        hidden_dims=[128, 128], activation="tanh", obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.001, num_learning_epochs=4, num_mini_batches=4,
        learning_rate=3e-4, schedule="adaptive", gamma=GAMMA, lam=0.95 ** (1 / 6),
        desired_kl=0.03, max_grad_norm=0.5,
    )


def _tensor(value: torch.Tensor | wp.array) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def residual_reward(env: ManagerBasedRLEnv, variant: str = "historical", gamma: float = GAMMA,
                    regularization: str = "legacy_mean6_v1") -> torch.Tensor:
    """Return a reward rate whose native dt integration gives the selected transition reward."""
    assert variant in ("historical", "S1", "D1"), f"Unknown residual reward: {variant}"
    adapter = env._residual_adapter
    assert regularization == "legacy_mean6_v1"
    success = env.termination_manager.get_term("success")
    dropped = env.termination_manager.get_term("object_dropped") & ~success
    adapter.last_success = (success & env.reset_buf).clone()
    adapter.last_dropped = (dropped & env.reset_buf).clone()
    if variant == "S1":
        reward = adapter.last_success.float()
    else:
        reward = 5 * (gamma * adapter.potential() * ~env.reset_buf - adapter.previous_potential)
        reward += 10 * success.float() - 3 * dropped.float()
        raw = adapter.raw_residual
        penalty = raw.tanh().square().mean(-1)
        reward -= (0.02 / 6) * penalty
        if variant == "D1":
            reward = reward / 10
    # RewardManager multiplies each term by dt; this term is an already integrated transition reward.
    return reward / env.step_dt


@configclass
class ResidualRewardsCfg:
    task = RewardTermCfg(func=residual_reward, weight=1.0, params={"variant": "historical", "gamma": GAMMA})


def record_residual_outcome(env: ManagerBasedRLEnv, env_id: int) -> dict:
    """Record mutually exclusive terminal causes before native auto-reset."""
    success = bool(env.termination_manager.get_term("success")[env_id].item())
    drop = bool(env.termination_manager.get_term("object_dropped")[env_id].item()) and not success
    timeout = bool(env.termination_manager.get_term("time_out")[env_id].item()) and not (success or drop)
    outcome = {"drop": drop, "timeout": timeout}
    term = env.action_manager.get_term("g1_action")
    if term.hand_target_contract["profile"] == "episode_alternating_abrupt_ramp075":
        outcome["hand_target_domain"] = term.hand_target_episode_domain(env_id)
    return {"residual_outcome": outcome}


class ResidualRslEnv(RslRlVecEnvWrapper):
    """Expose residual actions while retaining native stepping, rendering, and auto-reset.

    The actor input stays fixed for each base chunk, so its deterministic mean is exactly
    chunk-constant, matching LEAPP. Exploration is sampled independently at each control
    step.
    """

    def __init__(self, env: gym.Env, clip_actions: float | None = None) -> None:
        assert clip_actions is None, "The residual uses tanh bounds, not clipped Gaussian actions"
        super().__init__(env, clip_actions=None)
        assert self.num_actions in (23, 50), "Residuals require an EEF or joint G1 embodiment"
        self.residual_action_space = "joint" if self.num_actions == 50 else "eef"
        assert self.cfg.is_finite_horizon
        self.num_actions = 16 if self.residual_action_space == "joint" else 6
        self.base_action_dim = 16 if self.residual_action_space == "joint" else 23
        self.residual_input_mode = getattr(self.cfg, "residual_input_mode", "vision")
        self.residual_feature_dim = (
            None if self.residual_action_space == "joint" else RESIDUAL_FEATURE_DIMS[self.residual_input_mode]
        )
        self.residual_feedback_enabled = getattr(self.cfg, "residual_feedback_enabled", None)
        assert self.residual_action_space == "eef" or self.residual_feedback_enabled is None
        feedback_args = {}
        if self.residual_feedback_enabled is not None:
            assert type(self.residual_feedback_enabled) is bool
            initialization = self.cfg.residual_actor_initialization
            assert initialization["mode"] == "frozen_golden_matched"
            _same_record(initialization["feedback_enabled"], self.residual_feedback_enabled)
            _same_record(initialization["feedback_contract"], self.cfg.residual_feedback_contract)
            feedback_args = {
                "feedback_enabled": self.residual_feedback_enabled,
                "feedback_contract": deepcopy(self.cfg.residual_feedback_contract),
                "feedback_availability_hypothesis": self.cfg.residual_feedback_availability_hypothesis,
                "feedback_instance_id": str(Path(self.cfg.log_dir).resolve()) + "/native_training_policy",
            }
        policy_type, config_type = (
            (RobomimicStandJointPolicy, RobomimicStandJointPolicyCfg)
            if self.residual_action_space == "joint"
            else (RobomimicStandPolicy, RobomimicStandPolicyCfg)
        )
        self.policy = policy_type(
            config_type(
                robomimic_checkpoint=self.cfg.residual_base,
                device=self.device,
                cache_residual_inputs=True,
                residual_input_mode=self.residual_input_mode,
                timing_profile=getattr(self.cfg, "residual_timing_profile", "instant"),
                variance_noise_seed=getattr(self.cfg, "residual_variance_noise_seed", None),
                **feedback_args,
            )
        )
        freeze_base(self.policy._policy.policy)
        self.policy.reset()
        self.residual_horizon = self.policy._policy.policy.algo_config.horizon.action_horizon
        assert self.residual_action_space == "joint" or self.residual_horizon == 6
        self.raw_residual = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_dropped = torch.zeros_like(self.last_success)
        self.initial_z = self.apple_position()[:, 2].clone()
        self.palm_index = self.unwrapped.scene["robot"].body_names.index("left_hand_palm_link")
        self.previous_potential = self.potential()
        self.unwrapped._residual_adapter = self
        self._prepare(self.unwrapped.obs_buf)
        residual_config = (
            dict(
                feature_dim=self.residual_feature_dim,
                horizon=self.residual_horizon,
                input_mode=self.residual_input_mode,
                max_joint_delta=MAX_JOINT_DELTA,
                max_gripper_delta=MAX_GRIPPER_DELTA,
            )
            if self.residual_action_space == "joint"
            else dict(feature_dim=self.residual_feature_dim, horizon=6, max_translation=0.04)
        )
        initialization = getattr(self.cfg, "residual_actor_initialization", None)
        if self.residual_action_space == "joint" and initialization is not None:
            payload = torch.load(initialization["checkpoint"], map_location="cpu", weights_only=True)
            assert payload["config"] == residual_config, "Actor initialization joint residual configuration mismatch"
        context = {
            "format_version": 1,
            "base_sha256": checkpoint_sha256(self.cfg.residual_base),
            "config": residual_config,
            "action_space": self.residual_action_space,
            "input_mode": self.residual_input_mode,
            "actor_initialization": getattr(self.cfg, "residual_actor_initialization", None),
            "timing_profile": self.policy.config.timing_profile,
            "variance_noise_seed": self.policy._variance_noise_seed,
            "diffusion_contract": self.policy.diffusion_contract,
            "timing_parameters": TIMING_PROFILES[self.policy.config.timing_profile],
            "timing_startup": "one immediate prefilled chunk per reset; subsequent chunks delayed"
            if self.policy.config.timing_profile != "instant" else "instant",
            "deployment_pd_profile": getattr(self.cfg, "deployment_pd_profile", "current"),
            "deployment_hand_target_profile": self.unwrapped.action_manager.get_term("g1_action").hand_target_contract,
            "control_dt": self.unwrapped.step_dt,
            "exploration": "independent Gaussian each control step; deterministic input cached per base chunk",
            "use_fabric": bool(self.cfg.sim.use_fabric),
            "scene_source": "native Hubble placement/reset events",
            "reward": dict(self.cfg.rewards.task.params) if hasattr(self.cfg, "rewards") else {
                "variant": "historical", "gamma": GAMMA,
            },
        }
        if context["actor_initialization"] is not None:
            assert context["actor_initialization"]["base_sha256"] == context["base_sha256"], (
                "Actor initialization must match the actual frozen base"
            )
        if self.residual_feedback_enabled is not None:
            context.update({
                "model_type": MODEL_TYPE, "config": _configuration(self.residual_feedback_enabled),
                "feedback_enabled": self.residual_feedback_enabled,
                "feedback_contract": deepcopy(self.cfg.residual_feedback_contract),
                "feedback_contract_path": self.cfg.residual_feedback_contract_path,
                "feedback_contract_sha256": self.cfg.residual_feedback_contract_sha256,
                "feedback_availability_hypothesis": self.cfg.residual_feedback_availability_hypothesis,
                "feedback_instance_id": feedback_args["feedback_instance_id"],
            })
        destination = Path(self.cfg.log_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "residual_context.json").write_text(json.dumps(context, indent=2))
        self.unwrapped.episode_recorder.set_job_name("rsl_rl")
        self.unwrapped.episode_recorder.set_output_path(destination / "episode_results.jsonl")

    def apple_position(self) -> torch.Tensor:
        """Return apple world positions [m]."""
        return _tensor(self.unwrapped.scene["apple_01_objaverse_robolab"].data.root_pos_w)

    def potential(self) -> torch.Tensor:
        """Return the reach/lift/place potential; privileged poses never enter the actor."""
        apple = self.apple_position()
        plate = _tensor(self.unwrapped.scene["clay_plates_hot3d_robolab"].data.root_pos_w)
        palm = _tensor(self.unwrapped.scene["robot"].data.body_pos_w)[:, self.palm_index]
        reach = 1 - torch.tanh(10 * (palm - apple).norm(dim=-1))
        lift = ((apple[:, 2] - self.initial_z) / 0.09).clamp(0, 1)
        place = 1 - torch.tanh(8 * (apple[:, :2] - plate[:, :2]).norm(dim=-1))
        return 0.5 * reach + 1.5 * lift + 2 * lift * place

    @torch.inference_mode()
    def _prepare(self, obs: dict) -> None:
        pixels = obs["camera_obs"]["robot_head_cam_rgb"].float().flatten(1)
        assert (pixels.mean(1) > 2).all() and (pixels.std(1) > 2).all(), "Invalid/black policy camera"
        self.base_action = self.policy.get_action(self.env, obs)
        if self.residual_action_space == "joint":
            # Keep the unthresholded action used to train gripper overrides.
            self.base_action = self.policy.raw_action.clone()
        inputs = self.policy.residual_inputs
        if self.residual_feature_dim is None:
            self.residual_feature_dim = inputs.shape[-1] - self.residual_horizon * self.base_action_dim
            assert self.residual_feature_dim > 0
        assert inputs.shape == (
            self.num_envs,
            self.residual_feature_dim
            + self.residual_horizon * self.base_action_dim
            + (20 if self.residual_feedback_enabled is not None else 0),
        )
        # Snapshot: a later chunk must not mutate the previous PPO observation.
        observations = {"policy": inputs.clone()}
        if self.residual_feedback_enabled is not None:
            observations["critic"] = inputs[:, :358].clone()
        self._observations = TensorDict(observations, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        """Return cached inputs without advancing base history or consuming another action."""
        return self._observations

    @torch.inference_mode()
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert actions.shape == (self.num_envs, self.num_actions) and torch.isfinite(actions).all()
        self.raw_residual = actions.to(self.device)
        if self.residual_action_space == "joint":
            corrected = apply_joint_residual(self.base_action, self.raw_residual)
            combined = self.policy._assemble_action(corrected)
        else:
            delta = 0.04 * self.raw_residual.tanh()
            combined = self.base_action.clone()
            combined[:, 2:5] += delta[:, :3]
            combined[:, 9:12] += delta[:, 3:]
        # Native auto-reset clears this buffer inside step; snapshot the finishing lengths first.
        episode_lengths = self.episode_length_buf.clone() + 1
        obs, rewards, terminated, truncated, extras = self.env.step(combined)
        done = terminated | truncated
        ids = done.nonzero().flatten()
        if self.residual_feedback_enabled is not None:
            self.policy.record_native_step(self.env, ids)
        extras["residual_episode"] = {
            "success": self.last_success[ids].clone(),
            "drop": self.last_dropped[ids].clone(),
            "timeout": (truncated & ~self.last_success & ~self.last_dropped)[ids].clone(),
            "length": episode_lengths[ids],
        }
        if done.any():
            self.policy.reset(ids)
            self.initial_z[ids] = self.apple_position()[ids, 2]
            extras.setdefault("log", {})["residual/success_rate"] = self.last_success[ids].float().mean()
        self.previous_potential = self.potential()
        self._prepare(obs)
        return self.get_observations(), rewards, done.long(), extras

    def close(self) -> None:
        """Detach the residual adapter and close the native environment."""
        self.unwrapped._residual_adapter = None
        super().close()


def environment_registration_callback() -> list[str]:
    """Register the unchanged Hubble scene and opt into residual PPO task settings."""
    from isaaclab_arena.environments.isaaclab_interop import environment_registration_callback as register_arena
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    remaining = register_arena()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--residual_base", required=True)
    parser.add_argument("--residual_reward", choices=("historical", "S1", "D1"), default="historical")
    parser.add_argument("--residual_input_mode", choices=("vision", "blind"), default="vision")
    parser.add_argument("--residual_timing_profile", choices=tuple(TIMING_PROFILES), default="instant")
    parser.add_argument("--residual_variance_noise_seed", type=int, default=None)
    parser.add_argument("--residual_actor_init_checkpoint", default=None)
    parser.add_argument("--residual_actor_init_sha256", default=None)
    parser.add_argument("--residual_feedback_mode", choices=("control", "feedback"), default=None)
    parser.add_argument("--residual_feedback_contract", default=None)
    parser.add_argument("--residual_feedback_contract_sha256", default=None)
    parser.add_argument("--residual_feedback_availability_hypothesis",
                        choices=("zero_joint_publication_delay_unqualified",), default=None)
    parser.add_argument("--residual_feedback_auxiliary_seed", type=int, default=29451)
    args, remaining = parser.parse_known_args(remaining)
    task_parser = argparse.ArgumentParser(add_help=False)
    task_parser.add_argument("--task", required=True)
    task = task_parser.parse_known_args()[0].task
    assert task == "hubble_g1_static_pick_and_place", "Residual action/reward contract is specific to Hubble G1"
    spec = gym.spec(task)
    cfg = spec.kwargs["env_cfg_entry_point"]
    from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action import G1DecoupledWBCJointAction

    action_space = "joint" if cfg.actions.g1_action.class_type is G1DecoupledWBCJointAction else "eef"
    assert action_space == "eef" or args.residual_feedback_mode is None, "Causal20 feedback requires EEF actions"
    cfg.residual_base = str(Path(args.residual_base).resolve(strict=True))
    cfg.residual_input_mode = args.residual_input_mode
    cfg.residual_timing_profile = args.residual_timing_profile
    cfg.residual_variance_noise_seed = args.residual_variance_noise_seed
    cfg.rewards = ResidualRewardsCfg()
    cfg.rewards.task.params["variant"] = args.residual_reward
    cfg.episode_recorders.residual_outcome = EpisodeRecorderTermCfg(func=record_residual_outcome)
    cfg.is_finite_horizon = True
    cfg.terminations.time_out.time_out = True
    runner_cfg = ResidualRslRunnerCfg()
    runner_cfg.actor.action_space = runner_cfg.critic.action_space = action_space
    assert (args.residual_actor_init_checkpoint is None) == (args.residual_actor_init_sha256 is None), (
        "Actor initialization requires both checkpoint and expected SHA-256"
    )
    cfg.residual_actor_initialization = None
    if args.residual_actor_init_checkpoint is not None:
        runner_cfg.actor.actor_init_checkpoint = str(Path(args.residual_actor_init_checkpoint).resolve(strict=True))
        runner_cfg.actor.actor_init_sha256 = args.residual_actor_init_sha256
        runner_cfg.actor.actor_init_base_sha256 = checkpoint_sha256(cfg.residual_base)
        cfg.residual_actor_initialization = {
            "mode": "actor_only", "checkpoint": runner_cfg.actor.actor_init_checkpoint,
            "sha256": runner_cfg.actor.actor_init_sha256, "base_sha256": runner_cfg.actor.actor_init_base_sha256,
        }
    cfg.residual_feedback_enabled = None
    if args.residual_feedback_mode is None:
        assert all(value is None for value in (args.residual_feedback_contract,
                   args.residual_feedback_contract_sha256, args.residual_feedback_availability_hypothesis))
        assert args.residual_feedback_auxiliary_seed == 29451, "Feedback options require an explicit mode"
    else:
        assert cfg.residual_actor_initialization is not None, "Matched modes require verified golden initialization"
        assert args.residual_input_mode == "vision"
        assert args.residual_timing_profile == "fresh240"
        assert args.residual_feedback_availability_hypothesis == "zero_joint_publication_delay_unqualified"
        assert type(args.residual_variance_noise_seed) is int and args.residual_variance_noise_seed >= 0
        assert args.residual_feedback_auxiliary_seed >= 0
        assert args.residual_feedback_contract and args.residual_feedback_contract_sha256
        contract_path = Path(args.residual_feedback_contract).resolve(strict=True)
        contract_bytes = contract_path.read_bytes()
        assert hashlib.sha256(contract_bytes).hexdigest() == args.residual_feedback_contract_sha256
        contract = json.loads(contract_bytes)
        _same_record(contract, feedback_contract(**contract["age_limits_ns"]))
        cfg.residual_feedback_enabled = args.residual_feedback_mode == "feedback"
        cfg.residual_feedback_contract = deepcopy(contract)
        cfg.residual_feedback_contract_path = str(contract_path)
        cfg.residual_feedback_contract_sha256 = args.residual_feedback_contract_sha256
        cfg.residual_feedback_availability_hypothesis = args.residual_feedback_availability_hypothesis
        runner_cfg.actor.feedback_enabled = cfg.residual_feedback_enabled
        runner_cfg.actor.feedback_contract = deepcopy(contract)
        runner_cfg.actor.feedback_auxiliary_seed = args.residual_feedback_auxiliary_seed
        runner_cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        cfg.residual_actor_initialization.update({
            "mode": "frozen_golden_matched", "feedback_enabled": cfg.residual_feedback_enabled,
            "auxiliary_seed": args.residual_feedback_auxiliary_seed, "feedback_contract": deepcopy(contract),
        })
    spec.kwargs["rsl_rl_cfg_entry_point"] = runner_cfg
    return remaining
