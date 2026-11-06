# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL model interface and deterministic compact-residual checkpoint conversion."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from robomimic.models.residual_policy_nets import ResidualWristPolicy, checkpoint_sha256
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from isaaclab_arena.policy.residual_observation import RESIDUAL_FEATURE_DIMS, residual_mode_from_config
from robomimic.models.feedback_residual_policy_nets import (
    FeedbackResidualPolicy, MODEL_TYPE, _check_state, _same_record, write_export_binding,
)

RESIDUAL_CONFIG = {"feature_dim": 220, "horizon": 6, "max_translation": 0.04}


def _feedback_binding(compact: FeedbackResidualPolicy) -> dict:
    return deepcopy({"model_type": MODEL_TYPE, "config": compact.config,
                     "base_sha256": compact.base_sha256, "golden_sha256": compact.golden_sha256,
                     "auxiliary_seed": compact.auxiliary_seed, "feedback_contract": compact.contract})


class _FeedbackRslHead(torch.nn.Module):
    """Delegate to the single tested matched actor; bind non-tensor mode in native checkpoints."""

    def __init__(self, compact: FeedbackResidualPolicy) -> None:
        super().__init__()
        self.actor = compact.actor
        self._binding = _feedback_binding(compact)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.actor(inputs[..., :358], inputs[..., 358:])

    def get_extra_state(self) -> dict:
        assert type(self.actor.feedback_enabled) is bool and (
            self.actor.feedback_enabled == self._binding["config"]["feedback_enabled"]
        ), "Runtime matched mode changed"
        return deepcopy(self._binding)

    def set_extra_state(self, state: dict) -> None:
        _same_record(state, self._binding)  # Never adopt a checkpoint's control/treatment mode.


class ResidualRslModel(MLPModel):
    """Expose the established residual actor or critic through RSL-RL's model API.

    Inputs concatenate frozen features and the complete uncorrected action chunk.
    RSL-RL samples raw Gaussian corrections; the environment applies the bounded
    translation. Observation normalization must remain disabled for chunk equivalence.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: str = "tanh",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        actor_init_checkpoint: str | None = None,
        actor_init_sha256: str | None = None,
        actor_init_base_sha256: str | None = None,
        feedback_enabled: bool | None = None,
        feedback_contract: dict | None = None,
        feedback_auxiliary_seed: int = 29451,
    ) -> None:
        assert obs_set in ("actor", "critic")
        initialization = (actor_init_checkpoint, actor_init_sha256, actor_init_base_sha256)
        assert all(value is None for value in initialization) or all(initialization), (
            "Actor initialization requires checkpoint, expected SHA-256 and frozen-base SHA-256"
        )
        assert obs_set == "actor" or actor_init_checkpoint is None, "Actor initialization cannot initialize a critic"
        assert tuple(hidden_dims) == (128, 128) and activation == "tanh"
        assert not obs_normalization, "Residual observation normalization must stay disabled"
        assert output_dim == (6 if obs_set == "actor" else 1)
        assert feedback_enabled is None or type(feedback_enabled) is bool, "Explicit matched bool or None required"
        assert feedback_enabled is not None or (feedback_contract is None and feedback_auxiliary_seed == 29451), (
            "Feedback configuration requires explicit matched mode"
        )
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        if feedback_enabled is not None:
            assert obs_set == "actor" and self.obs_dim == 378, "Matched actor requires 358 legacy + 20 feedback"
            assert actor_init_checkpoint is not None, "Matched actor requires verified golden initialization"
            assert type(feedback_contract) is dict, "Matched actor requires an explicit causal contract"
            assert type(self.distribution) is GaussianDistribution and self.distribution.std_type == "log"
            assert self.distribution.log_std_param.shape == (6,)
            compact = FeedbackResidualPolicy(
                actor_init_checkpoint, golden_sha256=actor_init_sha256, base_sha256=actor_init_base_sha256,
                contract=feedback_contract, feedback_enabled=feedback_enabled, seed=feedback_auxiliary_seed,
            )
            self.mlp = _FeedbackRslHead(compact)
            self.actor_initialization = {
                "mode": "frozen_golden_matched", "checkpoint": str(Path(actor_init_checkpoint).resolve(strict=True)),
                "sha256": actor_init_sha256, "base_sha256": actor_init_base_sha256,
                "feedback_enabled": feedback_enabled, "auxiliary_seed": feedback_auxiliary_seed,
                "feedback_contract": deepcopy(compact.contract),
            }
            return
        feature_dim = self.obs_dim - 6 * 23
        assert feature_dim in RESIDUAL_FEATURE_DIMS.values(), (
            "Expected frozen 220-D visual or 92-D blind features followed by the 6x23 base chunk"
        )
        if obs_set == "actor":
            assert isinstance(self.distribution, GaussianDistribution) and self.distribution.std_type == "log"
        else:
            assert self.distribution is None
        compact = ResidualWristPolicy(**{**RESIDUAL_CONFIG, "feature_dim": feature_dim})
        self.mlp = compact.actor if obs_set == "actor" else compact.critic
        self.actor_initialization = None
        if actor_init_checkpoint is not None:
            checkpoint_path = Path(actor_init_checkpoint).resolve(strict=True)
            assert checkpoint_sha256(str(checkpoint_path)) == actor_init_sha256, "Actor checkpoint SHA-256 mismatch"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert payload["format_version"] == 1
            assert payload["base_sha256"] == actor_init_base_sha256, "Actor/base checkpoint SHA-256 mismatch"
            assert payload["config"] == compact.config, "Actor initialization configuration mismatch"
            state = {key.removeprefix("actor."): value for key, value in payload["state_dict"].items()
                     if key.startswith("actor.")}
            expected = self.mlp.state_dict()
            assert state.keys() == expected.keys(), "Actor initialization parameter keys mismatch"
            for key, value in state.items():
                assert value.shape == expected[key].shape and value.dtype == expected[key].dtype, (
                    f"Actor initialization shape/dtype mismatch: {key}"
                )
                assert torch.isfinite(value).all(), f"Non-finite actor initialization parameter: {key}"
            # No extra model construction: loading only existing parameters preserves the RNG stream,
            # freshly initialized critic, configured Gaussian scale and subsequently created optimizer.
            self.mlp.load_state_dict(state, strict=True)
            self.actor_initialization = {
                "mode": "actor_only", "checkpoint": str(checkpoint_path),
                "sha256": actor_init_sha256, "base_sha256": actor_init_base_sha256,
            }


def _export_feedback_checkpoint(checkpoint_path: Path, output: str, base: str, context: dict, context_path: Path):
    """Dispatch matched state before the legacy tensor-only loop; no optimizer or critic is exported."""
    assert context["model_type"] == MODEL_TYPE, "Unknown residual model type"
    assert context["feedback_availability_hypothesis"] == "zero_joint_publication_delay_unqualified"
    assert context["timing_profile"] == "fresh240"
    assert type(context["variance_noise_seed"]) is int and context["variance_noise_seed"] >= 0
    diffusion = context["diffusion_contract"]
    assert type(diffusion) is dict and diffusion.keys() == {
        "variance_noise_seed", "variance_mode", "initial_noise", "torch_version", "cuda_version",
        "diffusers_version", "generator_device_type",
    }, "Actual training diffusion/runtime contract required"
    _same_record(diffusion["variance_noise_seed"], context["variance_noise_seed"])
    assert diffusion["variance_mode"] == "fixed_per_chunk_per_environment"
    base_sha = checkpoint_sha256(base)
    assert context["base_sha256"] == base_sha, "Residual/base checkpoint SHA-256 mismatch"
    initialization = context["actor_initialization"]
    assert initialization["mode"] == "frozen_golden_matched"
    assert initialization["base_sha256"] == base_sha
    _same_record(initialization["feedback_enabled"], context["feedback_enabled"])
    _same_record(initialization["feedback_contract"], context["feedback_contract"])
    model = FeedbackResidualPolicy(
        initialization["checkpoint"], golden_sha256=initialization["sha256"], base_sha256=base_sha,
        contract=context["feedback_contract"], feedback_enabled=context["feedback_enabled"],
        seed=initialization["auxiliary_seed"],
    )
    _same_record(context["config"], model.config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor = payload["actor_state_dict"]
    assert actor.keys() == {"mlp._extra_state", "distribution.log_std_param"} | {
        f"mlp.{name}" for name in model.state_dict()
    }, "Matched RSL actor state keys differ"
    _same_record(actor["mlp._extra_state"], _feedback_binding(model))
    _check_state({"log_std": actor["distribution.log_std_param"]}, {"log_std": (6,)})
    state = {name: actor[f"mlp.{name}"] for name in model.state_dict()}
    _check_state(state, model._state_shapes)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        critic = ResidualWristPolicy(**RESIDUAL_CONFIG).float().critic
    _check_state(payload["critic_state_dict"], {f"mlp.{name}": tuple(value.shape)
                                             for name, value in critic.state_dict().items()})
    model.load_state_dict(state, strict=True)
    _same_record(json.loads(context_path.read_text()), context)
    Path(output).parent.mkdir(parents=True, exist_ok=True)  # Existing train.bash exports into new models/.
    model.save(output)  # Includes independent verified-golden prior digest and exclusive-write checks.
    write_export_binding(output, checkpoint_path, context_path)
    return model.eval()


def export_residual_checkpoint(
    checkpoint: str,
    output: str,
    base: str,
    context_path: str | None = None,
) -> torch.nn.Module:
    """Convert an RSL-RL checkpoint to the existing deterministic deployment format.

    Args:
        checkpoint: RSL-RL checkpoint with separate actor and critic state dictionaries.
        output: New compact residual checkpoint path.
        base: Frozen base checkpoint, used to verify the training context's SHA-256.
        context_path: Training manifest; defaults to residual_context.json beside the checkpoint.

    Returns:
        The converted CPU model in evaluation mode.
    """
    checkpoint_path = Path(checkpoint)
    assert Path(output).resolve() != checkpoint_path.resolve(), "Do not overwrite the RSL-RL checkpoint"
    context_file = Path(context_path) if context_path else checkpoint_path.parent / "residual_context.json"
    context = json.loads(context_file.read_text())
    assert context["format_version"] == 1
    if "model_type" in context:
        return _export_feedback_checkpoint(checkpoint_path, output, base, context, context_file)
    config = context["config"]
    assert config in [{**RESIDUAL_CONFIG, "feature_dim": size} for size in RESIDUAL_FEATURE_DIMS.values()], (
        "Residual configuration mismatch"
    )
    input_mode = residual_mode_from_config(config)
    assert context.get("input_mode", "vision") == input_mode, "Residual input mode mismatch"
    base_sha = checkpoint_sha256(base)
    assert context["base_sha256"] == base_sha, "Residual/base checkpoint SHA-256 mismatch"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ResidualWristPolicy(**context["config"])
    state = {}
    for role in ("actor", "critic"):
        for key, value in payload[f"{role}_state_dict"].items():
            assert torch.isfinite(value).all(), f"Non-finite {role} parameter: {key}"
            if role == "actor" and key == "distribution.log_std_param":
                state["log_std"] = value
            else:
                assert key.startswith("mlp."), f"Unsupported {role} state: {key}"
                state[f"{role}.{key.removeprefix('mlp.')}"] = value
    model.load_state_dict(state, strict=True)
    model.eval()
    model.save(
        output,
        base_sha,
        metadata={
            "source": "rsl_rl",
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_iteration": payload.get("iter"),
            "input_mode": input_mode,
            "training_timing_profile": context.get("timing_profile", "instant"),
            "training_variance_noise_seed": context.get("variance_noise_seed"),
            "training_diffusion_contract": context.get("diffusion_contract"),
            "training_timing_parameters": context.get("timing_parameters"),
            "training_deployment_pd_profile": context.get("deployment_pd_profile", "current"),
            "training_deployment_hand_target_profile": context.get("deployment_hand_target_profile", {"profile": "current"}),
            "training_actor_initialization": context.get("actor_initialization"),
            "exploration": "independent_per_env_step; deterministic correction constant over cached base chunk",
        },
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--context", default=None)
    args = parser.parse_args()
    export_residual_checkpoint(args.checkpoint, args.output, args.base, args.context)
