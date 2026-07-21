# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deployment environment that runs the LEAPP-exported RobomimicStandPolicy in simulation.

Structured to parallel :class:`isaaclab.envs.leapp_deployment_env.LeappDeploymentEnv`. Where that
env bypasses all Isaac Lab managers and wires scene-entity data directly to the model via
``isaaclab_connection`` metadata, this env *keeps* the ObservationManager and the WBC action term:
the G1 observations are transform/camera terms, and the 23-dim action must flow through the
decoupled-WBC + PINK-IK action term, neither of which reduces to a direct scene read/write. Only
the neural policy is replaced by the LEAPP ``InferenceManager``.

I/O wiring is resolved from the model's input/output names against the ObservationManager's term
groups (mirroring the original's ``isaaclab_connection`` parse):

- ``ObsInputSpec``    -- read ``obs[group][term]``, frame-stacked on the deployment side
- ``NoiseInputSpec``  -- fresh Gaussian noise each step (e.g. ``diffusion_noise``)
- ``ActionOutputSpec`` -- the assembled action, applied through the WBC term via ``env.step()``

The sim/scene/manager setup and the physics/decimation loop are delegated to a
``ManagerBasedRLEnv`` (rather than driven manually as in ``LeappDeploymentEnv``), because the
observation and action managers require the full env, and its termination/reset handling is
non-trivial to reproduce by hand.
"""

from __future__ import annotations

import json
import logging
import torch
from dataclasses import dataclass
from typing import Any

try:
    from leapp import InferenceManager
    from leapp.utils.tensor_description import map_to_torch_dtype
except ImportError as e:
    raise ImportError("LEAPP package is required for policy deployment. Install with: pip install leapp") from e

from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

logger = logging.getLogger(__name__)

# Model inputs supplied by the deployment runtime rather than the ObservationManager. ``diffusion_noise``
# is the diffusion policy's initial Gaussian noise, lifted to a graph input by the export so the model
# stays stochastic; fresh noise is drawn each step. (Absent when the export baked the noise in.)
_EXTERNAL_INPUT_NAMES = {"diffusion_noise"}


# ══════════════════════════════════════════════════════════════════
# I/O spec dataclasses
# ══════════════════════════════════════════════════════════════════


@dataclass
class ObsInputSpec:
    """Read an observation term from a group, frame-stacked on the deployment side."""

    group_name: str
    term_name: str


@dataclass
class NoiseInputSpec:
    """Sample fresh Gaussian noise each step for a non-observation model input."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class ActionOutputSpec:
    """Route a model output to the action applied through the WBC action term via ``env.step()``."""


# ══════════════════════════════════════════════════════════════════
# RobomimicStandLeappDeploymentEnv
# ══════════════════════════════════════════════════════════════════


class RobomimicStandLeappDeploymentEnv:
    """Runs the LEAPP-exported RobomimicStandPolicy in an Isaac Lab Arena scene.

    The environment self-configures ``num_envs`` and the frame-stack length from the exported
    model's input shapes, reads observations through the ObservationManager and frame-stacks them,
    runs the LEAPP model, and applies the resulting 23-dim action through the WBC action term.
    """

    def __init__(self, cfg: Any, leapp_yaml_path: str):
        """Initialize the deployment environment.

        Args:
            cfg: A composed Arena ``ManagerBasedRLEnvCfg`` (e.g. from ``ArenaEnvBuilder``) with
                non-concatenated observation terms and cameras enabled.
            leapp_yaml_path: Path to the LEAPP ``.yaml`` pipeline description.
        """
        self._is_closed = False
        self._leapp_yaml_path = leapp_yaml_path
        self._step_count = 0

        # ── LEAPP InferenceManager ────────────────────────────────
        self.inference = InferenceManager(leapp_yaml_path)
        if len(self.inference.nodes) != 1:
            raise ValueError(
                f"Expected a single-node LEAPP graph, got nodes {list(self.inference.nodes)}."
                " This deployment env only supports the RobomimicStandPolicy export."
            )
        self._node_name = next(iter(self.inference.nodes))
        node = self.inference.nodes[self._node_name]
        self._model_device = node.device
        self._input_dtypes = {d["name"]: map_to_torch_dtype(d["dtype"]) for d in node.input_descriptions}
        self._input_shapes = {
            d["name"]: tuple(json.loads(d["shape"]) if isinstance(d["shape"], str) else d["shape"])
            for d in node.input_descriptions
        }
        self._obs_input_names = [k.split("/", 1)[1] for k in self.inference.inputs if k.split("/", 1)[1] not in _EXTERNAL_INPUT_NAMES]
        num_envs, self._frame_stack = self._read_batch_and_frame_stack()

        # ── Manager-based env (obs + WBC action managers, decimation, reset, events) ──
        # Built directly (not via gym.make); variation_recorder defaults to None -- deployment
        # doesn't record variations, and the metrics/episode managers self-build from the cfg.
        cfg.scene.num_envs = num_envs
        self.env = IsaacLabArenaManagerBasedRLEnv(cfg)

        # ── Frame-stack buffer (cross-call state the exported graph leaves out) ──
        self._obs_history: dict[str, torch.Tensor] | None = None
        self._pending_reset: list[int] = []
        self._last_obs: dict[str, Any] = {}

        # ── Action-chunk replay ───────────────────────────────────
        # The exported model returns a (num_envs, Ta, 23) chunk; serve one action per step and only
        # re-run the diffusion model when the chunk is exhausted (so it runs once per Ta steps).
        self._action_chunk: torch.Tensor | None = None
        self._chunk_index: int = 0

        # ── Resolve I/O mappings ──────────────────────────────────
        self._input_mapping: dict[str, ObsInputSpec | NoiseInputSpec] = {}
        self._output_mapping: dict[str, ActionOutputSpec] = {}
        self._resolve_io()

        logger.info(
            "RobomimicStandLeappDeploymentEnv ready — node '%s', %d inputs, %d outputs, frame_stack=%d, num_envs=%d",
            self._node_name,
            len(self._input_mapping),
            len(self._output_mapping),
            self._frame_stack,
            num_envs,
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def physics_dt(self) -> float:
        return self.env.physics_dt

    @property
    def step_dt(self) -> float:
        return self.env.step_dt

    @property
    def device(self) -> str:
        return self.env.device

    # ── Setup helpers ─────────────────────────────────────────────

    def _read_batch_and_frame_stack(self) -> tuple[int, int]:
        """Read ``(num_envs, frame_stack)`` from an observation input's leading dimensions.

        Uses an observation input, since external inputs (e.g. noise) have a different layout.
        """
        shape = self._input_shapes[self._obs_input_names[0]]
        if len(shape) < 2:
            raise ValueError(f"Expected observation inputs shaped (num_envs, frame_stack, ...), got {shape}.")
        return int(shape[0]), int(shape[1])

    # ── I/O Resolution ────────────────────────────────────────────

    def _resolve_io(self):
        """Build ``_input_mapping`` and ``_output_mapping`` from the model's I/O names.

        Each model input is resolved to the ObservationManager term that produces it, or to a
        runtime-sampled external input; the single action output is applied via ``env.step()``.
        """
        # obs term -> group, from the ObservationManager (parallels the original connection parse)
        term_to_group = {
            term: group for group, terms in self.env.observation_manager.active_terms.items() for term in terms
        }

        for key in self.inference.inputs:
            name = key.split("/", 1)[1]
            if name in _EXTERNAL_INPUT_NAMES:
                self._input_mapping[key] = NoiseInputSpec(
                    shape=self._input_shapes[name], dtype=self._input_dtypes[name]
                )
            elif name in term_to_group:
                self._input_mapping[key] = ObsInputSpec(group_name=term_to_group[name], term_name=name)
            else:
                raise KeyError(
                    f"LEAPP input '{name}' is neither an external input {sorted(_EXTERNAL_INPUT_NAMES)}"
                    f" nor an observation term. Available terms: {sorted(term_to_group)}."
                )

        for key in self.inference.outputs:
            self._output_mapping[key] = ActionOutputSpec()

    # ── Read / Write ──────────────────────────────────────────────

    def _read_inputs(self) -> dict[str, torch.Tensor]:
        """Read all mapped inputs: the current frame-stacked observations and fresh noise.

        Assumes the frame-stack buffer has already been rolled for this step (done in ``step``).

        Returns:
            A mapping from ``"node_name/tensor_name"`` to the tensor passed to the LEAPP pipeline.
        """
        inputs: dict[str, torch.Tensor] = {}
        for key, spec in self._input_mapping.items():
            name = key.split("/", 1)[1]
            if isinstance(spec, ObsInputSpec):
                value = self._obs_history[spec.term_name]
            else:  # NoiseInputSpec
                value = torch.randn(spec.shape, device=self._model_device, dtype=spec.dtype)
            inputs[key] = value.to(device=self._model_device, dtype=self._input_dtypes[name])
        return inputs

    def _write_outputs(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the assembled action chunk ``(num_envs, Ta, 23)`` from the model outputs.

        Args:
            outputs: Model outputs keyed by ``"node_name/tensor_name"`` from ``run_policy()``.
        """
        (action_key,) = self._output_mapping  # the RobomimicStandPolicy export has a single action output
        return outputs[action_key]

    def _update_history(self, obs: dict[str, Any]) -> None:
        """Roll the frame-stack buffer with the latest observation (mirrors RobomimicStandPolicy)."""
        raw = {
            spec.term_name: obs[spec.group_name][spec.term_name].clone()
            for spec in self._input_mapping.values()
            if isinstance(spec, ObsInputSpec)
        }
        if self._obs_history is None:
            self._obs_history = {k: v.unsqueeze(1).repeat_interleave(self._frame_stack, dim=1) for k, v in raw.items()}
        else:
            for k, v in raw.items():
                self._obs_history[k] = torch.roll(self._obs_history[k], shifts=-1, dims=1)
                self._obs_history[k][:, -1] = v
            # Re-seed history for envs that terminated on the previous step (obs is now post-reset).
            if self._pending_reset:
                ids = self._pending_reset
                for k, v in raw.items():
                    self._obs_history[k][ids] = v[ids].unsqueeze(1)
                self._pending_reset = []

    # ── Public API ────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """Reset the scene, inference state, frame-stack buffer, and action chunk.

        Returns:
            The initial observation (frame-stack init is deferred to the first ``step``).
        """
        obs, _ = self.env.reset()
        self.inference.reset()
        self._obs_history = None
        self._pending_reset = []
        self._action_chunk = None
        self._chunk_index = 0
        self._last_obs = obs
        return obs

    def step(self) -> torch.Tensor:
        """Run one step: roll frame-stack -> (re-plan if the chunk is exhausted) -> apply action.

        The exported model returns a ``(num_envs, Ta, 23)`` action chunk; one action is served per
        step and the diffusion model is re-run only every ``Ta`` steps.

        Returns:
            The action applied this step, shape ``(num_envs, 23)``.
        """
        self._step_count += 1

        # 1. Roll the frame-stack buffer with the current observation (every step, to stay fresh).
        self._update_history(self._last_obs)

        # 2. Re-plan (denoise a new chunk) only when the current chunk is exhausted.
        if self._action_chunk is None or self._chunk_index >= self._action_chunk.shape[1]:
            inputs = self._read_inputs()
            with torch.inference_mode():
                outputs = self.inference.run_policy(inputs)
            self._action_chunk = self._write_outputs(outputs)  # (num_envs, Ta, 23)
            self._chunk_index = 0

        # 3. Serve one action from the chunk and apply it (WBC term + physics via env.step).
        action = self._action_chunk[:, self._chunk_index].to(device=self.env.device)
        self._chunk_index += 1
        obs, _, terminated, truncated, _ = self.env.step(action)
        self._last_obs = obs

        # 4. On termination, re-seed frame-stack history next update and force a re-plan.
        done = (terminated | truncated).nonzero().flatten().tolist()
        if done:
            self._pending_reset = sorted(set(self._pending_reset).union(done))
            self._action_chunk = None
        return action

    def close(self):
        """Clean up the environment and release simulator-owned resources."""
        if not self._is_closed:
            self.env.close()
            self._is_closed = True
