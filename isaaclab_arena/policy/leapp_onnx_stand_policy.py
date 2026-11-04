# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a complete single-environment LEAPP graph through the native action manager."""

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.deployment_timing import DeploymentTiming, TIMING_PROFILES
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


_OBS_SHAPES = {
    "left_eef_pos": (3,),
    "left_eef_quat": (4,),
    "left_wrist_pose_pelvis_frame": (4, 4),
    "right_eef_pos": (3,),
    "right_eef_quat": (4,),
    "right_wrist_pose_pelvis_frame": (4, 4),
    "robot_head_cam_rgb": (240, 320, 3),
}


@dataclass
class LeappOnnxStandPolicyCfg(PolicyCfg):
    """Configuration for an eight-input graph or explicit causal20 extension."""

    onnx_model_path: str
    device: str = "cuda:0"
    timing_profile: str = "instant"
    feedback_contract_path: str | None = None
    feedback_contract_sha256: str | None = None
    feedback_instance_id: str | None = None
    feedback_availability_hypothesis: str | None = None


def _feedback_contract(config: LeappOnnxStandPolicyCfg) -> dict | None:
    """Read an independently pinned causal20 schema; never infer one from a port name."""
    if config.feedback_contract_path is None:
        assert all(value is None for value in (config.feedback_contract_sha256, config.feedback_instance_id,
                                              config.feedback_availability_hypothesis))
        return None
    from robomimic.models.feedback_residual_policy_nets import feedback_contract, _same_record
    from isaaclab_arena.policy.native_feedback import ZERO_DELAY_HYPOTHESIS

    assert type(config.feedback_instance_id) is str and config.feedback_instance_id
    assert config.feedback_availability_hypothesis == ZERO_DELAY_HYPOTHESIS
    assert config.timing_profile == "fresh240", "Feedback requires supported causal timing"
    payload = Path(config.feedback_contract_path).resolve(strict=True).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == config.feedback_contract_sha256, "Feedback contract hash differs"
    contract = json.loads(payload)
    _same_record(contract, feedback_contract(**contract["age_limits_ns"]))
    return contract


@register_policy
class LeappOnnxStandPolicy(PolicyBase[LeappOnnxStandPolicyCfg]):
    """Run the exported final 23-D action directly, without a second residual or assembly."""

    name = "leapp_onnx_stand"

    def __init__(self, config: LeappOnnxStandPolicyCfg):
        import onnx
        import onnxruntime as ort

        super().__init__(config)
        self._feedback_contract = _feedback_contract(config)
        self._native_feedback, self._feedback_tick, self._feedback_reset_ids = None, 0, ()
        assert config.timing_profile in TIMING_PROFILES
        device = torch.device(config.device)
        assert device.type == "cuda", "Full-graph evaluation requires CUDA ONNX Runtime"
        assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA ONNX Runtime is unavailable"
        model_path = Path(config.onnx_model_path).resolve()
        assert model_path.is_file(), f"Missing ONNX graph: {model_path}"
        model = onnx.load(str(model_path), load_external_data=False)
        assert [output.name for output in model.graph.output] == ["action"], (
            "This consumer requires the assembled action output, not a raw base chunk"
        )
        locations = {
            entry.value for initializer in model.graph.initializer for entry in initializer.external_data
            if entry.key == "location"
        }
        artifacts = [model_path]
        for location in sorted(locations):
            external_path = (model_path.parent / location).resolve()
            assert model_path.parent in external_path.parents, "External graph data must stay beside the model"
            assert external_path.is_file(), f"Missing external graph data: {external_path}"
            artifacts.append(external_path)
        artifact_identity = []
        for path in artifacts:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            artifact_identity.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": device.index or 0, "use_tf32": 0}),
                       "CPUExecutionProvider"],
        )
        assert self._session.get_providers()[0] == "CUDAExecutionProvider", "CUDA session silently fell back to CPU"
        self._session.disable_fallback()
        inputs = {item.name: item for item in self._session.get_inputs()}
        expected_inputs = set(_OBS_SHAPES) | {"diffusion_noise"}
        if self._feedback_contract is not None:
            expected_inputs.add("residual_feedback")
        assert set(inputs) == expected_inputs, "Unexpected full-graph input contract"
        if self._feedback_contract is not None:
            assert tuple(inputs["residual_feedback"].shape) == (1, 20)
            assert inputs["residual_feedback"].type == "tensor(float)"
        for name, shape in _OBS_SHAPES.items():
            assert tuple(inputs[name].shape) == (1, 2, *shape), f"Unexpected input shape: {name}"
            expected_type = "tensor(uint8)" if name == "robot_head_cam_rgb" else "tensor(float)"
            assert inputs[name].type == expected_type, f"Unexpected input dtype: {name}"
        assert tuple(inputs["diffusion_noise"].shape) == (1, 16, 21)
        assert inputs["diffusion_noise"].type == "tensor(float)"
        outputs = self._session.get_outputs()
        assert len(outputs) == 1 and tuple(outputs[0].shape) == (1, 6, 23)
        assert outputs[0].type == "tensor(float)"
        assert outputs[0].name == "action", "Runtime output differs from the assembled action graph"
        self._output_name = outputs[0].name
        self._provenance = {
            "policy_type": self.name,
            "artifact_kind": "full_onnx_action_graph",
            "onnx_model_path": str(model_path),
            "artifacts": artifact_identity,
            "runtime": {
                "onnxruntime": ort.__version__, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "providers": self._session.get_providers(), "provider_options": self._session.get_provider_options(),
                "graph_optimization": "BASIC", "runtime_fallback_disabled": True,
                "cpu_shape_operator_placement_allowed": True,
            },
            "input_contract": "Seven raw observations stacked twice plus fresh initial diffusion noise",
            "output_contract": "Six final 23-D actions consumed directly; no extra assembly or residual correction",
        }
        if self._feedback_contract is not None:
            self._provenance["input_contract"] += " plus one immutable causal20 packet at actual submission"
            self._provenance["feedback"] = {
                "contract": self._feedback_contract,
                "contract_path": str(Path(config.feedback_contract_path).resolve()),
                "contract_sha256": config.feedback_contract_sha256,
                "instance_id": config.feedback_instance_id,
                "availability_hypothesis": config.feedback_availability_hypothesis,
                "boundary": "Named simulated q and previously accepted native action row; not physical sensing or actuation",
            }
        self.reset()

    def get_action(self, env, observation) -> torch.Tensor:
        """Return one final action, applying the selected causal timing approximation."""
        assert env.unwrapped.num_envs == 1, "Full-graph evaluation currently requires one environment"
        assert self._pending_camera is None, "Previous action was not recorded as consumed"
        if self._native_feedback is not None:
            self._native_feedback._check(self._feedback_tick, "ready")
            assert not self._feedback_reset_ids, "Reset terminal scheduler slots before another action"
        policy_obs = observation.get("policy", observation)
        camera = observation.get("camera_obs", {})["robot_head_cam_rgb"]
        assert camera.dtype == torch.uint8 and tuple(camera.shape) == (1, 240, 320, 3)
        pixels = camera.float().flatten(start_dim=1)
        variance, mean = torch.var_mean(pixels, dim=1, correction=0)
        camera_stats = torch.stack((mean, variance.sqrt()), dim=-1)
        assert torch.isfinite(camera_stats).all() and (camera_stats > 2.0).all(), "Invalid or blank actor RGB frame"
        obs = {key: policy_obs[key].clone().float() for key in _OBS_SHAPES if key != "robot_head_cam_rgb"}
        obs["robot_head_cam_rgb"] = camera.clone()
        for key, value in obs.items():
            assert tuple(value.shape) == (1, *_OBS_SHAPES[key]), f"Unexpected raw observation shape: {key}"
            assert torch.isfinite(value).all(), f"Nonfinite raw observation: {key}"
        if self.config.timing_profile != "instant":
            if self._timing is None:
                self._timing = DeploymentTiming(
                    self.config.timing_profile, 1, env.unwrapped.step_dt,
                    **({"track_consumption": True} if self._feedback_contract is not None else {}),
                )
                if self._feedback_contract is not None:
                    from isaaclab_arena.policy.native_feedback import NativeFeedback

                    names, positions = self._native_joint_state(env)
                    self._native_feedback = NativeFeedback(
                        self._timing, names, instance_id=self.config.feedback_instance_id,
                        timing_profile=self.config.timing_profile,
                        joint_delay_hypothesis=self.config.feedback_availability_hypothesis,
                        **self._feedback_contract["age_limits_ns"],
                    )
                    self._native_feedback.initialize(positions, global_tick=self._feedback_tick)
            obs = self._timing.observe(obs)
        if self._history is None:
            self._history = {key: value.unsqueeze(1).repeat_interleave(2, dim=1) for key, value in obs.items()}
        else:
            for key, value in obs.items():
                self._history[key] = torch.stack((self._history[key][:, -1], value), dim=1)
        source = self._history
        if self._timing is not None:
            source = self._timing.source(source)
            self._timing.deliver(self._queues, None)
        needs_inference = self._timing.needs_inference() if self._timing is not None else not self._queues[0]
        if needs_inference:
            feed = {key: value.detach().cpu().numpy() for key, value in source.items()}
            if self._native_feedback is not None:
                (packet,) = self._native_feedback.submission_packets([0], global_tick=self._feedback_tick)
                feed["residual_feedback"] = np.array([packet.features], dtype=np.float32)
            feed["diffusion_noise"] = torch.randn((1, 16, 21), device=env.unwrapped.device).cpu().numpy()
            output = self._session.run([self._output_name], feed)[0]
            assert output.shape == (1, 6, 23) and output.dtype == np.float32 and np.isfinite(output).all()
            chunk = torch.from_numpy(output).to(device=env.unwrapped.device)[0]
            if self._timing is None or self._timing.submit(0, chunk, None, None):
                self._queues[0].extend(chunk)
        if self._timing is None:
            action = self._queues[0].popleft().unsqueeze(0)
        else:
            action = self._timing.consume(self._queues)
            self._timing.advance()
        if self._native_feedback is not None:
            self._native_feedback.prepare_step(action[:, :2], global_tick=self._feedback_tick)
        self._pending_camera = camera_stats.detach().cpu()
        return action

    @staticmethod
    def _native_joint_state(env) -> tuple[list[str], torch.Tensor]:
        data = env.unwrapped.scene["robot"].data
        positions = data.joint_pos
        if not isinstance(positions, torch.Tensor):
            import warp as wp

            positions = wp.to_torch(positions)
        return list(data.joint_names), positions

    def record_native_step(self, env, done_ids: torch.Tensor) -> None:
        """Accept only after env.step returns successfully, before episode recording/reset."""
        if self._feedback_contract is None:
            return
        assert self._native_feedback is not None and self._pending_camera is not None
        names, positions = self._native_joint_state(env)
        assert tuple(names) == self._native_feedback._names, "Native joint order changed"
        ids = done_ids.flatten().tolist()
        self._native_feedback.accept_step(ids, positions, global_tick=self._feedback_tick)
        self._feedback_tick += 1
        self._feedback_reset_ids = tuple(ids)

    def record_evaluation_step(self) -> None:
        """Count an action and camera sample only after the native step succeeds."""
        assert self._pending_camera is not None, "No unrecorded evaluation action"
        if self._native_feedback is not None:
            self._native_feedback._check(self._feedback_tick, "ready")
        self._camera_stats.append(self._pending_camera)
        self._pending_camera = None

    def evaluation_metrics(self) -> dict:
        """Return graph provenance and camera validity; graph-internal correction is unobservable."""
        assert self._pending_camera is None and self._camera_stats
        camera = torch.cat(self._camera_stats).double()
        return {
            **self._provenance, "format": 1, "num_envs": 1, "control_steps": len(self._camera_stats),
            "timing_profile": self.config.timing_profile,
            "timing_parameters": TIMING_PROFILES[self.config.timing_profile],
            "timing_startup": "instant" if self._timing is None else "one immediate prefilled chunk per reset",
            "residual_model_path": None, "rms_mm": None, "per_axis_rms_mm": None, "p95_abs_mm": None,
            "component_count": None, "sum_squared_mm2": None, "saturation_count": None,
            "saturation_threshold_mm": None, "saturation_frequency": None,
            "definition": (
                "Correction RMS is unavailable: the full graph exposes final actions, not its base/residual split."
            ),
            "camera_validity": {
                "frame_count": len(self._camera_stats), "all_finite": True, "all_nonblank": True,
                "min_mean": float(camera[:, 0].min()), "max_mean": float(camera[:, 0].max()),
                "min_std": float(camera[:, 1].min()), "max_std": float(camera[:, 1].max()),
                "minimum_mean_std_threshold_dn": 2.0,
                "sampling": "Raw native RGB on each consumed tick; timing profiles subsequently sample and hold it.",
            },
        }

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Discard pending inference and pad the next history from the new episode."""
        if self._native_feedback is not None:
            assert env_ids is not None, "Full feedback reset requires a new policy instance"
            self._native_feedback._check(self._feedback_tick, "ready")
            ids = tuple(env_ids.flatten().tolist())
            assert not ids or ids == self._feedback_reset_ids, "Reset must match accepted terminal slots"
        if env_ids is not None:
            ids = env_ids.flatten().tolist()
            if not ids:
                return
            assert ids == [0], "Single-environment graph can only reset environment zero"
            assert self._pending_camera is None, "Record the consumed action before resetting"
            if self._timing is not None:
                self._timing.reset(ids)
            self._feedback_reset_ids = ()
        else:
            self._timing = None
            self._camera_stats = []
        self._history = None
        self._queues = [deque()]
        self._pending_camera = None

    def close(self) -> None:
        """Release the owned inference session."""
        self._session = None
