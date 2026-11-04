# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the unmodified velocity-only AGILE deployment export."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
import yaml

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy

if TYPE_CHECKING:
    import torch

    from isaaclab_arena_g1.g1_env.robot_model import RobotModel


class G1AgileDeploymentPolicy(WBCPolicy):
    """Use exported preprocessing, absolute joint targets, and independent per-env histories."""

    def __init__(
        self,
        robot_model: RobotModel,
        config_path: str,
        model_path: str,
        num_envs: int = 1,
        memory_init_steps: int = 150,
    ):
        parent = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((parent / config_path).read_text())
        model = next(iter(config["models"].values()))
        path = parent / model_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == model["parameters"]["sha256sum"]
        assert num_envs > 0 and memory_init_steps >= 0
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.num_envs = num_envs
        self.memory_init_steps = memory_init_steps
        self.input_names = [value.name for value in self.session.get_inputs()]
        self.output_names = [value.name for value in self.session.get_outputs()]
        self.state = {
            value["name"]: np.zeros((num_envs, *value["shape"][1:]), np.float32)
            for value in model["inputs"] if value.get("is_state")
        }
        input_names = next(value for value in model["inputs"] if value["name"] == "joint_pos")["element_names"][0]
        self.controlled_names = next(
            value for value in model["outputs"] if value["name"] == "action_joint_pos"
        )["element_names"][0]
        order = robot_model.wbc_g1_joints_order
        lower_indices = robot_model.get_joint_group_indices("lower_body")
        self.input_indices = [order[name] for name in input_names]
        self.output_indices = [lower_indices.index(order[name]) for name in self.controlled_names]
        self.num_lower = len(lower_indices)
        self.velocity_commands = np.zeros((num_envs, 3), np.float32)
        self._pending_init = np.ones(num_envs, dtype=bool)

    def reset(self, env_ids: np.ndarray | torch.Tensor | None) -> None:
        """Clear selected histories; initialize from their next fresh reset observation."""
        ids = env_ids.cpu().numpy() if hasattr(env_ids, "cpu") else env_ids
        for value in self.state.values():
            value[ids] = 0
        self.velocity_commands[ids] = 0
        self._pending_init[ids] = True

    def set_goal(self, goal: dict[str, np.ndarray]) -> None:
        """Set body velocity commands [m/s, m/s, rad/s]; this export has no height input."""
        if "navigate_cmd" in goal:
            command = np.asarray(goal["navigate_cmd"], dtype=np.float32)
            assert command.shape == (self.num_envs, 3)
            self.velocity_commands[:] = command

    def _infer(self, env_ids: np.ndarray, stationary: bool = False) -> np.ndarray:
        observation = self.observation
        inputs = dict(self.state)
        inputs.update(
            root_link_quat_w=observation["floating_base_pose"][:, 3:7].astype(np.float32),
            root_ang_vel_b=observation["root_ang_vel_b"].astype(np.float32),
            velocity_commands=np.zeros_like(self.velocity_commands) if stationary else self.velocity_commands,
            joint_pos=observation["q"][:, self.input_indices].astype(np.float32),
            joint_vel=observation["dq"][:, self.input_indices].astype(np.float32),
        )
        # Preserve the exact deployment bytes and their fixed batch dimension of one.
        rows = [
            self.session.run(self.output_names, {key: value[i:i + 1] for key, value in inputs.items()})
            for i in env_ids
        ]
        outputs = {name: np.concatenate([row[j] for row in rows]) for j, name in enumerate(self.output_names)}
        for name, value in self.state.items():
            value[env_ids] = outputs[name + "_out"]
        return outputs["action_joint_pos"]

    def get_action(self, time: float | None = None) -> dict[str, np.ndarray]:
        """Initialize pending histories without physics, then return absolute joint targets [rad]."""
        pending = np.flatnonzero(self._pending_init)
        if len(pending):
            for _ in range(self.memory_init_steps):
                self._infer(pending, stationary=True)
            self._pending_init[pending] = False
        action = np.zeros((self.num_envs, self.num_lower), np.float32)
        action[:, self.output_indices] = self._infer(np.arange(self.num_envs))
        return {"body_action": action}
