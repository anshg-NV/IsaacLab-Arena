# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check recorded joint-action conversion and chunk-relative training labels."""

import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.utils.relative_action_utils import joint_relative_to_absolute

from isaaclab_arena.scripts.imitation_learning import convert_dataset_to_joint_actions as conversion
from isaaclab_arena.scripts.imitation_learning.relative_action_stats import collect_joint_deltas, report_joint
from isaaclab_arena_g1.g1_env.g1_arm_joints import G1_ARM_JOINT_NAMES
from isaaclab_arena_g1.g1_env.g1_supplemental_info import G1SupplementalInfo


def test_convert_dataset_preserves_source_and_uses_commanded_joints(tmp_path, monkeypatch):
    source, output = tmp_path / "eef.hdf5", tmp_path / "joint.hdf5"
    actions = np.zeros((16, 23), dtype=np.float32)
    actions[:, 0] = -0.5
    commands = np.zeros((16, 43), dtype=np.float32)
    for name, (lower, upper) in G1SupplementalInfo().joint_limits.items():
        if name in G1_ARM_JOINT_NAMES:
            commands[:, conversion.G1_SIM_JOINT_NAMES.index(name)] = (lower + upper) / 2
    for name, value in conversion._CLOSED_LEFT_HAND.items():
        commands[:, conversion.G1_SIM_JOINT_NAMES.index(name)] = value
    # A settled/randomized first observation must not be mistaken for a joint-order error.
    measured = np.full((16, 43), 0.123, dtype=np.float32)
    with h5py.File(source, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["num_samples"] = 16
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("processed_actions", data=commands)
        demo.create_dataset("obs/robot_joint_pos", data=measured)
        handle.create_dataset("mask/train", data=np.array([b"demo_0"]))
    monkeypatch.setattr(sys, "argv", ["convert", "--input", str(source), "--output", str(output)])
    conversion.main()
    with h5py.File(source, "r") as original, h5py.File(output, "r") as converted:
        assert "eef_actions" not in original["data/demo_0"]
        np.testing.assert_array_equal(original["data/demo_0/actions"], actions)
        demo = converted["data/demo_0"]
        np.testing.assert_array_equal(demo["eef_actions"], actions)
        np.testing.assert_array_equal(demo["actions"][:, :2], actions[:, :2])
        arm_ids = conversion.joint_indices(G1_ARM_JOINT_NAMES)
        np.testing.assert_array_equal(demo["actions"][:, 2:], commands[:, arm_ids])
        np.testing.assert_array_equal(demo["obs/left_arm_joint_pos"], measured[:, arm_ids[:7]])
        np.testing.assert_array_equal(demo["obs/right_arm_joint_pos"], measured[:, arm_ids[7:]])
        np.testing.assert_array_equal(converted["mask/train"], original["mask/train"])
    # A demo exactly one prediction horizon long still contributes one training window per arm.
    deltas = collect_joint_deltas(str(output), (2, 9), ("left_arm_joint_pos", "right_arm_joint_pos"), 2, 16, 1)
    assert deltas.shape == (32, 7)
    np.testing.assert_allclose(deltas[:16], commands[:, arm_ids[:7]] - measured[:, arm_ids[:7]])


def test_relative_joint_training_and_inference_use_the_latest_observation():
    policy = object.__new__(DiffusionPolicyUNet)
    policy.rel_joint_mode = True
    policy.rel_config = SimpleNamespace(
        anchor_joint_pos_keys=["left_arm_joint_pos", "right_arm_joint_pos"], joint_indices=[2, 9], joint_scale=1.1,
    )
    policy.algo_config = SimpleNamespace(horizon=SimpleNamespace(observation_horizon=2, action_horizon=2, prediction_horizon=3))
    policy.action_check_done = False
    policy.device = torch.device("cpu")
    obs = {"left_arm_joint_pos": torch.zeros(2, 2, 7), "right_arm_joint_pos": torch.zeros(2, 2, 7)}
    obs["left_arm_joint_pos"][:, -1] = 1.5
    obs["right_arm_joint_pos"][:, -1] = -1.5
    actions = torch.zeros(2, 3, 16)
    actions[..., 0] = -0.5
    actions[..., 2:9] = 1.72
    actions[..., 9:] = -1.83
    result = policy.process_batch_for_training({"obs": obs, "actions": actions})
    torch.testing.assert_close(result["actions"][..., :2], actions[..., :2])
    torch.testing.assert_close(result["actions"][..., 2:9], torch.full((2, 3, 7), 0.2))
    torch.testing.assert_close(result["actions"][..., 9:], torch.full((2, 3, 7), -0.3))
    restored = joint_relative_to_absolute(result["actions"], policy._relative_joint_anchors(obs), [2, 9], 1.1)
    torch.testing.assert_close(restored, actions)


def test_joint_stats_print_a_usable_relative_config(capsys):
    report_joint(np.zeros((32, 7)))
    output = capsys.readouterr().out
    assert '"anchor_joint_pos_keys": ["left_arm_joint_pos", "right_arm_joint_pos"]' in output
    assert '"joint_indices": [2, 9]' in output
    assert '"joint_scale": 0.05' in output
