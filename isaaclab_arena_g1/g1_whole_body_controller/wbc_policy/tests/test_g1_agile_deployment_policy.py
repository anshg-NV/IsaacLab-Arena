# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check the deployment graph contract and independent reset histories without simulation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.g1_agile_deployment_policy import G1AgileDeploymentPolicy


@pytest.fixture
def policy():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/g1_agile_deployment.yaml").read_text())
    model = next(iter(config["models"].values()))
    names = next(v for v in model["inputs"] if v["name"] == "joint_pos")["element_names"][0]
    controlled = next(v for v in model["outputs"] if v["name"] == "action_joint_pos")["element_names"][0]
    # Deliberately permute both WBC and lower-body order to exercise the name mappings.
    order = {name: i for i, name in enumerate(reversed(names))}
    lower = [order[name] for name in reversed(controlled + ["waist_yaw_joint"])]
    robot = SimpleNamespace(wbc_g1_joints_order=order, get_joint_group_indices=lambda _: lower)
    policy = G1AgileDeploymentPolicy(
        robot, "config/g1_agile_deployment.yaml", str(root / "models/agile_deployment/agile_velocity.onnx"), 3,
    )
    rng = np.random.default_rng(29)
    policy.set_observation({
        "floating_base_pose": np.tile([0., 0., .73, 1., 0., 0., 0.], (3, 1)),
        "root_ang_vel_b": np.zeros((3, 3)),
        "q": rng.uniform(-.15, .15, (3, len(names))),
        "dq": np.zeros((3, len(names))),
    })
    return policy


def test_matches_direct_export_after_150_stationary_calls(policy):
    observation = policy.observation
    inputs = {name: value[:1].copy() for name, value in policy.state.items()}
    inputs.update(
        root_link_quat_w=observation["floating_base_pose"][:1, 3:7].astype(np.float32),
        root_ang_vel_b=observation["root_ang_vel_b"][:1].astype(np.float32),
        velocity_commands=np.zeros((1, 3), np.float32),
        joint_pos=observation["q"][:1, policy.input_indices].astype(np.float32),
        joint_vel=observation["dq"][:1, policy.input_indices].astype(np.float32),
    )
    for _ in range(151):
        outputs = dict(zip(policy.output_names, policy.session.run(policy.output_names, inputs)))
        for name in policy.state:
            inputs[name] = outputs[name + "_out"]
    action = policy.get_action()["body_action"]
    np.testing.assert_array_equal(action[0, policy.output_indices], outputs["action_joint_pos"][0])
    for name in policy.state:
        np.testing.assert_array_equal(policy.state[name][0], outputs[name + "_out"][0])
    np.testing.assert_array_equal(action[:, 0], 0)  # Uncontrolled waist yaw.


def test_partial_reset_does_not_advance_other_histories(policy):
    import copy

    policy.get_action()
    before = {name: value.copy() for name, value in policy.state.items()}
    policy.reset(np.array([1]))
    for name, value in policy.state.items():
        np.testing.assert_array_equal(value[[0, 2]], before[name][[0, 2]])
        assert not value[1].any()
    # Control instance shares only immutable session/config, with separate state arrays.
    control = copy.copy(policy)
    control.state = {name: value.copy() for name, value in before.items()}
    control._pending_init = np.zeros(3, dtype=bool)
    expected = control.get_action()["body_action"]
    actual = policy.get_action()["body_action"]
    np.testing.assert_array_equal(actual[[0, 2]], expected[[0, 2]])
    for name, value in policy.state.items():
        np.testing.assert_array_equal(value[[0, 2]], control.state[name][[0, 2]])
    assert not policy._pending_init.any()
    policy.reset(None)
    assert policy._pending_init.all() and not any(value.any() for value in policy.state.values())


def test_height_is_not_forwarded_and_velocity_is_preserved(policy):
    velocity = np.tile([.1, -.05, .03], (3, 1))
    policy.set_goal({"navigate_cmd": velocity, "base_height_command": np.full((3, 1), .72)})
    action = policy.get_action()["body_action"]
    policy.reset(None)
    policy.set_goal({"navigate_cmd": velocity, "base_height_command": np.full((3, 1), .50)})
    np.testing.assert_array_equal(action, policy.get_action()["body_action"])
    np.testing.assert_allclose(policy.velocity_commands, velocity)
    np.testing.assert_array_equal(policy.state["velocity_commands_history"][:, :-1], 0)
    assert "base_height_command" not in policy.input_names
