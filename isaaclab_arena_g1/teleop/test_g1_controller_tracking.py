# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real G1 retargeting graph with DeviceIO controller samples."""

import numpy as np

import pytest

pytest.importorskip("isaacteleop")

from isaacteleop import schema
from isaacteleop.retargeting_engine.interface import ComputeContext, ExecutionEvents
from isaacteleop.retargeting_engine.interface.tensor_group import TensorGroup

from isaaclab_arena_g1.teleop.g1_pink_locomanipulation_pipeline import _build_g1_pink_locomanipulation_pipeline


def _controller(position=(0.2, 0.3, 0.8), quaternion=(0.0, 0.0, 0.0, 1.0), valid=True):
    grip = schema.ControllerPose(schema.Pose(schema.Point(*position), schema.Quaternion(*quaternion)), valid)
    # Aim may be unavailable even while the grip pose and buttons are usable.
    aim = schema.ControllerPose(schema.Pose(), False)
    buttons = schema.ControllerInputState(False, False, False, False, 0.0, 0.0, 0.8, 0.7)
    return schema.ControllerSnapshotTrackedT(schema.ControllerSnapshot(grip, aim, buttons))


def _step(pipeline, left, right, context=None):
    inputs = {}
    for leaf in pipeline.get_leaf_nodes():
        inputs[leaf.name] = {}
        for name, spec in leaf.input_spec().items():
            group = TensorGroup(spec)
            if leaf.name == "world_T_anchor":
                transform = np.eye(4, dtype=np.float32)
                transform[:3, 3] = [1.0, 2.0, 3.0]
                group[0] = transform
            else:
                group[0] = left if name.endswith("left") else right
            inputs[leaf.name][name] = group
    return np.from_dlpack(pipeline.execute_pipeline(inputs, context)["action"][0]).copy()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "bad_sample",
    [
        {"valid": False, "quaternion": (0.0, 0.0, 0.0, 0.0)},
        {"valid": False, "position": (99.0, 99.0, 99.0)},
        {"quaternion": (0.0, 0.0, 0.0, 0.0)},
        {"quaternion": (float("nan"), 0.0, 0.0, 1.0)},
        {"position": (float("nan"), 0.0, 0.0)},
    ],
)
def test_invalid_grip_holds_wrist_and_recovers(side, bad_sample):
    pipeline = _build_g1_pink_locomanipulation_pipeline()
    good = _controller()
    initial = _step(pipeline, good, good)
    np.testing.assert_allclose(initial[2:5], [1.2, 2.3, 3.8])
    bad = _controller(**bad_sample)
    for _ in range(5):
        held = _step(pipeline, bad if side == "left" else good, bad if side == "right" else good)
        assert np.isfinite(held).all()
        np.testing.assert_array_equal(held, initial)
    moved = _controller(position=(0.4, 0.5, 1.0))
    recovered = _step(pipeline, moved, moved)
    np.testing.assert_allclose(recovered[2:5], [1.4, 2.5, 4.0])
    np.testing.assert_allclose(recovered[9:12], [1.4, 2.5, 4.0])
    np.testing.assert_array_equal(recovered[:2], initial[:2])


def test_startup_without_tracking_then_valid_controllers():
    pipeline = _build_g1_pink_locomanipulation_pipeline()
    invalid = _controller(valid=False, quaternion=(0.0, 0.0, 0.0, 0.0))
    for _ in range(10):
        action = _step(pipeline, invalid, invalid)
        assert np.isfinite(action).all()
        np.testing.assert_array_equal(action[5:9], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(action[12:16], [0.0, 0.0, 0.0, 1.0])
    action = _step(pipeline, _controller(), _controller())
    np.testing.assert_allclose(action[2:5], [1.2, 2.3, 3.8])


def test_reset_during_tracking_loss_clears_previous_episode_pose():
    pipeline = _build_g1_pink_locomanipulation_pipeline()
    _step(pipeline, _controller(), _controller())
    invalid = _controller(valid=False, quaternion=(0.0, 0.0, 0.0, 0.0))
    reset = ComputeContext(execution_events=ExecutionEvents(reset=True))
    action = _step(pipeline, invalid, invalid, reset)
    np.testing.assert_array_equal(action[2:9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(action[9:16], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    recovered = _step(pipeline, _controller(), _controller())
    np.testing.assert_allclose(recovered[2:5], [1.2, 2.3, 3.8])
