# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise timestamp alignment through the real recorder and HDF5 exporter, without simulation."""

import h5py
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderManager

from isaaclab_arena.utils.isaaclab_utils import recorders


@pytest.mark.parametrize("with_cameras", [False, True])
def test_timestamps_survive_export_pause_and_reset(tmp_path, with_cameras):
    actions = torch.zeros((2, 3))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        cfg=SimpleNamespace(env_name="timestamp_test", get_ep_meta=lambda: {}),
        sim=SimpleNamespace(is_playing=lambda: True),
        scene=SimpleNamespace(get_state=lambda **kwargs: {"joint_pos": actions}),
        action_manager=SimpleNamespace(
            action=actions,
            active_terms=["joints"],
            get_term=lambda name: SimpleNamespace(processed_actions=actions),
        ),
        obs_buf={"policy": {"joint_pos": actions}, "camera_obs": {"rgb": torch.zeros((2, 2, 2, 3))}},
    )
    cfg = recorders.ArenaEnvRecorderManagerCfg() if with_cameras else ActionStateRecorderManagerCfg()
    cfg.record_pre_step_timestamps = recorders.PreStepTimestampsRecorderCfg()
    cfg.dataset_export_dir_path = str(tmp_path)
    cfg.dataset_filename = "timestamps"
    manager = RecorderManager(cfg, env)

    def record_step(index):
        actions.fill_(index)
        # A backwards wall-clock adjustment must not affect the monotonic intervals.
        with (
            patch.object(recorders.time, "time_ns", return_value=1_800_000_000_000_000_003 - index * 20_000_000),
            patch.object(recorders.time, "monotonic_ns", return_value=1_000_000_000_000_007 + index * 20_000_000),
        ):
            manager.record_pre_step()
        manager.record_post_step()

    try:
        manager.record_post_reset([0, 1])
        for index in range(5):
            manager.set_recording_enabled(index != 2, [1])
            record_step(index)
        manager.export_episodes()
        manager.reset()
        manager.record_post_reset([0, 1])
        manager.set_recording_enabled(False, [1])
        record_step(5)
        record_step(6)
        manager.export_episodes([0])
    finally:
        manager.close()

    with h5py.File(tmp_path / "timestamps.hdf5", "r") as dataset:
        for name, indices in zip(("demo_0", "demo_1", "demo_2"), ([0, 1, 2, 3, 4], [0, 1, 3, 4], [5, 6])):
            demo = dataset[f"data/{name}"]
            indices = np.array(indices, dtype=np.int64)
            assert demo.attrs["num_samples"] == len(indices)
            for field, origin, step in (
                ("unix_ns", 1_800_000_000_000_000_003, -20_000_000),
                ("monotonic_ns", 1_000_000_000_000_007, 20_000_000),
            ):
                timestamps = demo[f"timestamps/{field}"]
                assert timestamps.dtype == np.dtype("int64")
                assert timestamps.shape == (len(indices),)
                np.testing.assert_array_equal(timestamps[:], origin + indices * step)
            np.testing.assert_array_equal(demo["actions"][:, 0], indices)
            np.testing.assert_array_equal(demo["obs/joint_pos"][:, 0], indices)
            if with_cameras:
                assert len(demo["camera_obs/rgb"]) == len(indices)
