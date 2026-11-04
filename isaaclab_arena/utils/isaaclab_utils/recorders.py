# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class PreStepTimestampsRecorder(RecorderTerm):
    """Record host clocks alongside each pre-step action and observation sample.

    Each ``timestamps`` field becomes an int64 array of shape (T,) per episode.
    ``unix_ns`` is time since the Unix epoch [ns]; ``monotonic_ns`` has an arbitrary
    origin and should be differenced to measure sample intervals [ns]. These mark
    recorder sampling, not camera exposure or the later episode flush to disk.
    """

    def record_pre_step(self) -> tuple[str, dict[str, torch.Tensor]]:
        """Return one pair of host timestamps per environment for this recorded step."""
        unix_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        return "timestamps", {
            "unix_ns": torch.full((self._env.num_envs,), unix_ns, dtype=torch.int64, device="cpu"),
            "monotonic_ns": torch.full((self._env.num_envs,), monotonic_ns, dtype=torch.int64, device="cpu"),
        }


@configclass
class PreStepTimestampsRecorderCfg(RecorderTermCfg):
    """Configuration for recording per-sample host timestamps."""

    class_type: type[RecorderTerm] = PreStepTimestampsRecorder


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


class PostStepFlatPolicyActionObservationRecorder(RecorderTerm):
    """Recorder term that records the ``action`` observation group at the end of each step.

    Mirrors the locomanip mimic patch's post-step action recorder, but no-ops on envs
    whose policy does not expose an ``action`` observation group so it can be safely
    enabled for any task.
    """

    def record_post_step(self):
        obs_buf = getattr(self._env, "obs_buf", None)
        if not isinstance(obs_buf, dict) or "action" not in obs_buf:
            return None, None
        return "action", obs_buf["action"]


@configclass
class PostStepFlatPolicyActionObservationRecorderCfg(RecorderTermCfg):
    """Configuration for the post-step ``action`` observation recorder term."""

    class_type: type[RecorderTerm] = PostStepFlatPolicyActionObservationRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder manager extended with arena-specific recorder terms."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()
    record_post_step_flat_policy_action_observations = PostStepFlatPolicyActionObservationRecorderCfg()
