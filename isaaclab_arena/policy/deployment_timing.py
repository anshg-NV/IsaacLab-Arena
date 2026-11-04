# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Causal, opt-in deployment timing approximations at the 20 ms control rate.

The first chunk is prefilled at reset. Subsequent inference takes 240 ms; the
recorded profiles additionally approximate the measured FIFO input backlog.
These are controlled interventions, not a reconstruction of unlogged callbacks.
"""

from collections import deque

import torch


TIMING_PROFILES = {
    "instant": {"service_ticks": 0, "queue_ticks": 0, "proprio_period": 1, "camera_period": 1},
    "fresh240": {"service_ticks": 12, "queue_ticks": 0, "proprio_period": 3, "camera_period": 2},
    "recorded420": {"service_ticks": 12, "queue_ticks": 9, "proprio_period": 3, "camera_period": 2},
    "recorded620": {"service_ticks": 12, "queue_ticks": 19, "proprio_period": 3, "camera_period": 2},
}


class DeploymentTiming:
    """Keep each environment's source observation and delayed chunk aligned."""

    def __init__(
        self, profile: str, num_envs: int, step_dt: float,
        *, track_consumption: bool = False,
    ) -> None:
        assert profile in TIMING_PROFILES and profile != "instant"
        assert abs(step_dt - 0.02) < 1e-6, "Timing profiles require a 20 ms control step"
        self.profile = dict(TIMING_PROFILES[profile])
        self.num_envs = num_envs
        self.sensor_cache: dict[str, torch.Tensor] = {}
        self.ticks = [0] * num_envs
        self.histories = [deque(maxlen=self.profile["queue_ticks"] + 1) for _ in range(num_envs)]
        self.pending = [None] * num_envs
        self.initialized = [False] * num_envs
        self.last_actions = [None] * num_envs
        self.last_corrections = [None] * num_envs
        self._consumption = [
            dict(generation=0, next_submission=0, pending=None, active=None, next_row=0, last=None)
            for _ in range(num_envs)
        ] if track_consumption else None
    def observe(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Sample-and-hold sensors before constructing the two-frame history."""
        for key, value in obs.items():
            period = self.profile["camera_period" if key == "robot_head_cam_rgb" else "proprio_period"]
            ids = [i for i, tick in enumerate(self.ticks) if tick % period == 0]
            if key not in self.sensor_cache:
                self.sensor_cache[key] = value.clone()
            else:
                self.sensor_cache[key][ids] = value[ids]
        return self.sensor_cache

    def source(self, stacked: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select a causal historical input; pad initial backlog with reset data."""
        for i in range(self.num_envs):
            self.histories[i].append({key: value[i].clone() for key, value in stacked.items()})
        return {key: torch.stack([history[0][key] for history in self.histories]) for key in stacked}

    def deliver(self, action_queues: list[deque], correction_queues: list[deque] | None) -> list[tuple]:
        """Deliver completed chunks, returning their matching residual inputs."""
        delivered = []
        for i, packet in enumerate(self.pending):
            if packet is None or packet[0] > self.ticks[i]:
                continue
            _, actions, inputs, corrections = packet
            action_queues[i].clear()
            action_queues[i].extend(actions)
            if correction_queues is not None:
                correction_queues[i].clear()
                correction_queues[i].extend(corrections)
            delivered.append((i, inputs))
            self.pending[i] = None
            if self._consumption is not None:
                state = self._consumption[i]
                state["active"], state["pending"] = state["pending"], None
                state["next_row"] = 0
        return delivered

    def needs_inference(self) -> list[int]:
        """Submit the next job immediately when the single server is available."""
        return [i for i, packet in enumerate(self.pending) if packet is None]

    def submit(self, i: int, actions: torch.Tensor, inputs: torch.Tensor | None,
               corrections: torch.Tensor | None) -> bool:
        """Queue a result, or return True for the one prefilled startup chunk."""
        assert self.pending[i] is None
        if self._consumption is not None:
            state = self._consumption[i]
            chunk = (state["next_submission"], len(actions))
            state["next_submission"] += 1
            state["pending" if self.initialized[i] else "active"] = chunk
        if not self.initialized[i]:
            self.initialized[i] = True
            return True
        self.pending[i] = (self.ticks[i] + self.profile["service_ticks"], actions, inputs, corrections)
        return False

    def consume(self, queues: list[deque], corrections: bool = False) -> torch.Tensor:
        """Replay chunk steps, then hold the last actually consumed target."""
        last = self.last_corrections if corrections else self.last_actions
        for i, queue in enumerate(queues):
            if queue:
                last[i] = queue.popleft()
                if self._consumption is not None and not corrections:
                    state = self._consumption[i]
                    ordinal, length = state["active"]
                    row = state["next_row"]
                    assert row < length, "Action queue no longer matches its submitted chunk"
                    state["last"] = {
                        "generation": state["generation"], "submission_ordinal": ordinal,
                        "row_index": row, "chunk_length": length, "held": False,
                    }
                    state["next_row"] += 1
            elif self._consumption is not None and not corrections and self._consumption[i]["last"] is not None:
                self._consumption[i]["last"]["held"] = True
            assert last[i] is not None, "A reset requires a prefilled initial chunk"
        return torch.stack(last)

    def consumption(self, env_id: int) -> dict[str, int | bool] | None:
        """Return detached last action-dequeue metadata, not downstream execution.

        Args:
            env_id: Environment slot whose most recent action consume is queried.

        Returns:
            Generation, submission ordinal, row, chunk length and hold state;
            None when tracking is disabled or no action has been consumed since reset.
        """
        last = None if self._consumption is None else self._consumption[env_id]["last"]
        return None if last is None else dict(last)

    def advance(self) -> None:
        """Advance one native control tick without stepping simulation ourselves."""
        self.ticks = [tick + 1 for tick in self.ticks]

    def reset(self, env_ids: list[int]) -> None:
        """Discard pre-reset work only for the specified environments."""
        for i in env_ids:
            self.ticks[i] = 0
            self.histories[i].clear()
            self.pending[i] = None
            self.initialized[i] = False
            self.last_actions[i] = None
            self.last_corrections[i] = None
            if self._consumption is not None:
                state = self._consumption[i]
                state.update(generation=state["generation"] + 1, next_submission=0,
                             pending=None, active=None, next_row=0, last=None)
