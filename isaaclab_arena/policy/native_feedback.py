# SPDX-FileCopyrightText: 2026 The Isaac AutoData Project Developers
# SPDX-License-Identifier: Apache-2.0

"""Synchronous native feedback boundary; no stepping, inference or scheduling."""

import torch

from isaaclab_arena.feedback_packet import (
    CANONICAL_Q_NAMES, FeedbackCache, FeedbackPacket,
)

CONTROL_NS = 20_000_000
ZERO_DELAY_HYPOTHESIS = "zero_joint_publication_delay_unqualified"
METADATA_KEYS = {"generation", "submission_ordinal", "row_index", "chunk_length", "held"}


class NativeFeedback:
    """Use actual q and scheduler rows, accepting command history only after env.step.

    Phase offsets order synthetic native events, not measured physical timestamps.
    Caller owns real inference, immutable packet/chunk binding and scheduler resets.
    """

    def __init__(self, scheduler, joint_names: list[str], *, instance_id: str, timing_profile: str,
                 joint_delay_hypothesis: str, max_header_age_ns: int, max_receipt_age_ns: int,
                 max_pending_age_ns: int) -> None:
        assert type(instance_id) is str and instance_id
        assert joint_delay_hypothesis == ZERO_DELAY_HYPOTHESIS, "Declare the unqualified zero-delay hypothesis"
        assert timing_profile == "fresh240", "Queued-input profiles unsupported"
        assert scheduler.profile["queue_ticks"] == 0 and scheduler.profile["service_ticks"] == 12
        assert scheduler._consumption is not None, "Enable actual scheduler consumption metadata"
        assert len(joint_names) == len(set(joint_names)) and all(type(name) is str for name in joint_names)
        assert set(CANONICAL_Q_NAMES).issubset(joint_names), "Missing named measured finger q"
        self._timing, self.instance_id = scheduler, instance_id
        self._names = tuple(joint_names)
        self._indices = [joint_names.index(name) for name in CANONICAL_Q_NAMES]
        self._count = scheduler.num_envs
        self._cache = FeedbackCache(self._count, max_header_age_ns=max_header_age_ns,
                                    max_receipt_age_ns=max_receipt_age_ns, max_pending_age_ns=max_pending_age_ns)
        self._tick, self._phase, self._pending = 0, "uninitialized", None
        self._submitted = set()
        self._expected_generation = [None] * self._count
        self._scheduler_ticks = [0] * self._count
        self._submission_counts = [0] * self._count

    def _check(self, tick: int, phase: str) -> None:
        self._cache.generation(0)  # Enforce the cache's serialized constructing owner before any mutation.
        assert type(tick) is int and tick == self._tick, "Wrong global control tick"
        assert self._phase == phase, "Wrong native feedback phase; successful step or explicit abort required"

    def _ids(self, ids: list[int]) -> list[int]:
        assert isinstance(ids, (list, tuple)) and len(ids) == len(set(ids)), "Require distinct environment IDs"
        assert all(type(i) is int and 0 <= i < self._count for i in ids), "Invalid environment ID"
        return list(ids)

    def _q(self, positions: torch.Tensor) -> list[list[float]]:
        assert isinstance(positions, torch.Tensor) and positions.is_floating_point()
        assert positions.shape == (self._count, len(self._names)) and torch.isfinite(positions).all()
        assert (positions.abs() <= torch.finfo(torch.float32).max).all(), "Joint q must be float32 representable"
        return positions[:, self._indices].detach().cpu().tolist()

    def _observe(self, rows: list[list[float]], tick: int) -> None:
        stamp = tick * CONTROL_NS + 2
        for i, row in enumerate(rows):
            self._cache.update_joint_state(i, list(CANONICAL_Q_NAMES), row, header_ns=stamp,
                                           receipt_ns=stamp, expected_generation=self._cache.generation(i))

    def initialize(self, joint_positions: torch.Tensor, *, global_tick: int) -> None:
        """Copy initial native full-joint q at global tick0; no synthetic q substitution."""
        self._check(global_tick, "uninitialized")
        self._observe(self._q(joint_positions), global_tick)
        self._phase = "ready"

    def submission_packets(self, env_ids: list[int], *, global_tick: int) -> tuple[FeedbackPacket, ...]:
        """Transfer immutable packets on the actual inference-submission path only.

        Caller immediately binds each packet to its matching legacy/base input and
        retains it through scheduling/delivery; this is not a live actor observation.
        """
        self._check(global_tick, "ready")
        ids = self._ids(env_ids)
        assert ids and set(ids).issubset(self._timing.needs_inference()), "No actual inference request"
        assert not set(ids) & self._submitted, "Duplicate packet for environment/global tick"
        stamp = global_tick * CONTROL_NS + 2
        packets = tuple(self._cache.consume(self._cache.submit(i, cutoff_ns=stamp), now_ns=stamp) for i in ids)
        self._submitted.update(ids)
        return packets

    def prepare_step(self, grippers: torch.Tensor, *, global_tick: int) -> None:
        """Snapshot consumed rows after get_action; grippers are its post-threshold first2 fields."""
        self._check(global_tick, "ready")
        assert isinstance(grippers, torch.Tensor) and grippers.is_floating_point()
        assert grippers.shape == (self._count, 2) and torch.isfinite(grippers).all()
        assert ((grippers == 0.0) | (grippers == -0.5)).all(), "Require post-threshold grippers, not raw logits"
        grips, pending = grippers.detach().cpu().tolist(), []
        for i in range(self._count):
            metadata = self._timing.consumption(i)
            assert type(metadata) is dict and metadata.keys() == METADATA_KEYS, "No actual consumed row"
            assert all(type(metadata[k]) is int and metadata[k] >= 0 for k in METADATA_KEYS - {"held"})
            assert type(metadata["held"]) is bool and metadata["chunk_length"] == 6
            assert metadata["row_index"] < 6
            generation = self._expected_generation[i]
            assert generation is None or metadata["generation"] == generation, "Missing/extra scheduler reset"
            assert self._timing.ticks[i] == self._scheduler_ticks[i] + 1, "Require exactly one complete get_action"
            assert self._timing._consumption[i]["next_submission"] == self._submission_counts[i] + int(i in self._submitted), (
                "Actual inference submissions and immutable packets differ"
            )
            if self._scheduler_ticks[i] == 0:
                assert (metadata["submission_ordinal"], metadata["row_index"], metadata["held"]) == (0, 0, False)
                assert i in self._submitted, "Prefill lacks its actual feedback submission"
            identity = f"{self.instance_id}/env{i}/generation{metadata['generation']}/submission{metadata['submission_ordinal']}"
            pending.append((dict(metadata), tuple(grips[i]), identity, self._cache.generation(i),
                            self._timing.ticks[i], self._timing._consumption[i]["next_submission"]))
        self._pending, self._phase = tuple(pending), "pending_step"

    def accept_step(self, done_ids: list[int], next_joint_positions: torch.Tensor, *, global_tick: int) -> None:
        """Call only after successful native step, before policy.reset(done) and next get_action."""
        self._check(global_tick, "pending_step")
        done, rows = self._ids(done_ids), self._q(next_joint_positions)
        for i, (metadata, _, _, generation, tick, submissions) in enumerate(self._pending):
            assert self._timing.consumption(i) == metadata and self._timing.ticks[i] == tick, "Scheduler changed before acceptance"
            assert self._timing._consumption[i]["next_submission"] == submissions, "Submission changed before acceptance"
            assert self._cache.generation(i) == generation, "Feedback generation changed before acceptance"
        receipt = (global_tick + 1) * CONTROL_NS
        for i, (metadata, grips, identity, generation, tick, submissions) in enumerate(self._pending):
            if i not in done:
                self._cache.record_delivery(i, chunk_id=identity, row=metadata["row_index"], left=grips[0],
                                            right=grips[1], delivered_ns=global_tick * CONTROL_NS + 3,
                                            receipt_ns=receipt, expected_generation=generation)
            self._expected_generation[i] = metadata["generation"] + int(i in done)
            self._scheduler_ticks[i] = 0 if i in done else tick
            self._submission_counts[i] = 0 if i in done else submissions
        if done:
            self._cache.reset(done, now_ns=receipt + 1)  # Discard terminal history before auto-reset q enters cache.
        self._observe(rows, global_tick + 1)
        self._tick += 1
        self._pending, self._phase = None, "ready"
        self._submitted.clear()

    def abort_step(self, *, global_tick: int) -> None:
        """Invalidate an ambiguous/failed native step permanently; never accept its pending history."""
        self._check(global_tick, "pending_step")
        self._cache.reset(list(range(self._count)), now_ns=(global_tick + 1) * CONTROL_NS + 1)
        self._pending, self._phase = None, "failed"
