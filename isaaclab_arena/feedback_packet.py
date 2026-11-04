# SPDX-FileCopyrightText: 2026 The Isaac AutoData Project Developers
# SPDX-License-Identifier: Apache-2.0

"""Staged causal packet cache only; no native, ROS, scheduler, or hardware integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from threading import RLock, current_thread

CANONICAL_Q_NAMES = (
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)
FEEDBACK_DIM = 20


class FeedbackUnavailable(RuntimeError):
    """Required causal evidence is unavailable; callers must not infer with a substitute."""


@dataclass(frozen=True)
class FeedbackPacket:
    """Immutable submission evidence; ages [s] are measured at cutoff, not delivery."""

    env_id: int
    generation: int
    sequence: int
    cutoff_ns: int
    joint_header_ns: int
    joint_receipt_ns: int
    delivered_ns: int | None
    delivery_receipt_ns: int | None
    chunk_id: str | None
    consumed_row: int | None
    features: tuple[float, ...]


def timestamp(value: int) -> int:
    assert type(value) is int and value >= 0, "Timestamp must be nonnegative integer nanoseconds"
    return value


def finite_float(value: Real) -> float:
    assert isinstance(value, Real) and not isinstance(value, bool), "Expected a real number, not bool"
    value = float(value)
    assert math.isfinite(value) and abs(value) <= 3.4028234663852886e38, "Nonfinite/unrepresentable float32 value"
    return value


class FeedbackCache:
    """Keep one pending serialization token per environment, without implementing a scheduler.

    All timestamps must share one caller-declared clock domain [ns]. Header/receipt
    ages concern publication and callback receipt, never physical hand acquisition.
    Create and use the cache on one serialized owner thread. Sample receipt/current
    timestamps at that thread's callback entry, never before cross-thread queuing.
    A native loop or SingleThreadedExecutor can satisfy this contract; a mutually
    exclusive group on a MultiThreadedExecutor does not guarantee thread ownership.
    No tensors or mutable message arrays are retained. Methods synchronize on one lock.
    Ownership transfers to the complete input packet before enqueue/publication, not
    when inference finishes; several externally owned inputs may therefore be in flight.
    """

    @dataclass
    class _State:
        generation: int = 0
        sequence: int = 0
        barrier_ns: int = -1
        clock_ns: int | None = None
        paused: bool = False
        joint: tuple | None = None
        delivery: tuple | None = None
        pending: FeedbackPacket | None = None

    def __init__(self, num_envs: int, *, max_header_age_ns: int, max_receipt_age_ns: int,
                 max_pending_age_ns: int) -> None:
        if not __debug__:
            raise RuntimeError("Feedback validation requires Python assertions enabled")
        assert type(num_envs) is int and num_envs > 0
        for value in (max_header_age_ns, max_receipt_age_ns, max_pending_age_ns):
            assert timestamp(value) > 0, "Declare positive age limits explicitly"
            finite_float(value * 1e-9)  # Age features must remain representable at the float32 actor boundary.
        self.max_header_age_ns = max_header_age_ns
        self.max_receipt_age_ns = max_receipt_age_ns
        self.max_pending_age_ns = max_pending_age_ns
        self._states = [self._State() for _ in range(num_envs)]
        self._lock = RLock()
        self._owner_thread = current_thread()

    def _owner(self) -> None:
        if current_thread() is not self._owner_thread:
            raise FeedbackUnavailable("Cache requires its creating thread; serialize callbacks on that owner")

    def _state(self, env_id: int) -> _State:
        assert type(env_id) is int and 0 <= env_id < len(self._states), "Invalid environment index"
        return self._states[env_id]

    @staticmethod
    def _invalidate(state: _State, now_ns: int) -> None:
        state.generation += 1
        state.barrier_ns = now_ns
        state.clock_ns = now_ns
        state.joint = state.delivery = state.pending = None

    def _clock(self, state: _State, now_ns: int) -> None:
        timestamp(now_ns)
        if state.clock_ns is not None and now_ns < state.clock_ns:
            self._invalidate(state, now_ns)
            raise FeedbackUnavailable("Clock regression invalidated this environment generation")
        state.clock_ns = now_ns

    @staticmethod
    def _generation(state: _State, expected_generation: int) -> None:
        if type(expected_generation) is not int or expected_generation != state.generation:
            raise FeedbackUnavailable("Old/invalid generation; delayed data rejected")

    def generation(self, env_id: int) -> int:
        self._owner()
        with self._lock:
            return self._state(env_id).generation

    def reset(self, env_ids: list[int], *, now_ns: int) -> None:
        """Invalidate selected environments and require strictly post-reset evidence [ns]."""
        self._transition(env_ids, now_ns, None)

    def pause(self, env_ids: list[int], *, now_ns: int) -> None:
        """Invalidate selected environments; paused caches cannot accept or submit data."""
        self._transition(env_ids, now_ns, True)

    def resume(self, env_ids: list[int], *, now_ns: int) -> None:
        """Start a new generation; fresh post-resume q is mandatory, history starts invalid."""
        self._transition(env_ids, now_ns, False)

    def _transition(self, env_ids: list[int], now_ns: int, paused: bool | None) -> None:
        self._owner()
        timestamp(now_ns)
        with self._lock:
            assert env_ids and len(set(env_ids)) == len(env_ids), "Select distinct environments"
            states = [self._state(env_id) for env_id in env_ids]
            for state in states:
                self._invalidate(state, now_ns)
                if paused is not None:
                    state.paused = paused

    def update_joint_state(self, env_id: int, names: list[str], positions: list[float], *,
                           header_ns: int, receipt_ns: int, expected_generation: int) -> None:
        """Copy named absolute joint q [rad]; malformed/stale data invalidates this generation."""
        self._owner()
        with self._lock:
            state = self._state(env_id)
            self._generation(state, expected_generation)
            if state.paused:
                raise FeedbackUnavailable("Paused; wait for resume and fresh q")
            try:
                self._clock(state, receipt_ns)
                timestamp(header_ns)
                assert state.barrier_ns < header_ns <= receipt_ns, "Future/pre-reset header"
                assert receipt_ns - header_ns <= self.max_header_age_ns, "Stale header at receipt"
                assert len(names) == len(positions) and all(type(name) is str for name in names)
                assert len(set(names)) == len(names), "Duplicate joint names"
                assert set(CANONICAL_Q_NAMES).issubset(names), "Missing named finger q"
                values = {name: finite_float(value) for name, value in zip(names, positions, strict=True)}
                if state.joint is not None:
                    assert header_ns > state.joint[0] and receipt_ns > state.joint[1], "Out-of-order/duplicate JointState"
                state.joint = (header_ns, receipt_ns, tuple(values[name] for name in CANONICAL_Q_NAMES))
            except (AssertionError, TypeError, ValueError) as error:
                self._invalidate(state, state.clock_ns if state.clock_ns is not None else -1)
                raise FeedbackUnavailable(str(error)) from error

    def record_delivery(self, env_id: int, *, chunk_id: str, row: int, left: float, right: float,
                        delivered_ns: int, receipt_ns: int, expected_generation: int) -> None:
        """Observe an actually delivered post-threshold WBC row, not a proposed action.

        Gripper commands are exactly 0.0 (open) or -0.5 (close), before the decoder
        maps them to fourteen joint references. This records evidence; it sends nothing.
        delivered_ns is the source event time; receipt_ns is the current owner callback
        entry time [ns]. A delayed trace never rewrites an already submitted packet.
        """
        self._owner()
        with self._lock:
            state = self._state(env_id)
            self._generation(state, expected_generation)
            if state.paused:
                raise FeedbackUnavailable("Paused; no delivered history accepted")
            try:
                self._clock(state, receipt_ns)
                timestamp(delivered_ns)
                assert delivered_ns <= receipt_ns, "Future delivery at receipt"
                assert type(chunk_id) is str and chunk_id
                assert type(row) is int and 0 <= row < 6, "Expected consumed row0..5"
                left, right = finite_float(left), finite_float(right)
                assert left in (0.0, -0.5) and right in (0.0, -0.5), "Expected post-threshold delivered grippers"
                if delivered_ns <= state.barrier_ns:
                    raise FeedbackUnavailable("Pre-reset delivery rejected without changing valid history")
                if state.delivery is not None:
                    previous = state.delivery
                    if delivered_ns < previous[0]:
                        raise FeedbackUnavailable("Older delivery rejected without changing valid history")
                    if delivered_ns == previous[0]:
                        assert (chunk_id, row, left, right) == previous[2:], "Conflicting duplicate delivery"
                        raise FeedbackUnavailable("Duplicate delivery rejected without changing valid history")
                state.delivery = (delivered_ns, receipt_ns, chunk_id, row, left, right)
            except (AssertionError, TypeError, ValueError) as error:
                self._invalidate(state, state.clock_ns if state.clock_ns is not None else -1)
                raise FeedbackUnavailable(str(error)) from error

    def submit(self, env_id: int, *, cutoff_ns: int) -> FeedbackPacket:
        """Freeze q/history at the current owner-thread cutoff [ns], never a historical query.

        Sample cutoff immediately before this call, on the same serialized owner thread.
        Source and receipt times must both precede this cutoff; no retroactive backfill.
        """
        self._owner()
        with self._lock:
            state = self._state(env_id)
            self._clock(state, cutoff_ns)
            if state.paused or state.joint is None:
                raise FeedbackUnavailable("Paused or missing valid measured q; no zero substitution")
            if state.pending is not None:
                raise FeedbackUnavailable("One pending packet per environment; scheduler must consume or reset it")
            header, receipt, q = state.joint
            if not (state.barrier_ns < header <= receipt <= cutoff_ns
                    and cutoff_ns - header <= self.max_header_age_ns
                    and cutoff_ns - receipt <= self.max_receipt_age_ns):
                self._invalidate(state, cutoff_ns)
                raise FeedbackUnavailable("Stale/noncausal q at submission; generation invalidated")
            delivery = state.delivery
            assert delivery is None or delivery[0] <= delivery[1] <= cutoff_ns
            history = (0.0, 0.0, 0.0, 0.0) if delivery is None else (delivery[3] / 5, delivery[4], delivery[5], 1.0)
            features = q + ((cutoff_ns - header) * 1e-9, (cutoff_ns - receipt) * 1e-9) + history
            assert len(features) == FEEDBACK_DIM and all(math.isfinite(value) for value in features)
            state.sequence += 1
            packet = FeedbackPacket(env_id, state.generation, state.sequence, cutoff_ns, header, receipt,
                                    None if delivery is None else delivery[0], None if delivery is None else delivery[1],
                                    None if delivery is None else delivery[2], None if delivery is None else delivery[3],
                                    features)
            state.pending = packet
            return packet

    def consume(self, packet: FeedbackPacket, *, now_ns: int) -> FeedbackPacket:
        """Transfer the exact still-fresh token immediately before enqueue/publication.

        The owner serializes the complete input first, samples now_ns [ns], then
        consumes and immediately publishes/enqueues on that same serialized thread.
        On enqueue/publication exception the caller must reset the generation with
        a fresh owner timestamp; reset cannot retract an externally accepted message.
        This is not a backend completion callback or physical hand-freshness gate.
        Original cutoff features never change, including their encoded ages/history.
        """
        self._owner()
        with self._lock:
            state = self._state(packet.env_id)
            self._generation(state, packet.generation)
            self._clock(state, now_ns)
            if state.paused or state.pending is not packet:
                raise FeedbackUnavailable("Invalid/already consumed packet token")
            if (now_ns - packet.cutoff_ns > self.max_pending_age_ns
                    or now_ns - packet.joint_header_ns > self.max_header_age_ns
                    or now_ns - packet.joint_receipt_ns > self.max_receipt_age_ns):
                self._invalidate(state, now_ns)
                raise FeedbackUnavailable("Packet or original joint evidence expired before transfer; generation invalidated")
            state.pending = None
            return packet
