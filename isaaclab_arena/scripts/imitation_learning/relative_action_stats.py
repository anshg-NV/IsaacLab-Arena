# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report the chunk-anchored relative action distribution of an HDF5 dataset.

Diffusion Policy trains on actions in ``[-1, 1]``, so the relative actions produced by
``algo.relative_actions`` are divided by a position and a rotation scale. This script measures
the deltas the training pipeline would see and prints the scales that make them fill the range,
to paste into a training config.

A chunk starting at absolute index ``t0 - (observation_horizon - 1)`` is anchored on the measured
end-effector pose at ``t0``, which is the alignment ``DiffusionPolicyUNet`` uses.

    python isaaclab_arena/scripts/imitation_learning/relative_action_stats.py dataset.hdf5
    python isaaclab_arena/scripts/imitation_learning/relative_action_stats.py dataset.hdf5 --num_demos 500
    python isaaclab_arena/scripts/imitation_learning/relative_action_stats.py joint.hdf5 --action_space joint
"""

import argparse
import json
import h5py
import numpy as np
import torch

import robomimic.utils.relative_action_utils as RelActionUtils

DEFAULT_POS_INDICES = (2, 9)
"""Start indices of the left and right eef positions in the 23-dim G1 action."""

DEFAULT_ANCHOR_KEYS = (("left_eef_pos", "left_eef_quat"), ("right_eef_pos", "right_eef_quat"))
"""Observation keys holding the measured pose of each end effector, in the same order as the indices."""

DEFAULT_JOINT_INDICES = (2, 9)
"""Start indices of the left and right arm joint blocks in the 16-dim G1 joint action."""

DEFAULT_JOINT_ANCHOR_KEYS = ("left_arm_joint_pos", "right_arm_joint_pos")
"""Observation keys holding the measured arm joint angles, in the same order as the indices."""


def collect_deltas(
    file_path: str,
    pos_indices: tuple[int, ...],
    anchor_keys: tuple[tuple[str, str], ...],
    observation_horizon: int,
    prediction_horizon: int,
    num_demos: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect the anchor-relative position and rotation deltas of every action chunk.

    Args:
        file_path: HDF5 dataset to read.
        pos_indices: Start index of each end effector's position in one action; its xyzw
            quaternion is expected at the following three dimensions.
        anchor_keys: ``(position key, quaternion key)`` under ``obs`` for each end effector.
        observation_horizon: Observation horizon the policy is trained with.
        prediction_horizon: Number of actions in a predicted chunk.
        num_demos: Number of demos to read.

    Returns:
        Position deltas [m] and rotation vectors [rad], each of shape (num samples, 3).
    """
    positions, rotations = [], []
    with h5py.File(file_path, "r") as f:
        demos = list(f["data"].keys())[:num_demos]
        assert demos, "dataset contains no demos"

        for demo in demos:
            episode = f["data"][demo]
            actions = torch.from_numpy(episode["actions"][...]).double()
            num_steps = actions.shape[0]
            if num_steps < observation_horizon + prediction_horizon:
                continue

            # Chunk starts are offset from their anchor by the observation frames that precede it.
            anchor_steps = np.arange(observation_horizon - 1, num_steps - prediction_horizon)
            chunk_starts = anchor_steps - (observation_horizon - 1)
            chunks = torch.stack([actions[s : s + prediction_horizon] for s in chunk_starts])

            for pos_index, (anchor_pos_key, anchor_quat_key) in zip(pos_indices, anchor_keys):
                anchor_pos = torch.from_numpy(episode[f"obs/{anchor_pos_key}"][anchor_steps]).double()
                anchor_quat = torch.from_numpy(episode[f"obs/{anchor_quat_key}"][anchor_steps]).double()

                delta_pos = chunks[..., pos_index : pos_index + 3] - anchor_pos.unsqueeze(1)
                delta_quat = RelActionUtils.quat_multiply(
                    chunks[..., pos_index + 3 : pos_index + 7],
                    RelActionUtils.quat_conjugate(anchor_quat).unsqueeze(1),
                )
                positions.append(delta_pos.reshape(-1, 3).numpy())
                rotations.append(RelActionUtils.quat_to_rotvec(delta_quat).reshape(-1, 3).numpy())

    return np.concatenate(positions), np.concatenate(rotations)


def collect_joint_deltas(
    file_path: str,
    joint_indices: tuple[int, ...],
    anchor_keys: tuple[str, ...],
    observation_horizon: int,
    prediction_horizon: int,
    num_demos: int,
) -> np.ndarray:
    """Collect the anchor-relative joint deltas of every action chunk.

    Args:
        file_path: HDF5 dataset to read.
        joint_indices: Start index of each joint block within one action.
        anchor_keys: Observation key under ``obs`` holding the measured angles of each block.
        observation_horizon: Observation horizon the policy is trained with.
        prediction_horizon: Number of actions in a predicted chunk.
        num_demos: Number of demos to read.

    Returns:
        Joint deltas [rad], of shape (num samples, num joints per block).
    """
    deltas = []
    with h5py.File(file_path, "r") as f:
        demos = list(f["data"].keys())[:num_demos]
        assert demos, "dataset contains no demos"

        for demo in demos:
            episode = f["data"][demo]
            actions = episode["actions"][...]
            num_steps = actions.shape[0]
            if num_steps < max(observation_horizon, prediction_horizon):
                continue

            chunk_starts = np.arange(num_steps - max(observation_horizon, prediction_horizon) + 1)
            anchor_steps = chunk_starts + observation_horizon - 1
            chunks = np.stack([actions[s : s + prediction_horizon] for s in chunk_starts])

            for joint_index, anchor_key in zip(joint_indices, anchor_keys):
                anchor = episode[f"obs/{anchor_key}"][anchor_steps].astype(np.float64)
                num_joints = anchor.shape[-1]
                delta = chunks[..., joint_index : joint_index + num_joints].astype(np.float64) - anchor[:, None, :]
                deltas.append(delta.reshape(-1, num_joints))

    assert deltas, "No demos are long enough for the requested horizons"
    return np.concatenate(deltas)


def report_joint(deltas: np.ndarray) -> None:
    """Print the joint delta distribution and the scale that maps it onto ``[-1, 1]``."""
    scale = max(0.05, float(np.ceil(np.percentile(np.abs(deltas), 99.9) * 20.0) / 20.0))
    clipped = float(np.mean(np.abs(deltas) > scale))
    print(f"  joint delta [rad]: mean {deltas.mean():.4f} std {deltas.std():.4f}")
    print(f"    |delta| p99.9 {np.percentile(np.abs(deltas), 99.9):.4f}  max {np.abs(deltas).max():.4f}")
    print(
        f"    at joint_scale={scale}: normalized std {deltas.std() / scale:.3f},"
        f" {100 * clipped:.4f}% of values clipped"
    )
    print(f"\n  samples: {len(deltas)}")
    print("\n  suggested config block:")
    print('      "relative_actions": {')
    print('          "enabled": true,')
    print(f'          "joint_indices": {json.dumps(DEFAULT_JOINT_INDICES)},')
    print(f'          "anchor_joint_pos_keys": {json.dumps(DEFAULT_JOINT_ANCHOR_KEYS)},')
    print(f'          "joint_scale": {scale}')
    print("      }")


def report(positions: np.ndarray, rotations: np.ndarray) -> None:
    """Print the delta distribution and the scales that map it onto ``[-1, 1]``."""
    scales = {}
    for name, deltas, unit in (("pos", positions, "m"), ("rot", rotations, "rad")):
        scale = float(np.ceil(np.percentile(np.abs(deltas), 99.9) * 20.0) / 20.0)
        scales[name] = scale
        clipped = float(np.mean(np.abs(deltas) > scale))
        print(f"  {name} delta [{unit}]: mean {np.round(deltas.mean(0), 4)} std {np.round(deltas.std(0), 4)}")
        print(f"    |delta| p99.9 {np.percentile(np.abs(deltas), 99.9):.4f}  max {np.abs(deltas).max():.4f}")
        print(
            f"    at {name}_scale={scale}: normalized std {deltas.std() / scale:.3f},"
            f" {100 * clipped:.4f}% of values clipped"
        )

    print(f"\n  samples: {len(positions)}")
    print("\n  suggested config block:")
    print('      "relative_actions": {')
    print('          "enabled": true,')
    print(f'          "pos_scale": {scales["pos"]},')
    print(f'          "rot_scale": {scales["rot"]}')
    print("      }")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", type=str, help="Path to the HDF5 dataset.")
    parser.add_argument("--observation_horizon", type=int, default=2, help="Policy observation horizon.")
    parser.add_argument("--prediction_horizon", type=int, default=16, help="Policy prediction horizon.")
    parser.add_argument("--num_demos", type=int, default=200, help="Number of demos to sample.")
    parser.add_argument(
        "--pos_indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_POS_INDICES),
        help="Start index of each end effector position within one action.",
    )
    parser.add_argument(
        "--action_space",
        choices=("eef", "joint"),
        default="eef",
        help="Action space of the dataset: end-effector poses, or upper-body arm joint angles.",
    )
    args = parser.parse_args()

    if args.action_space == "joint":
        deltas = collect_joint_deltas(
            args.input_file,
            joint_indices=DEFAULT_JOINT_INDICES,
            anchor_keys=DEFAULT_JOINT_ANCHOR_KEYS,
            observation_horizon=args.observation_horizon,
            prediction_horizon=args.prediction_horizon,
            num_demos=args.num_demos,
        )
        print(f"Chunk-anchored relative joint actions over {args.num_demos} demos of {args.input_file}:")
        report_joint(deltas)
        return

    positions, rotations = collect_deltas(
        args.input_file,
        pos_indices=tuple(args.pos_indices),
        anchor_keys=DEFAULT_ANCHOR_KEYS,
        observation_horizon=args.observation_horizon,
        prediction_horizon=args.prediction_horizon,
        num_demos=args.num_demos,
    )
    print(f"Chunk-anchored relative actions over {args.num_demos} demos of {args.input_file}:")
    report(positions, rotations)


if __name__ == "__main__":
    main()
