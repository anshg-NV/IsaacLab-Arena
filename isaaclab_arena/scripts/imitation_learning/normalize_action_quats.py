# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rescale the end-effector quaternions in an HDF5 dataset's ``actions`` back to unit norm.

Recorded actions carry roughly 2% norm drift on their quaternion components. That is harmless
for the rotation itself -- scaling a quaternion does not change the rotation it represents --
but it lets ``w`` exceed 1.0 whenever the pose is near identity, which trips the ``[-1,1]``
range check that Diffusion Policy applies to actions at training time.

Renormalizing is lossless: it fixes the range violation without altering any commanded pose.

    python isaaclab_arena/scripts/imitation_learning/normalize_action_quats.py dataset.hdf5
    python isaaclab_arena/scripts/imitation_learning/normalize_action_quats.py dataset.hdf5 --apply
"""

import argparse
import h5py
import numpy as np

DEFAULT_QUAT_STARTS = (5, 12)
"""Start indices of the left and right eef ``(x, y, z, w)`` quaternions in the 23-dim G1 action."""


def normalize_action_quats(file_path: str, quat_starts: tuple[int, ...], apply: bool) -> None:
    """Report, and optionally apply in place, unit-norm rescaling of the action quaternions.

    Args:
        file_path: HDF5 dataset to inspect or edit in place.
        quat_starts: Start index of each ``(x, y, z, w)`` quaternion block within one action.
        apply: Whether to write the rescaled actions back; otherwise the file is only read.
    """
    mode = "r+" if apply else "r"
    with h5py.File(file_path, mode) as f:
        demos = list(f["data"].keys())
        total = 0
        out_of_range = 0
        worst_norm = 1.0

        for demo in demos:
            ds = f["data"][demo]["actions"]
            actions = ds[...]
            total += actions.shape[0]
            out_of_range += int((np.abs(actions) > 1.0).any(axis=1).sum())

            for start in quat_starts:
                quat = actions[:, start : start + 4]
                norms = np.linalg.norm(quat, axis=1, keepdims=True)
                worst_norm = max(worst_norm, float(np.abs(norms - 1.0).max() + 1.0))
                assert (norms > 1e-6).all(), f"{demo}: degenerate quaternion at action index {start}"
                actions[:, start : start + 4] = quat / norms

            if apply:
                ds[...] = actions

        still_bad = 0
        if apply:
            for demo in demos:
                still_bad += int((np.abs(f["data"][demo]["actions"][...]) > 1.0).any(axis=1).sum())

        verb = "Rescaled" if apply else "Would rescale"
        print(f"{verb} quaternions at action indices {list(quat_starts)} across {len(demos)} demos.")
        print(f"  samples: {total}, out of [-1,1] before: {out_of_range} ({100 * out_of_range / total:.3f}%)")
        print(f"  worst quaternion norm deviation: {worst_norm:.5f}")
        if apply:
            print(f"  out of [-1,1] after: {still_bad}")
        else:
            print("  pass --apply to write the rescaled actions back in place")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", type=str, help="Path to the HDF5 dataset.")
    parser.add_argument(
        "--quat_starts",
        type=int,
        nargs="+",
        default=list(DEFAULT_QUAT_STARTS),
        help="Start index of each (x, y, z, w) quaternion block within one action.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Edit the dataset in place; without it the file is only inspected."
    )
    args = parser.parse_args()

    normalize_action_quats(args.input_file, tuple(args.quat_starts), args.apply)


if __name__ == "__main__":
    main()
