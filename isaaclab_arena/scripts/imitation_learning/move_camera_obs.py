# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Move a camera observation dataset from ``camera_obs`` into ``obs`` for every demo in an HDF5 file.

Generated datasets store image observations under ``data/demo_N/camera_obs`` while
proprioceptive observations live under ``data/demo_N/obs``. This script relocates a
camera key (default ``robot_head_cam_rgb``) into ``obs`` so all observations share one group,
then removes the now-empty ``camera_obs`` group.

G1 Mimic gripper labels are canonicalized to {-0.5, 0} by default. Use
``--no-canonicalize_mimic_grippers`` to preserve the original target labels.

The original file is preserved: the input is copied to ``<input>_preproc.hdf5`` and all
edits are made on that copy in place (the move uses ``h5py.Group.move``, no data copy).
"""

import argparse
import h5py
import os
import shutil
import numpy as np


def move_camera_obs(file_path: str, key: str) -> None:
    with h5py.File(file_path, "r+") as f:
        demos = list(f["data"].keys())
        moved = 0
        for demo in demos:
            demo_group = f["data"][demo]
            src = f"camera_obs/{key}"
            dst = f"obs/{key}"

            if src not in demo_group:
                print(f"[skip] {demo}: 'camera_obs/{key}' not found")
            elif dst in demo_group:
                print(f"[skip] {demo}: 'obs/{key}' already exists")
            else:
                demo_group.move(src, dst)
                moved += 1

            # Remove the camera_obs group if it is now empty.
            if "camera_obs" in demo_group and len(demo_group["camera_obs"]) == 0:
                del demo_group["camera_obs"]

        print(f"Moved '{key}' from 'camera_obs' to 'obs' in {moved}/{len(demos)} demos.")


def canonicalize_mimic_gripper_labels(file_path: str) -> None:
    """Encode G1 Mimic's executed hand decisions as deployment-compatible labels.

    Args:
        file_path: Writable derived G1 dataset with 23D actions, never the source recording.

    Native Mimic closes for every nonzero hand value, including release interpolation.
    Only target labels change; recorded observations and all wrist/body channels stay intact.
    This is not appropriate for other controllers or raw learned-policy outputs.
    """
    with h5py.File(file_path, "r+") as f:
        for demo in f["data"].values():
            actions = demo["actions"]
            assert actions.ndim == 2 and actions.shape[1] == 23
            assert np.issubdtype(actions.dtype, np.floating)
            grippers = actions[:, :2]
            assert np.isfinite(grippers).all() and ((grippers >= -0.5) & (grippers <= 0.0)).all()
            actions[:, :2] = np.where(grippers != 0.0, -0.5, 0.0)
        f["data"].attrs["mimic_gripper_labels"] = "native_nonzero_closed_to_binary_v1"


def preprocess(input_file: str, key: str | list[str], *, canonicalize_mimic_grippers: bool = True) -> str:
    root, ext = os.path.splitext(input_file)
    output_file = f"{root}_preproc{ext}"

    shutil.copy2(input_file, output_file)
    print(f"Copied '{input_file}' -> '{output_file}'")

    for camera_key in [key] if isinstance(key, str) else key:
        move_camera_obs(output_file, camera_key)
    if canonicalize_mimic_grippers:
        canonicalize_mimic_gripper_labels(output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=str, help="Path to the source HDF5 dataset (preserved, not modified).")
    parser.add_argument(
        "--key",
        type=str,
        nargs="+",
        default="robot_head_cam_rgb",
        help="Camera keys to move from 'camera_obs' to 'obs' (for RGBD, pass both robot_head_cam_rgb and robot_head_cam_depth).",
    )
    parser.add_argument(
        "--canonicalize_mimic_grippers", action=argparse.BooleanOptionalAction, default=True,
        help="For native G1 Mimic only: preserve executed nonzero=closed hand timing as binary target labels (default: enabled).",
    )
    args = parser.parse_args()

    output_file = preprocess(args.input_file, args.key, canonicalize_mimic_grippers=args.canonicalize_mimic_grippers)
    print(f"Wrote preprocessed dataset to '{output_file}'")


if __name__ == "__main__":
    main()
