# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Move a camera observation dataset from ``camera_obs`` into ``obs`` for every demo in an HDF5 file.

Generated datasets store image observations under ``data/demo_N/camera_obs`` while
proprioceptive observations live under ``data/demo_N/obs``. This script relocates a
camera key (default ``robot_head_cam_rgb``) into ``obs`` so all observations share one group,
then removes the now-empty ``camera_obs`` group.

The original file is preserved: the input is copied to ``<input>_preproc.hdf5`` and all
edits are made on that copy in place (the move uses ``h5py.Group.move``, no data copy).
"""

import argparse
import h5py
import os
import shutil


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


def preprocess(input_file: str, key: str) -> str:
    root, ext = os.path.splitext(input_file)
    output_file = f"{root}_preproc{ext}"

    shutil.copy2(input_file, output_file)
    print(f"Copied '{input_file}' -> '{output_file}'")

    move_camera_obs(output_file, key)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=str, help="Path to the source HDF5 dataset (preserved, not modified).")
    parser.add_argument(
        "--key",
        type=str,
        default="robot_head_cam_rgb",
        help="Name of the dataset to move from 'camera_obs' to 'obs'.",
    )
    args = parser.parse_args()

    output_file = preprocess(args.input_file, args.key)
    print(f"Wrote preprocessed dataset to '{output_file}'")


if __name__ == "__main__":
    main()
