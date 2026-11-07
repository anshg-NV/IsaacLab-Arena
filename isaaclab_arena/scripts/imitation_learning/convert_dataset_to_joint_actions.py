# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert an end-effector-action G1 dataset into a joint-space-action one.

Existing demos do not need to be re-recorded. Every demo already carries
``processed_actions`` -- the full-body joint position targets the decoupled-WBC + PINK-IK
action term actually commanded, recorded by ``PostStepProcessedActionsRecorder`` -- so the
joint-space labels are a slice of data that is already on disk. This script rewrites each
demo's ``actions`` from that slice and adds the matching arm-joint observations:

    actions            (T, 23) eef poses  ->  (T, 16) two grippers + 14 arm joint angles
    obs/left_arm_joint_pos    (T, 7)  new, measured left arm joint angles [rad]
    obs/right_arm_joint_pos   (T, 7)  new, measured right arm joint angles [rad]

The gripper scalars are copied unchanged from the end-effector action, so the downstream
binarize-and-expand handling is identical. The original ``actions`` are kept as
``eef_actions`` so a converted file can still be compared against the policy it came from.

    python isaaclab_arena/scripts/imitation_learning/convert_dataset_to_joint_actions.py \
        --input dataset/generated_preproc.hdf5 --output dataset/generated_preproc_joint.hdf5
"""

import argparse
import h5py
import numpy as np
import os
import shutil

from isaaclab_arena_g1.g1_env.g1_arm_joints import (
    G1_ARM_JOINT_NAMES,
    G1_JOINT_ACTION_DIM,
    G1_LEFT_ARM_JOINT_NAMES,
    G1_RIGHT_ARM_JOINT_NAMES,
    NUM_ARM_JOINTS,
)
from isaaclab_arena_g1.g1_env.g1_supplemental_info import G1SupplementalInfo

G1_SIM_JOINT_NAMES: tuple[str, ...] = (
    # Isaac Lab orders articulation joints breadth-first over the kinematic tree, which is the
    # order ``processed_actions`` and ``obs/robot_joint_pos`` are written in. Mirrored by the
    # deployment ``forward_position_controller`` joint list in the G1 bringup config.
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    # Each hand contributes three finger chains at once, so a depth level holds the whole left
    # hand before the whole right hand -- unlike the body, where the two sides alternate.
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
)
"""Articulation joint order of the 43-dim ``processed_actions`` / ``robot_joint_pos`` arrays."""

_WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

_CLOSED_LEFT_HAND = {
    "left_hand_index_0_joint": -0.6,
    "left_hand_index_1_joint": -1.2,
    "left_hand_middle_0_joint": -0.6,
    "left_hand_middle_1_joint": -1.2,
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_thumb_1_joint": 0.7,
    "left_hand_thumb_2_joint": 0.7,
}
"""Closed-hand finger targets the WBC upper-body controller writes; the right hand negates them."""

GRIPPER_CLOSE_THRESHOLD = -0.25
"""Gripper scalars below this are treated as "close"; matches the policy's own threshold."""

FINGER_MISMATCH_TOLERANCE = 0.05
"""Fraction of steps whose finger target may lag the gripper command, per demo.

The WBC settles the finger state one or two steps after the command flips, so short demos with
several grasps disagree on a few percent of steps. A mistaken joint order disagrees on far more.
"""


def joint_indices(names) -> np.ndarray:
    """Positions of the given joint names within the 43-dim articulation joint order.

    Args:
        names: Joint names to look up.
    """
    return np.array([G1_SIM_JOINT_NAMES.index(name) for name in names], dtype=np.int64)


def verify_joint_order(demo: h5py.Group) -> None:
    """Check the assumed articulation joint order against invariants the demo must satisfy.

    A wrong ordering would silently mislabel every action, so this asserts three things the
    recorded data pins down independently: the WBC holds the waist at zero, every finger target
    is exactly one of the two values that finger is ever commanded (open or closed), and those
    targets track the gripper scalar. The WBC's finger state settles a step or two behind the
    command, so a small fraction of samples around a gripper flip legitimately disagree; only
    a mistaken joint order pushes the disagreement past FINGER_MISMATCH_TOLERANCE.

    Args:
        demo: A demo group holding ``processed_actions`` and ``actions``.
    """
    processed = demo["processed_actions"][:]
    grippers = demo["actions"][:, :2]
    assert processed.shape[1] == len(G1_SIM_JOINT_NAMES), (
        f"expected {len(G1_SIM_JOINT_NAMES)} joint targets, got {processed.shape[1]}"
    )

    waist = processed[:, joint_indices(_WAIST_JOINT_NAMES)]
    assert np.allclose(waist, 0.0, atol=1e-6), (
        "waist joint targets are not zero; the assumed joint order or the WBC waist handling changed"
    )

    # Initial observations may already include settling and joint randomization; validate
    # arm targets against their limits without assuming one exact reset posture.
    joint_limits = G1SupplementalInfo().joint_limits
    for name in G1_ARM_JOINT_NAMES:
        lower, upper = joint_limits[name]
        column = processed[:, G1_SIM_JOINT_NAMES.index(name)]
        assert column.min() >= lower - 1e-3 and column.max() <= upper + 1e-3, (
            f"'{name}' targets span [{column.min():.4f}, {column.max():.4f}], outside its URDF"
            f" limit [{lower:.4f}, {upper:.4f}]; the assumed joint order is wrong"
        )

    for side, sign, gripper in (("left", 1.0, grippers[:, 0]), ("right", -1.0, grippers[:, 1])):
        closed = gripper < GRIPPER_CLOSE_THRESHOLD
        for left_name, value in _CLOSED_LEFT_HAND.items():
            name = left_name.replace("left_", f"{side}_", 1)
            actual = processed[:, G1_SIM_JOINT_NAMES.index(name)]
            expected = np.where(closed, sign * value, 0.0)

            is_open_or_closed = np.isclose(actual, 0.0, atol=1e-5) | np.isclose(actual, sign * value, atol=1e-5)
            assert is_open_or_closed.all(), (
                f"'{name}' takes values other than open/closed; the assumed joint order is wrong"
                " (a non-finger joint landed in this slot)"
            )
            mismatch = float(np.mean(~np.isclose(actual, expected, atol=1e-5)))
            assert mismatch <= FINGER_MISMATCH_TOLERANCE, (
                f"finger target '{name}' disagrees with the gripper scalar on {mismatch:.1%} of"
                f" steps (tolerance {FINGER_MISMATCH_TOLERANCE:.1%}); the assumed joint order or"
                " the closed-hand constants changed"
            )


def convert_demo(demo: h5py.Group, arm_ids: np.ndarray, left_ids: np.ndarray, right_ids: np.ndarray) -> None:
    """Rewrite one demo's actions in joint space and add the arm-joint observations.

    Args:
        demo: The demo group to rewrite in place.
        arm_ids: Indices of the 14 arm joints within the articulation joint order.
        left_ids: Indices of the left arm joints.
        right_ids: Indices of the right arm joints.
    """
    processed = demo["processed_actions"][:]
    eef_actions = demo["actions"][:]
    measured = demo["obs"]["robot_joint_pos"][:]

    joint_actions = np.concatenate([eef_actions[:, :2], processed[:, arm_ids]], axis=1).astype(np.float32)
    assert joint_actions.shape[1] == G1_JOINT_ACTION_DIM

    demo.move("actions", "eef_actions")
    demo.create_dataset("actions", data=joint_actions)
    demo["obs"].create_dataset("left_arm_joint_pos", data=measured[:, left_ids].astype(np.float32))
    demo["obs"].create_dataset("right_arm_joint_pos", data=measured[:, right_ids].astype(np.float32))


def verify_demos(demos: h5py.Group, names: list, count: int) -> None:
    """Check the assumed articulation joint order against the first ``count`` demos.

    Args:
        demos: The group holding every demo.
        names: Demo names, in order.
        count: How many demos to check; 0 disables the check.
    """
    if not count:
        return
    for name in names[:count]:
        verify_joint_order(demos[name])
    print(f"Joint order verified against {min(count, len(names))} demos.")


def main() -> None:
    """Copy the dataset and rewrite every demo's action space from end-effector to joint space."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Source HDF5 dataset with end-effector actions.")
    parser.add_argument("--output", required=True, help="Destination HDF5 dataset to write.")
    parser.add_argument(
        "--verify_demos",
        type=int,
        default=20,
        help="Number of demos to check the assumed articulation joint order against (0 disables).",
    )
    args = parser.parse_args()

    arm_ids = joint_indices(G1_ARM_JOINT_NAMES)
    left_ids = joint_indices(G1_LEFT_ARM_JOINT_NAMES)
    right_ids = joint_indices(G1_RIGHT_ARM_JOINT_NAMES)
    assert len(arm_ids) == NUM_ARM_JOINTS

    # The source dataset is often a symlink into a shared checkout, and the conversion rewrites
    # every demo in place, so refuse anything that could write through to the original.
    assert not os.path.islink(args.output), f"--output is a symlink and would be written through: {args.output}"
    assert not os.path.exists(args.output) or not os.path.samefile(args.input, args.output), (
        "--input and --output are the same file; the conversion rewrites its output in place"
    )

    print(f"Copying {args.input} -> {args.output}")
    shutil.copyfile(args.input, args.output)

    with h5py.File(args.output, "r+") as handle:
        demos = handle["data"]
        names = list(demos)
        verify_demos(demos, names, args.verify_demos)
        for index, name in enumerate(names):
            convert_demo(demos[name], arm_ids, left_ids, right_ids)
            if (index + 1) % 250 == 0:
                print(f"  converted {index + 1}/{len(names)} demos")

    size_gb = os.path.getsize(args.output) / 2**30
    print(f"Converted {len(names)} demos to {G1_JOINT_ACTION_DIM}-dim joint actions ({size_gb:.1f} GB).")


if __name__ == "__main__":
    main()
