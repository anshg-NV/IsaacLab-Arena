# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical G1 upper-body arm joint order for joint-space policies.

One list, shared by everything that has to agree on what "the arm joints" means and in
what order: the ``left_arm_joint_pos`` / ``right_arm_joint_pos`` observation terms, the
16-dim joint-space policy action, the dataset conversion that builds that action, and the
deployment decoder that turns it back into a joint command. The observation terms resolve
these names with ``preserve_order=True``, so the order below -- not the articulation's own
ordering -- is what the policy sees and emits.

Proximal to distal per arm, which is also the order the WBC joint-order YAML
(``loco_manip_g1_joints_order_43dof.yaml``) and the deployment
``upper_body_forward_joint_command_controller`` use within an arm.

The waist and finger joints are deliberately absent. ``waist_yaw/roll/pitch`` are held at
zero by the decoupled WBC lower-body policy, and the fingers are a deterministic expansion
of the two gripper scalars, so neither carries information the policy would have to predict.
"""

import torch

G1_LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

G1_RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_ARM_JOINT_NAMES: tuple[str, ...] = G1_LEFT_ARM_JOINT_NAMES + G1_RIGHT_ARM_JOINT_NAMES

NUM_ARM_JOINTS_PER_SIDE = len(G1_LEFT_ARM_JOINT_NAMES)
NUM_ARM_JOINTS = len(G1_ARM_JOINT_NAMES)

# Layout of the 16-dim joint-space policy action. The two gripper scalars keep the indices
# they have in the 23-dim end-effector action, so the gripper handling (binarize, then
# expand to finger targets) is unchanged; only what follows them differs.
G1_JOINT_ACTION_LEFT_GRIPPER_IDX = 0
G1_JOINT_ACTION_RIGHT_GRIPPER_IDX = 1
G1_JOINT_ACTION_LEFT_ARM_START_IDX = 2
G1_JOINT_ACTION_LEFT_ARM_END_IDX = G1_JOINT_ACTION_LEFT_ARM_START_IDX + NUM_ARM_JOINTS_PER_SIDE
G1_JOINT_ACTION_RIGHT_ARM_START_IDX = G1_JOINT_ACTION_LEFT_ARM_END_IDX
G1_JOINT_ACTION_RIGHT_ARM_END_IDX = G1_JOINT_ACTION_RIGHT_ARM_START_IDX + NUM_ARM_JOINTS_PER_SIDE
G1_JOINT_ACTION_DIM = G1_JOINT_ACTION_RIGHT_ARM_END_IDX

# Per-dimension labels for the 16-dim action, for LEAPP export element names.
G1_JOINT_ACTION_ELEMENT_NAMES: tuple[str, ...] = ("left_gripper", "right_gripper") + G1_ARM_JOINT_NAMES


def joint_action_to_wbc(action: torch.Tensor, joint_names: list[str], base_height: float) -> torch.Tensor:
    """Expand absolute 16-D joint actions into the articulation's 50-D standing commands.

    Grippers stay continuous until this final conversion. Arm columns follow the
    canonical policy order; output joint columns follow the simulator's order.
    """
    assert action.shape[-1] == G1_JOINT_ACTION_DIM
    assert len(joint_names) == len(set(joint_names)) == 43, "Expected the 43-joint G1 articulation"
    zero = torch.zeros_like(action[..., :1])
    targets = dict(zip(G1_ARM_JOINT_NAMES, action[..., 2:].split(1, dim=-1), strict=True))
    finger_suffixes = ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
    closed_left = (-0.6, -1.2, -0.6, -1.2, 0.0, 0.7, 0.7)
    for column, side, sign in ((0, "left", 1.0), (1, "right", -1.0)):
        closed = (action[..., column : column + 1] < -0.25).to(action.dtype)
        for suffix, target in zip(finger_suffixes, closed_left, strict=True):
            targets[f"{side}_hand_{suffix}_joint"] = closed * (sign * target)
    assert targets.keys() <= set(joint_names), "Missing G1 arm or finger joints"
    joints = torch.cat([targets.get(name, zero) for name in joint_names], dim=-1)
    return torch.cat(
        (
            joints,
            torch.zeros_like(action[..., :3]),
            torch.full_like(zero, base_height),
            torch.zeros_like(action[..., :3]),
        ),
        dim=-1,
    )
