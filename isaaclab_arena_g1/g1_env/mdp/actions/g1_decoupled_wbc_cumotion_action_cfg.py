# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_cumotion_action import G1DecoupledWBCCuMotionAction
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg


@configclass
class G1DecoupledWBCCuMotionActionCfg(G1DecoupledWBCPinkActionCfg):
    """G1 WBC action with the upper body driven by cuMotion RMPFlow instead of PINK IK.

    Inherits the navigation and WBC settings from the PINK config unchanged; only the
    upper-body solver differs. The defaults below mirror the G1 deployment stack's
    ``ik_controller`` parameters.
    """

    class_type: type[ActionTerm] = G1DecoupledWBCCuMotionAction
    """Specifies the action term class type for G1 WBC with upper body cuMotion RMPFlow control."""

    arm_control_mode: str = "legacy_action_step"
    """Opt into physics_step_rnea for measured-feedback arm position/effort updates every 5 ms."""

    rnea_urdf_sha256: str = ""
    """Independently pinned exact fixed-base URDF; required only for physics_step_rnea."""

    # Joint velocity ceiling [rad/s], from the deployment controller_manager.yaml.
    max_joint_velocity: float = 1.2

    # L2 drift [rad] between integrated and measured arm joints past which the integrator
    # hard-snaps to the measured state, as ``BimanualIkController`` does.
    drift_reset_threshold: float = 0.5

    # Frames RMPFlow drives. The deployment controller targets the palms.
    left_ee_frame: str = "left_hand_palm_link"
    right_ee_frame: str = "right_hand_palm_link"

    # Frames the commanded poses describe. The WBC action space is defined on the wrist
    # links, so the fixed wrist->palm offset is composed into each target.
    left_command_frame: str = "left_wrist_yaw_link"
    right_command_frame: str = "right_wrist_yaw_link"
