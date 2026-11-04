# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.g1_wbc_upperbody_ik.g1_wbc_upperbody_cumotion_controller import (  # noqa: E501
    G1WBCUpperbodyCuMotionController,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_cumotion_action_cfg import (
        G1DecoupledWBCCuMotionActionCfg,
    )

# Same URDF the PINK path's robot model is built from, so both solvers see one robot.
_G1_URDF_NUCLEUS_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf"


class G1DecoupledWBCCuMotionAction(G1DecoupledWBCPinkAction):
    """G1 decoupled WBC action with the upper body solved by cuMotion RMPFlow.

    Identical to :class:`G1DecoupledWBCPinkAction` apart from the upper-body solver: the
    per-environment PINK controllers are replaced with
    :class:`G1WBCUpperbodyCuMotionController`, which is a port of the deployment stack's
    ``BimanualIkController``. Everything downstream -- the WBC closed loop, navigation
    P-controller, and action layout -- is inherited unchanged.

    The legacy mode integrates four internal ticks per action and holds the final position.
    The opt-in physics_step_rnea mode instead samples feedback and writes arm position/effort
    once per physics step. Neither mode alone qualifies complete hardware-controller parity.
    """

    cfg: G1DecoupledWBCCuMotionActionCfg

    def __init__(self, cfg: G1DecoupledWBCCuMotionActionCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        assert cfg.arm_control_mode in ("legacy_action_step", "physics_step_rnea")
        self._physics_step_rnea = cfg.arm_control_mode == "physics_step_rnea"
        super().__init__(cfg, env)

        assert not self.cfg.upperbody_extra_active_joints, (
            "cuMotion drives the arms-only 14-DOF XRDF, which fixes the waist at its default"
            f" position; upperbody_extra_active_joints={self.cfg.upperbody_extra_active_joints} cannot be"
            " honoured. Use the PINK action term for waist-in-IK configurations."
        )

        urdf_path = retrieve_file_path(_G1_URDF_NUCLEUS_PATH, force_download=False)
        if self._physics_step_rnea:
            import hashlib
            from pathlib import Path

            assert type(cfg.rnea_urdf_sha256) is str and len(cfg.rnea_urdf_sha256) == 64
            assert hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest() == cfg.rnea_urdf_sha256

        # One control tick per physics step, `decimation` ticks per environment step --
        # the sim counterpart of the deployment stack's 200 Hz controller / 50 Hz targets.
        control_dt = float(env.physics_dt)
        substeps = max(1, round(float(env.step_dt) / control_dt))

        self.upperbody_controllers = [
            G1WBCUpperbodyCuMotionController(
                robot_model=self.robot_model,
                urdf_path=urdf_path,
                control_dt=control_dt,
                substeps=substeps,
                ee_frames={"left": self.cfg.left_ee_frame, "right": self.cfg.right_ee_frame},
                command_frames={"left": self.cfg.left_command_frame, "right": self.cfg.right_command_frame},
                max_joint_velocity=self.cfg.max_joint_velocity,
                drift_reset_threshold=self.cfg.drift_reset_threshold,
                physics_step_rnea=self._physics_step_rnea,
            )
            for _ in range(self.num_envs)
        ]

        # Articulation indices of the joints cuMotion solves for, so episode resets can seed
        # each integrator from the measured arm state the way ``on_activate`` does on hardware.
        sim_joint_names = self._asset.data.joint_names
        self._cumotion_arm_sim_indices = [
            sim_joint_names.index(name) for name in self.upperbody_controllers[0].arm_joint_names
        ]
        if self._physics_step_rnea:
            self._validate_physics_step_mode()
            self._cumotion_pending_reset = np.ones(self.num_envs, dtype=bool)
            self._cumotion_command_ready = np.zeros(self.num_envs, dtype=bool)
            self._cumotion_effort = torch.zeros((self.num_envs, 14), device=self.device)

    def _validate_physics_step_mode(self) -> None:
        """Require the existing independent upper/lower composition and 5 ms apply cadence."""
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.g1_decoupled_whole_body_policy import (
            G1DecoupledWholeBodyPolicy,
        )
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.identity_policy import IdentityPolicy

        assert self._env.physics_dt == 0.005 and self._env.step_dt == 0.02
        assert type(self._env.cfg.decimation) is int and self._env.cfg.decimation == 4
        assert not self._env._physics_handles_decimation
        assert type(self.wbc_policy) is G1DecoupledWholeBodyPolicy
        assert type(self.wbc_policy.upper_body_policy) is IdentityPolicy
        arm_names = self.upperbody_controllers[0].arm_joint_names
        assert len(arm_names) == len(set(arm_names)) == 14
        assert all(c.arm_joint_names == arm_names for c in self.upperbody_controllers)
        lower = self.robot_model.get_joint_group_indices("lower_body")
        assert not set(lower) & {self.robot_model.dof_index(n) for n in arm_names}
        assert self._processed_actions.shape == (self.num_envs, self._asset.num_joints)
        assert self._joint_ids == slice(None), "Physics-step mode requires full articulation order"

    def process_actions(self, actions: torch.Tensor) -> None:
        """Reseed after final reset_to state installation, then call the existing parent once."""
        if self._physics_step_rnea and self._cumotion_pending_reset.any():
            q = self._asset.data.joint_pos.torch[:, self._cumotion_arm_sim_indices].cpu().numpy()
            v = self._asset.data.joint_vel.torch[:, self._cumotion_arm_sim_indices].cpu().numpy()
            for env_id in np.flatnonzero(self._cumotion_pending_reset):
                assert np.isfinite([q[env_id], v[env_id]]).all()
                self.upperbody_controllers[env_id].reset(q[env_id], v[env_id])
                self._cumotion_pending_reset[env_id] = False
        super().process_actions(actions)

    def apply_actions(self) -> None:
        """Replace only arm position/effort targets; preserve one parent hand/lower apply."""
        if self._physics_step_rnea:
            assert not self._cumotion_pending_reset.any() and self._cumotion_command_ready.all(), (
                "A new process_actions call is required after reset before physics apply"
            )
            assert not self._env._physics_handles_decimation and self._env.physics_dt == 0.005
            assert not self._asset.data.joint_vel_target.torch[:, self._cumotion_arm_sim_indices].any(), (
                "Physics-step bimanual mode requires zero arm velocity targets"
            )
            q = self._asset.data.joint_pos.torch[:, self._cumotion_arm_sim_indices].cpu().numpy()
            v = self._asset.data.joint_vel.torch[:, self._cumotion_arm_sim_indices].cpu().numpy()
            for env_id, controller in enumerate(self.upperbody_controllers):
                target, effort = controller.step_with_feedback(q[env_id], v[env_id])
                self._processed_actions[env_id, self._cumotion_arm_sim_indices] = torch.as_tensor(
                    target, dtype=self._processed_actions.dtype, device=self.device)
                self._cumotion_effort[env_id] = torch.as_tensor(
                    effort, dtype=self._cumotion_effort.dtype, device=self.device)
            assert torch.isfinite(self._cumotion_effort).all()
            assert torch.isfinite(self._processed_actions[:, self._cumotion_arm_sim_indices]).all()
        super().apply_actions()
        if self._physics_step_rnea:
            self._asset.set_joint_effort_target_index(target=self._cumotion_effort, joint_ids=self._cumotion_arm_sim_indices)

    def _invalidate_arm_state(self, env_ids) -> None:
        """Invalidate only selected episode state; never seed from pre-reset_to measurements."""
        selected = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device)
        if selected.numel() == 0:
            return
        assert selected.ndim == 1 and selected.dtype in (torch.int32, torch.int64)
        assert ((selected >= 0) & (selected < self.num_envs)).all()
        ids = selected.cpu().tolist()
        self._cumotion_pending_reset[ids] = True
        self._cumotion_command_ready[ids] = False
        self._cumotion_effort[selected] = 0
        self._asset.set_joint_effort_target_index(
            target=self._cumotion_effort[selected], joint_ids=self._cumotion_arm_sim_indices, env_ids=selected)

    def reset(self, env_ids=None) -> None:
        """Keep parent reset semantics while invalidating opt-in arm state per environment."""
        super().reset(env_ids)
        if self._physics_step_rnea:
            self._invalidate_arm_state(env_ids)

    def compute_upperbody_joint_positions(
        self,
        controller: G1WBCUpperbodyCuMotionController,
        body_data: dict[str, np.ndarray],
        left_hand_state: torch.Tensor,
        right_hand_state: torch.Tensor,
    ) -> np.ndarray:
        """Track wrist targets with measured-state feedback for the integrator drift reset."""
        env_id = self.upperbody_controllers.index(controller)
        if self._physics_step_rnea:
            assert not self._cumotion_pending_reset[env_id]
            result = controller.latch_targets(body_data, float(left_hand_state), float(right_hand_state))
            self._cumotion_command_ready[env_id] = True
            return result
        joint_pos = self._asset.data.joint_pos.torch[env_id, self._cumotion_arm_sim_indices].cpu().numpy()
        joint_vel = self._asset.data.joint_vel.torch[env_id, self._cumotion_arm_sim_indices].cpu().numpy()
        return controller.inverse_kinematics(
            body_data,
            float(left_hand_state),
            float(right_hand_state),
            measured_joint_positions=joint_pos,
            measured_joint_velocities=joint_vel,
        )

    def seed_controllers_from_articulation(self, env_ids: list[int]) -> None:
        """Re-seed the given environments' integrators from the measured arm joint state.

        Port of ``IkControllerBase::on_activate``, which starts from the measured joint state
        rather than a nominal pose so the arms hold still until a command arrives.

        Args:
            env_ids: Environment indices to reset.
        """
        if self._physics_step_rnea:
            self._invalidate_arm_state(env_ids)
            return
        joint_pos = self._asset.data.joint_pos.torch[:, self._cumotion_arm_sim_indices].cpu().numpy()
        joint_vel = self._asset.data.joint_vel.torch[:, self._cumotion_arm_sim_indices].cpu().numpy()
        for env_id in env_ids:
            self.upperbody_controllers[env_id].reset(
                joint_positions=np.asarray(joint_pos[env_id], dtype=np.float64),
                joint_velocities=np.asarray(joint_vel[env_id], dtype=np.float64),
            )
