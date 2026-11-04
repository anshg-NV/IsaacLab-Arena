# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 upper-body inverse kinematics driven by the cuMotion RMPFlow policy.

Runs the same solver, on the same robot description and gains, as the G1 deployment stack's
``BimanualIkController``, so an action commanded in sim drives the arms the way it would on
hardware. The per-call update is a port of ``IkControllerBase::update()``:

    SyncIntegratorToHardware(dt)        # hard-snap the integrator if it has drifted
    ApplyPoseTargetsToRmpflow()         # push validated targets, holding the last good one
    rmpflow.eval_accel(q, q_dot, q_ddot)
    q_dot += dt * q_ddot                # v_target, and the integrator's new velocity
    q     += dt * q_dot                 # q_target, and the integrator's new position

**Rate.** This is the part that has to line up for trajectories to match. On hardware the IK
controller ticks at ``controller_manager.update_rate`` = 200 Hz while EE targets arrive from the
action decoder at 50 Hz, so every action gets exactly four RMPFlow integration steps against a
held target. Sim's ``sim.dt = 1/200`` with ``decimation = 4`` is the same 50/200 split, so one
:meth:`inverse_kinematics` call runs :attr:`substeps` = 4 integrations at ``control_dt`` = 1/200.
RMPFlow is a second-order policy with persistent velocity state -- integrating once at 50 Hz
instead of four times at 200 Hz gives a materially different trajectory, not a rounding error.

**No warmup.** ``BimanualIkController`` has none: on activation it points the targets at the
current end-effector poses and then runs one update per control cycle. :attr:`in_warmup` is
therefore ``False`` from construction, which makes the action term's warmup loop a no-op while
keeping the attribute for drop-in compatibility with the PINK controller.

Known differences from hardware that this seam cannot express:

* **Intra-window setpoints and feedback.** Hardware writes a position command and reads joint
  feedback on each 200 Hz tick; this samples feedback once, integrates four ticks, and returns
  the last target for sim to hold over the whole 20 ms step. Moving integration into
  ``ActionTerm.apply_actions`` would close this timing difference.
* **Joint impedance.** Hardware commands a uniform ``kp = 50``, ``kd = 1`` on all 14 arm joints,
  which reach the motors as ``effort = effort_ff + kp * (pos_cmd - pos) + kd * (vel_cmd - vel)``.
  The Arena G1 ``arms`` actuator implements the same law with different, per-joint gains --
  stiffness 100 (shoulder pitch/roll), 40 (shoulder yaw, elbow), 20 (wrists) and damping 5/2/2.
  The open-loop integrator matches the deployment equations, while physical tracking also
  depends on the actuator gains.
* **Feed-forward torque.** Legacy mode omits it. Opt-in physics-step mode computes the fixed-base
  pre-integration RNEA law. Deployment's default split path retains blended RNEA, whereas its
  single-controller path can overwrite it with gravity compensation at a safety-clamped q.
  Downstream blending, safety limits and motor effort delivery remain separate parity gates.

Waist joints are outside the arms-only XRDF and are held at their default positions, matching the
deployment controller; the PINK path's ``upperbody_extra_active_joints`` has no equivalent here.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from isaaclab_arena_g1.g1_env.robot_model import RobotModel
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.g1_wbc_upperbody_ik.cumotion_runtime import import_cumotion

# Robot description and RMPFlow gains, vendored verbatim from the G1 deployment stack.
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "g1_env" / "config"
DEFAULT_XRDF_PATH = _CONFIG_DIR / "g1_arms_only.xrdf"
DEFAULT_RMPFLOW_CONFIG_PATH = _CONFIG_DIR / "g1_bimanual_rmpflow.yaml"

# Frames RMPFlow drives, matching ``ik_controller``'s left/right_end_effector_frame.
DEFAULT_EE_FRAMES = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}

# Frames the incoming target poses are expressed for. The WBC action space is defined on the wrist
# links, so a commanded pose describes the wrist even though the palm is the controlled frame; the
# fixed wrist->palm transform is composed in, mirroring ``ExtractAndTransformPose``.
DEFAULT_COMMAND_FRAMES = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

# ``controller_manager.update_rate`` on hardware: the interval of one IK control tick.
DEFAULT_CONTROL_DT = 1.0 / 200.0

# Control ticks per commanded action. Hardware publishes EE targets at 50 Hz against a 200 Hz
# controller; sim's decimation=4 over sim.dt=1/200 is the same ratio.
DEFAULT_SUBSTEPS = 4

# ``max_joint_velocity`` and ``drift_reset_threshold`` from the deployment configuration.
DEFAULT_MAX_JOINT_VELOCITY = 1.2
DEFAULT_DRIFT_RESET_THRESHOLD = 0.5

# Damping-region fraction hard-coded in ``IkControllerBase::ApplyVelocityCap``.
_VELOCITY_DAMPING_REGION_FRACTION = 0.5

# Tolerances from ``IsReferencePoseValid``: a pose at the origin with identity rotation is the
# "unset" sentinel rather than a real command.
_POSITION_TOLERANCE = 1e-3
_ROTATION_TOLERANCE = 1e-3


def _is_reference_pose_valid(transform: np.ndarray) -> bool:
    """Return whether a commanded transform is a real target rather than an unset sentinel.

    Port of ``IsReferencePoseValid`` and ``IsPoseDataFinite``: rejects non-finite entries and the
    all-zeros sentinel that a zero-filled action produces, so neither can reach the integrator.

    Args:
        transform: 4x4 homogeneous target transform.

    Returns:
        True if the pose should be pushed to RMPFlow.
    """
    if not np.all(np.isfinite(transform)) or not np.any(transform):
        return False
    if not np.all(np.abs(transform[:3, 3]) <= _POSITION_TOLERANCE):
        return True
    # Position is at the origin -- only a non-identity rotation makes this a real command.
    cos_angle = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle)) >= _ROTATION_TOLERANCE


def _hand_joint_positions(hand_state: float) -> np.ndarray:
    """Return the seven hand joint targets for a binary gripper state.

    Mirrors ``G1WBCUpperbodyController.get_hand_joint_pos`` so the PINK and RMPFlow paths command
    identical fingers; only the arm solver differs. On hardware the fingers are driven by a
    separate ``fingers_forward_joint_command_controller``, not by the IK controller.

    Args:
        hand_state: 0 to open the hand, non-zero to close it.

    Returns:
        Hand joint positions [rad], ordered as the model's left-hand actuated joints.
    """
    hand_q_desired = np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if hand_state == 0:
        return hand_q_desired
    amp = 0.7
    hand_q_desired[1] += amp
    hand_q_desired[2] += amp

    amp_a = 0.6
    amp_b = 1.2
    hand_q_desired[3] -= amp_a
    hand_q_desired[4] -= amp_b
    hand_q_desired[5] -= amp_a
    hand_q_desired[6] -= amp_b
    return hand_q_desired


class G1WBCUpperbodyCuMotionController:
    """G1 upper-body controller backed by cuMotion RMPFlow.

    Exposes the same :meth:`inverse_kinematics` signature and :attr:`in_warmup` flag as the PINK
    controller, so it can be swapped in behind the existing action term.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        urdf_path: str | Path,
        control_dt: float = DEFAULT_CONTROL_DT,
        substeps: int = DEFAULT_SUBSTEPS,
        xrdf_path: str | Path = DEFAULT_XRDF_PATH,
        rmpflow_config_path: str | Path = DEFAULT_RMPFLOW_CONFIG_PATH,
        ee_frames: dict[str, str] | None = None,
        command_frames: dict[str, str] | None = None,
        max_joint_velocity: float = DEFAULT_MAX_JOINT_VELOCITY,
        drift_reset_threshold: float = DEFAULT_DRIFT_RESET_THRESHOLD,
        cspace_attractor: np.ndarray | None = None,
        physics_step_rnea: bool = False,
    ):
        """Initialize the controller.

        Args:
            robot_model: Full-robot model, used to map solved arm joints into the full
                configuration vector.
            urdf_path: URDF the XRDF is resolved against. Must contain the end-effector and
                command frames.
            control_dt: One IK control tick [s]. Defaults to the hardware controller period.
            substeps: Control ticks run per :meth:`inverse_kinematics` call. ``substeps *
                control_dt`` should equal the environment step period.
            xrdf_path: Arms-only XRDF defining the 14-DOF c-space cuMotion plans over.
            rmpflow_config_path: RMPFlow gains, in Lula ``format: rmpflow`` YAML.
            ee_frames: ``{"left": frame, "right": frame}`` frames RMPFlow drives to the targets.
            command_frames: ``{"left": frame, "right": frame}`` frames the incoming target poses
                describe. When these differ from ``ee_frames`` the fixed transform between them is
                composed into each target.
            max_joint_velocity: Joint velocity ceiling [rad/s] applied on top of the config's
                ``joint_velocity_cap_rmp``. Non-positive leaves the config value untouched.
            drift_reset_threshold: L2 distance [rad] between the integrated and measured joint
                positions past which the integrator hard-snaps to the measured state.
            cspace_attractor: Nullspace attractor [rad] over the 14 arm joints. Defaults to the
                zero configuration, matching ``BimanualIkController::OnSubclassActivate``.
            physics_step_rnea: Allocate per-controller fixed-base inverse dynamics for 5 ms updates.
        """
        assert control_dt > 0.0, f"control_dt must be positive, got {control_dt}"
        assert substeps >= 1, f"substeps must be at least 1, got {substeps}"
        assert drift_reset_threshold > 0.0, (
            f"drift_reset_threshold must be positive, got {drift_reset_threshold}"
        )

        self._cumotion = import_cumotion()
        self._robot_model = robot_model
        self._control_dt = float(control_dt)
        self._substeps = int(substeps)
        self._drift_reset_threshold = float(drift_reset_threshold)
        self._ee_frames = dict(ee_frames) if ee_frames else dict(DEFAULT_EE_FRAMES)
        self._command_frames = dict(command_frames) if command_frames else dict(DEFAULT_COMMAND_FRAMES)

        self._robot_description = self._cumotion.load_robot_from_file(str(xrdf_path), str(urdf_path))
        self._kinematics = self._robot_description.kinematics()
        self._num_dofs = self._robot_description.num_cspace_coords()

        frame_names = set(self._kinematics.frame_names())
        for side, frame in (*self._ee_frames.items(), *self._command_frames.items()):
            assert frame in frame_names, f"{side} frame '{frame}' is not in the URDF at '{urdf_path}'"

        # Map the XRDF's c-space ordering onto the full configuration vector once.
        self._arm_joint_names = [self._robot_description.cspace_coord_name(i) for i in range(self._num_dofs)]
        self._arm_full_indices = np.array(
            [self._robot_model.dof_index(name) for name in self._arm_joint_names], dtype=int
        )
        self._left_hand_full_indices = self._robot_model.get_hand_actuated_joint_indices(side="left")
        self._right_hand_full_indices = self._robot_model.get_hand_actuated_joint_indices(side="right")

        # One world per controller. Obstacles are not published from sim yet, so this carries only
        # the self-collision geometry the RMPFlow config declares.
        self._world = self._cumotion.create_world()
        config = self._cumotion.create_rmpflow_config_from_file(
            str(rmpflow_config_path), self._robot_description, self._world.add_world_view()
        )
        if max_joint_velocity > 0.0:
            # Port of ``IkControllerBase::ApplyVelocityCap``.
            config.set_param("joint_velocity_cap_rmp/max_velocity", max_joint_velocity)
            config.set_param(
                "joint_velocity_cap_rmp/velocity_damping_region",
                max_joint_velocity * _VELOCITY_DAMPING_REGION_FRACTION,
            )
        self._rmpflow = self._cumotion.create_rmpflow(config)
        for frame in self._ee_frames.values():
            self._rmpflow.add_target_frame(frame)

        self._cspace_attractor = (
            np.zeros(self._num_dofs) if cspace_attractor is None else np.asarray(cspace_attractor, dtype=np.float64)
        )
        assert self._cspace_attractor.shape == (self._num_dofs,), (
            f"cspace_attractor must have shape ({self._num_dofs},), got {self._cspace_attractor.shape}"
        )

        # Fixed command-frame -> end-effector-frame offsets, evaluated once at the default pose.
        # Both frames sit on the same rigid wrist body, so this transform is pose-independent.
        default_q = np.asarray(self._robot_description.default_cspace_configuration(), dtype=np.float64)
        self._command_to_ee = {
            side: self._kinematics.pose(default_q, self._ee_frames[side], self._command_frames[side])
            for side in self._ee_frames
        }

        # ``BimanualIkController`` has no warmup: it holds the current pose until a command
        # arrives. False from construction makes the action term's warmup loop a no-op.
        self.in_warmup = False
        self._q = np.zeros(self._num_dofs)
        self._q_dot = np.zeros(self._num_dofs)
        self._q_ddot = np.zeros(self._num_dofs)
        if physics_step_rnea:
            self._initialize_inverse_dynamics(urdf_path)
        self.reset()

    def _initialize_inverse_dynamics(self, urdf_path: str | Path) -> None:
        """Build private RNEA scratch; non-arm q/v/a remain zero as in deployment."""
        import pinocchio as pin

        self._pin = pin
        self._rnea_model = pin.buildModelFromUrdf(str(urdf_path))
        model = self._rnea_model
        assert model.nq == model.nv and self._num_dofs == 14
        assert len(set(self._arm_joint_names)) == 14
        assert all(model.existJointName(name) for name in self._arm_joint_names)
        joints = [model.joints[model.getJointId(name)] for name in self._arm_joint_names]
        assert all(j.nq == j.nv == 1 and j.idx_q == j.idx_v for j in joints)
        assert np.array_equal(model.gravity.linear, [0.0, 0.0, -9.81])
        self._rnea_indices = [j.idx_v for j in joints]
        self._rnea_data = model.createData()
        self._rnea_q, self._rnea_v, self._rnea_a = (np.zeros(model.nv) for _ in range(3))

    @property
    def arm_joint_names(self) -> list[str]:
        """Names of the joints RMPFlow solves for, in c-space order."""
        return list(self._arm_joint_names)

    @property
    def joint_positions(self) -> np.ndarray:
        """Current integrator position [rad] over the arm joints."""
        return self._q.copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """Current integrator velocity [rad/s] over the arm joints."""
        return self._q_dot.copy()

    def reset(self, joint_positions: np.ndarray | None = None, joint_velocities: np.ndarray | None = None) -> None:
        """Seed the integrator and point the targets at the current pose.

        Port of ``IkControllerBase::on_activate`` plus
        ``BimanualIkController::OnSubclassActivate``. Hardware seeds from the measured joint
        state, so callers should pass the articulation's arm joint positions at environment reset;
        without them the XRDF's default configuration is used.

        Args:
            joint_positions: Arm joint positions [rad] to resume from, in :attr:`arm_joint_names`
                order.
            joint_velocities: Arm joint velocities [rad/s] to resume from. Defaults to zero.
        """
        if joint_positions is None:
            self._q = np.asarray(self._robot_description.default_cspace_configuration(), dtype=np.float64).copy()
        else:
            self._q = self._as_arm_vector(joint_positions, "joint_positions")
        self._q_dot = (
            np.zeros(self._num_dofs) if joint_velocities is None
            else self._as_arm_vector(joint_velocities, "joint_velocities")
        )

        # Targets start at the current end-effector poses so the arms hold still until commanded.
        for ee_frame in self._ee_frames.values():
            self._rmpflow.set_pose_target(ee_frame, self._kinematics.pose(self._q, ee_frame))
        self._rmpflow.set_cspace_attractor(self._cspace_attractor)
        self.in_warmup = False

    def _as_arm_vector(self, values: np.ndarray, name: str) -> np.ndarray:
        """Validate and copy a 14-vector over the arm joints."""
        values = np.asarray(values, dtype=np.float64)
        assert values.shape == (self._num_dofs,), (
            f"{name} must have shape ({self._num_dofs},), got {values.shape}"
        )
        return values.copy()

    def _sync_integrator_to_hardware(
        self, measured_positions: np.ndarray | None, measured_velocities: np.ndarray | None
    ) -> bool:
        """Hard-snap the integrator to the measured state once it has drifted too far.

        Port of ``BimanualIkController::SyncIntegratorToHardware``: the integrator runs open-loop
        and is only corrected episodically, on an L2 drift threshold.

        Args:
            measured_positions: Measured arm joint positions [rad], or None to stay open-loop.
            measured_velocities: Measured arm joint velocities [rad/s]. Defaults to zero on snap.

        Returns:
            Whether the integrator was snapped.
        """
        if measured_positions is None:
            return False
        measured_positions = self._as_arm_vector(measured_positions, "measured_joint_positions")
        if float(np.linalg.norm(self._q - measured_positions)) <= self._drift_reset_threshold:
            return False
        self._q = measured_positions
        self._q_dot = (
            np.zeros(self._num_dofs) if measured_velocities is None
            else self._as_arm_vector(measured_velocities, "measured_joint_velocities")
        )
        return True

    def _set_pose_targets(self, body_target_pose: dict[str, np.ndarray]) -> None:
        """Push validated commanded poses to RMPFlow, rebased onto the controlled frames.

        Mirrors ``BimanualIkController::ApplyPoseTargetsToRmpflow``: a missing or invalid pose
        leaves that side's previous target in place rather than clearing it.

        Args:
            body_target_pose: Map of frame name to a 4x4 homogeneous target transform in the robot
                base frame, keyed by either the command frame or the end-effector frame.
        """
        for side, ee_frame in self._ee_frames.items():
            command_frame = self._command_frames[side]
            is_command_frame = command_frame in body_target_pose
            target = body_target_pose.get(command_frame) if is_command_frame else body_target_pose.get(ee_frame)
            if target is None:
                continue
            target = np.asarray(target, dtype=np.float64)
            if not _is_reference_pose_valid(target):
                continue
            base_to_ee = self._cumotion.Pose3(target)
            # base_T_ee = base_T_cmd * cmd_T_ee, as in ``ExtractAndTransformPose``.
            if is_command_frame:
                base_to_ee = base_to_ee * self._command_to_ee[side]
            self._rmpflow.set_pose_target(ee_frame, base_to_ee)

    def _step(self, compute_effort: bool = False) -> np.ndarray | None:
        """Advance the integrator by one control tick, as ``IkControllerBase::update()`` does."""
        self._rmpflow.eval_accel(self._q, self._q_dot, self._q_ddot)
        effort = None
        if compute_effort:
            assert np.isfinite([self._q, self._q_dot, self._q_ddot]).all()
            self._rnea_q[self._rnea_indices] = self._q
            self._rnea_v[self._rnea_indices] = self._q_dot
            self._rnea_a[self._rnea_indices] = self._q_ddot
            effort = self._pin.rnea(self._rnea_model, self._rnea_data,
                                    self._rnea_q, self._rnea_v, self._rnea_a)[self._rnea_indices].copy()
            assert np.isfinite(effort).all()
        self._q_dot = self._q_dot + self._control_dt * self._q_ddot
        self._q = self._q + self._control_dt * self._q_dot
        return effort

    def latch_targets(self, body_target_pose: dict[str, np.ndarray],
                      left_hand_state: float, right_hand_state: float) -> np.ndarray:
        """Latch the 50 Hz command without advancing the integrator or inverse dynamics."""
        self._set_pose_targets(body_target_pose)
        return self._body_joint_positions(left_hand_state, right_hand_state)

    def step_with_feedback(self, measured_positions: np.ndarray,
                           measured_velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance one 5 ms tick; effort uses the pre-integration q/v and same acceleration."""
        assert hasattr(self, "_rnea_model") and self._control_dt == DEFAULT_CONTROL_DT
        measured_positions = self._as_arm_vector(measured_positions, "measured_positions")
        measured_velocities = self._as_arm_vector(measured_velocities, "measured_velocities")
        assert np.isfinite([measured_positions, measured_velocities]).all()
        self._sync_integrator_to_hardware(measured_positions, measured_velocities)
        effort = self._step(compute_effort=True)
        assert np.isfinite([self._q, self._q_dot]).all()
        return self._q.copy(), effort

    def _body_joint_positions(self, left_hand_state: float, right_hand_state: float) -> np.ndarray:
        """Use the shared full-body ordering and existing mirrored binary finger targets."""
        body_q = np.zeros(self._robot_model.num_dofs)
        body_q[self._arm_full_indices] = self._q
        body_q[self._left_hand_full_indices] = _hand_joint_positions(left_hand_state)
        body_q[self._right_hand_full_indices] = -_hand_joint_positions(right_hand_state)
        return body_q

    def inverse_kinematics(
        self,
        body_target_pose: dict[str, np.ndarray],
        left_hand_state: float,
        right_hand_state: float,
        measured_joint_positions: np.ndarray | None = None,
        measured_joint_velocities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve for the joint positions that track the commanded end-effector poses.

        Runs :attr:`substeps` control ticks against the held target, matching the number of
        hardware controller ticks per commanded action.

        Args:
            body_target_pose: Map of frame name to a 4x4 homogeneous target transform in the robot
                base frame.
            left_hand_state: 0 to open the left hand, non-zero to close it.
            right_hand_state: 0 to open the right hand, non-zero to close it.
            measured_joint_positions: Measured arm joint positions [rad], in
                :attr:`arm_joint_names` order. Enables the drift hard-snap that hardware performs;
                omitting it leaves the integrator fully open-loop.
            measured_joint_velocities: Measured arm joint velocities [rad/s], used only when a
                snap occurs.

        Returns:
            Full-robot joint position vector [rad], with the arm and hand entries populated. Joints
            outside the arms-only model (legs, waist) are left at zero and are not read by the
            upper-body action term.
        """
        self._sync_integrator_to_hardware(measured_joint_positions, measured_joint_velocities)
        self._set_pose_targets(body_target_pose)
        for _ in range(self._substeps):
            self._step()

        return self._body_joint_positions(left_hand_state, right_hand_state)
