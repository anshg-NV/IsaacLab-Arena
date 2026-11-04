# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from types import SimpleNamespace

import torch
import warp as wp

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_box_lift_success(simulation_app) -> bool:
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply

    from isaaclab_arena.tasks.hubble_box_lift import box_lift_success, reset_box_lift_success

    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, -20.0, 3.0]])

    def make_state():
        scene = Scene()
        env = SimpleNamespace(scene=scene, num_envs=2, device="cpu", step_dt=0.02,
                              episode_length_buf=torch.zeros(2, dtype=torch.long))
        for name in ("apple", "box"):
            scene[name] = SimpleNamespace(data=SimpleNamespace(
                root_pos_w=wp.from_torch(scene.env_origins + torch.tensor([0.0, 0.0, 0.8232])),
                root_quat_w=wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)),
                root_lin_vel_w=wp.from_torch(torch.zeros(2, 3)),
                root_ang_vel_w=wp.from_torch(torch.zeros(2, 3)),
            ))
        wp.to_torch(scene["apple"].data.root_pos_w)[:, 2] -= 0.03
        for name, force in (("pick_up_object_contact_sensor", 1.0), ("apple_hand_contact_sensor", 0.0),
                            ("box_left_hand_contact_sensor", 2.0), ("box_right_hand_contact_sensor", 2.0)):
            values = torch.zeros(2, 1, 2, 3)
            values[:, 0, 0, 2] = force
            scene[name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=wp.from_torch(values)))
        params = {
            "object_cfg": SceneEntityCfg("apple"), "box_cfg": SceneEntityCfg("box"),
            "support_surface_z": 0.75, "object_center_offset_m": (0.0, 0.0, 0.0), "object_radius_m": 0.035,
            "box_corners_m": tuple(itertools.product((-0.099, 0.099), (-0.148, 0.148), (-0.0732, 0.0732))),
        }
        assert not box_lift_success(env, **params).any()
        return env, env._hubble_box_lift_success, params

    def advance(env, term, params, steps=1):
        for _ in range(steps):
            env.episode_length_buf += 1
            result = box_lift_success(env, **params)
        return result

    def lift(env, height=0.12):
        for name in ("box", "apple"):
            wp.to_torch(env.scene[name].data.root_pos_w)[:, 2] += height

    # Correct order and duration, including a translated parallel environment.
    env, term, params = make_state()
    assert not advance(env, term, params, 10).any()
    assert term.placed.all()
    lift(env)
    assert not advance(env, term, params, 19).any()
    assert advance(env, term, params).all()
    reset_box_lift_success(env, [0])
    assert not term.placed[0] and term.placed[1]
    assert torch.isinf(term.contact_age[0]).all() and torch.isfinite(term.contact_age[1]).all()
    assert advance(env, term, params).tolist() == [False, True]
    reset_box_lift_success(env, None)
    assert not term.placed.any() and not term.lift_steps.any()

    # Lifting the loaded box first cannot satisfy the placement phase.
    env, term, params = make_state()
    lift(env)
    assert not advance(env, term, params, 50).any()
    assert not term.placed.any()

    # Repeated success queries during one simulation step must not accumulate time.
    env, term, params = make_state()
    for _ in range(30):
        assert not term(env, **params).any()
    assert not term.placed.any()

    # Every physical condition is necessary, and a broken hold resets its timer.
    for invalid in ("left", "right", "held_apple", "empty", "no_support", "rim", "tilted", "low"):
        env, term, params = make_state()
        advance(env, term, params, 10)
        lift(env, 0.02 if invalid == "low" else 0.12)
        if invalid in ("left", "right", "no_support", "held_apple"):
            name = {"left": "box_left_hand_contact_sensor", "right": "box_right_hand_contact_sensor",
                    "no_support": "pick_up_object_contact_sensor", "held_apple": "apple_hand_contact_sensor"}[invalid]
            wp.to_torch(env.scene[name].data.force_matrix_w)[:] = 1.0 if invalid == "held_apple" else 0.0
        elif invalid in ("empty", "rim"):
            wp.to_torch(env.scene["apple"].data.root_pos_w)[:, 0] += 0.3 if invalid == "empty" else 0.08
        elif invalid == "tilted":
            wp.to_torch(env.scene["box"].data.root_quat_w)[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert not advance(env, term, params, 30).any(), invalid
        assert not term.lift_steps.any(), invalid

    # A modest 6 cm lift clears the table; 10 cm used to be unnecessarily high.
    env, term, params = make_state()
    advance(env, term, params, 10)
    lift(env, 0.06)
    assert advance(env, term, params, 20).all(), "Six-centimeter lift was rejected"

    # Preserve the user's 40-degree lift tolerance while rejecting larger tilts.
    for angle_deg, expected in ((39.0, True), (41.0, False)):
        env, term, params = make_state()
        advance(env, term, params, 10)
        lift(env, 0.2)
        half_angle = math.radians(angle_deg) / 2
        quat = torch.tensor([math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)]).repeat(2, 1)
        wp.to_torch(env.scene["box"].data.root_quat_w)[:] = quat
        wp.to_torch(env.scene["apple"].data.root_pos_w)[:] = (
            wp.to_torch(env.scene["box"].data.root_pos_w)
            + quat_apply(quat, torch.tensor([[0.0, 0.0, -0.03]]).repeat(2, 1))
        )
        assert advance(env, term, params, 20).tolist() == [expected, expected]

    # A real settled apple at the KLT wall exceeded the sphere-inset boundary
    # by only 0.06 mm. It must count as loaded, including during the lift.
    env, term, params = make_state()
    params["object_radius_m"] = 0.04000124
    wp.to_torch(env.scene["apple"].data.root_pos_w)[:, 0] += -0.04755735
    advance(env, term, params, 10)
    assert term.placed.all(), "Apple resting against the inner wall was rejected"
    lift(env)
    assert advance(env, term, params, 20).all()

    # Normal placement/lift motion is allowed; object geometry and contacts
    # distinguish a loaded lift from throwing the box or holding an empty box.
    env, term, params = make_state()
    wp.to_torch(env.scene["apple"].data.root_lin_vel_w)[:, 2] = 0.2
    advance(env, term, params, 10)
    assert term.placed.all(), "Moving apple cannot register placement"
    lift(env)
    for name in ("apple", "box"):
        wp.to_torch(env.scene[name].data.root_lin_vel_w)[:, 2] = 0.3
    wp.to_torch(env.scene["box"].data.root_ang_vel_w)[:, 2] = 1.0
    assert advance(env, term, params, 20).all(), "Moving two-hand lift was rejected"

    # A few missing contact samples cannot reset a valid grasp, but losing a
    # hand for longer than the contact grace interval must invalidate it.
    env, term, params = make_state()
    advance(env, term, params, 10)
    lift(env)
    for step in range(20):
        for name in ("pick_up_object_contact_sensor", "box_left_hand_contact_sensor", "box_right_hand_contact_sensor"):
            wp.to_torch(env.scene[name].data.force_matrix_w)[:] = 0.0 if step % 5 in (2, 3) else 2.0
        result = advance(env, term, params)
    assert result.all(), "Brief contact dropout reset the lift timer"
    wp.to_torch(env.scene["box_left_hand_contact_sensor"].data.force_matrix_w)[:] = 0.0
    assert not advance(env, term, params, 10).any()
    assert not term.lift_steps.any()

    # A hand's load may be distributed over multiple finger links.
    env, term, params = make_state()
    advance(env, term, params, 10)
    lift(env)
    for name in ("box_left_hand_contact_sensor", "box_right_hand_contact_sensor"):
        force = wp.to_torch(env.scene[name].data.force_matrix_w)
        force[:] = 0.0
        force[..., 2] = 0.3
    assert advance(env, term, params, 20).all()

    # Containment follows box yaw and the apple's off-center root.
    env, term, params = make_state()
    quat = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]).repeat(2, 1)
    wp.to_torch(env.scene["box"].data.root_quat_w)[:] = quat
    wp.to_torch(env.scene["apple"].data.root_quat_w)[:] = quat
    params["object_center_offset_m"] = (0.01, 0.0, 0.0)
    wp.to_torch(env.scene["apple"].data.root_pos_w)[:] = (
        wp.to_torch(env.scene["box"].data.root_pos_w)
        + quat_apply(quat, torch.tensor([[-0.01, 0.07, -0.03]]).repeat(2, 1))
    )
    advance(env, term, params, 10)
    lift(env)
    assert advance(env, term, params, 20).all()
    wp.to_torch(env.scene["box_left_hand_contact_sensor"].data.force_matrix_w)[:] = 0.0
    assert not advance(env, term, params, 10).any()
    assert not term.lift_steps.any()
    return True


def test_box_lift_success():
    assert run_simulation_app_function(_test_box_lift_success)


def _test_box_lift_configuration(simulation_app) -> bool:
    import argparse

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.tasks.hubble_box_lift import HubbleBoxLiftMimicEnvCfg
    from isaaclab_arena_environments.hubble_box_lift_environment import HubbleBoxLiftEnvironment

    assert EnvironmentRegistry().get_component_by_name("hubble_box_lift") is HubbleBoxLiftEnvironment
    parser = argparse.ArgumentParser()
    HubbleBoxLiftEnvironment.add_cli_args(parser)
    args = parser.parse_args([])
    assert args.destination == "purple_box" and args.object == "apple_01_objaverse_robolab"
    box = HubbleBoxLiftEnvironment()._make_destination(args)
    assert box.object_cfg.spawn.mass_props.mass == 1.12
    cfg = HubbleBoxLiftMimicEnvCfg(
        arm_mode=ArmMode.DUAL_ARM, pick_up_object_name=args.object, destination_location_name=args.destination,
    )
    assert cfg.datagen_config.name == "hubble_box_lift_D0"
    assert [step.subtask_term_signal for step in cfg.subtask_configs["left"]] == [
        "grasp_apple", "place_apple", "grasp_box_left", None,
    ]
    assert [step.subtask_term_signal for step in cfg.subtask_configs["right"]] == [
        "grasp_box_right", None,
    ]
    constraint = cfg.task_constraint_configs[0]
    assert constraint.coordination_synchronize_start
    assert constraint.eef_subtask_constraint_tuple == (("left", 3), ("right", 1))
    return True


def test_box_lift_configuration():
    assert run_simulation_app_function(_test_box_lift_configuration)
