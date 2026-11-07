# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check joint ordering, checkpoint normalization, and ONNX action layouts without simulation."""

import numpy as np
import sys
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.policy import robomimic_stand_policy as stand
from isaaclab_arena.policy.deployment_timing import DeploymentTiming
from isaaclab_arena.policy.leapp_onnx_stand_policy import LeappOnnxStandPolicy, LeappOnnxStandPolicyCfg
from isaaclab_arena.policy.robomimic_stand_joint_policy import RobomimicStandJointPolicy, RobomimicStandJointPolicyCfg
from isaaclab_arena.scripts.imitation_learning.convert_dataset_to_joint_actions import G1_SIM_JOINT_NAMES
from isaaclab_arena_g1.g1_env.g1_arm_joints import G1_ARM_JOINT_NAMES, G1_JOINT_ACTION_ELEMENT_NAMES


def _joint_policy(monkeypatch, stats=None, relative=False, **config_kwargs):
    modalities = {
        "low_dim": ["left_arm_joint_pos", "right_arm_joint_pos"],
        "rgb": ["robot_head_cam_rgb"],
        "depth": ["robot_head_cam_depth"],
    }

    def trajectory(obs_dict, **kwargs):
        left, right = obs_dict["left_arm_joint_pos"][:, -1], obs_dict["right_arm_joint_pos"][:, -1]
        row = torch.cat((torch.full_like(left[:, :1], -0.3), torch.zeros_like(left[:, :1]), left, right), dim=-1)
        return row[:, None].repeat(1, 2, 1)

    algorithm = SimpleNamespace(
        ac_dim=16,
        rel_joint_mode=relative,
        _get_action_trajectory=trajectory,
        nets=torch.nn.Linear(1, 1),
        ema=None,
        algo_config=SimpleNamespace(horizon=SimpleNamespace(action_horizon=2)),
        global_config=SimpleNamespace(
            observation=SimpleNamespace(modalities=SimpleNamespace(obs=modalities)),
            algo={"horizon": {"observation_horizon": 2}},
            train={"frame_stack": 2, "action_keys": ["actions"]},
        ),
    )
    wrapper = SimpleNamespace(
        policy=algorithm,
        action_normalization_stats=stats,
        _prepare_observation=lambda obs, **kwargs: obs,
        start_episode=lambda: None,
    )
    monkeypatch.setattr(stand.FileUtils, "policy_from_checkpoint", lambda **kwargs: (wrapper, {}))
    monkeypatch.setattr(stand, "version", lambda package: "test")
    config_kwargs.setdefault("robomimic_checkpoint", "unused")
    return RobomimicStandJointPolicy(RobomimicStandJointPolicyCfg(device="cpu", **config_kwargs))


def _joint_env():
    # Reordering the articulation must not reorder the policy's canonical arm vector.
    names = list(reversed(G1_SIM_JOINT_NAMES))
    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            num_envs=2,
            device="cpu",
            step_dt=0.02,
            action_manager=SimpleNamespace(total_action_dim=50),
            scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_names=names))},
        )
    )


def test_joint_policy_observations_order_and_partial_reset(monkeypatch):
    policy, env = _joint_policy(monkeypatch), _joint_env()
    assert PolicyRegistry()._cfg_types[RobomimicStandJointPolicy] is RobomimicStandJointPolicyCfg
    assert policy._camera_obs_keys == ["robot_head_cam_rgb", "robot_head_cam_depth"]
    observation = {
        "policy": {
            "left_arm_joint_pos": torch.arange(14).reshape(2, 7).float() / 10,
            "right_arm_joint_pos": -torch.arange(14).reshape(2, 7).float() / 10,
        },
        "camera_obs": {
            "robot_head_cam_rgb": torch.zeros(2, 2, 2, 3, dtype=torch.uint8),
            "robot_head_cam_depth": torch.zeros(2, 2, 2, 1),
        },
    }
    action = policy.get_action(env, observation)
    names = env.unwrapped.scene["robot"].data.joint_names
    arm_ids = [names.index(name) for name in G1_ARM_JOINT_NAMES]
    expected = torch.cat(tuple(observation["policy"].values()), dim=-1)
    torch.testing.assert_close(action[:, arm_ids], expected)
    assert action.shape == (2, 50)
    torch.testing.assert_close(action[:, 46], torch.full((2,), 0.72))
    assert action[:, names.index("left_hand_index_0_joint")].tolist() == pytest.approx([-0.6, -0.6])
    assert not action[:, names.index("right_hand_index_0_joint")].any()
    assert not action[:, names.index("left_hip_pitch_joint")].any()

    policy.reset(torch.tensor([0]))
    observation["policy"]["left_arm_joint_pos"] += 0.2
    next_action = policy.get_action(env, observation)
    torch.testing.assert_close(next_action[0, arm_ids[:7]], expected[0, :7] + 0.2)
    torch.testing.assert_close(next_action[1, arm_ids[:7]], expected[1, :7])


def test_joint_export_restores_action_units_before_expanding_hands(monkeypatch):
    stats = {"actions": {"scale": np.full((1, 16), 2.0), "offset": np.full((1, 16), 0.1)}}
    policy = _joint_policy(monkeypatch, stats=stats)
    env = _joint_env()
    policy._ensure_joint_slots(env)
    raw = torch.full((2, 3, 16), 0.6)
    raw[..., 0] = -0.2  # Restores to -0.3, so the left hand must close.
    raw[..., 1] = 0.0
    physical = policy._prepare_action_chunk(raw)
    exported = policy._export_action_chunk(physical)
    names = env.unwrapped.scene["robot"].data.joint_names
    arm_ids = [names.index(name) for name in G1_ARM_JOINT_NAMES]
    torch.testing.assert_close(exported[..., 2:], torch.full((2, 3, 14), 1.3))
    torch.testing.assert_close(exported[..., 0], torch.full((2, 3), -0.3))
    assert exported.shape == (2, 3, 16)
    assert policy.action_element_names == list(G1_JOINT_ACTION_ELEMENT_NAMES)
    assembled = policy._assemble_action(physical[:, 0])
    torch.testing.assert_close(assembled[:, arm_ids], exported[:, 0, 2:])
    torch.testing.assert_close(assembled[:, names.index("left_hand_index_0_joint")], torch.full((2,), -0.6))


def test_joint_policy_rejects_wrist_metrics():
    with pytest.raises(AssertionError, match="Wrist-translation metrics"):
        RobomimicStandJointPolicy(
            RobomimicStandJointPolicyCfg(
                robomimic_checkpoint="unused",
                record_residual_metrics=True,
            )
        )


def test_joint_relative_policy_rejects_double_normalization(monkeypatch):
    stats = {"actions": {"scale": np.full((1, 16), 2.0), "offset": np.zeros((1, 16))}}
    with pytest.raises(AssertionError, match="unnormalized action labels"):
        _joint_policy(monkeypatch, stats=stats, relative=True)


@pytest.mark.parametrize("correct_grippers", [False, True])
@torch.inference_mode()
def test_joint_residual_loads_and_overrides_grippers_after_unnormalizing(monkeypatch, tmp_path, correct_grippers):
    from robomimic.models.residual_policy_nets import checkpoint_sha256

    from isaaclab_arena.policy.residual_joint_policy import ResidualJointPolicy

    base_path, residual_path = tmp_path / "base.pth", tmp_path / "residual.pth"
    base_path.write_bytes(b"frozen base")
    residual = ResidualJointPolicy(feature_dim=28, horizon=2, input_mode="blind")
    if correct_grippers:
        with torch.no_grad():
            residual.actor[-1].bias.fill_(2.0)
            residual.actor[-1].bias[1] = -2.0
    residual.save(str(residual_path), checkpoint_sha256(str(base_path)))
    stats = {"actions": {"scale": np.full((1, 16), 2.0), "offset": np.full((1, 16), 0.1)}}
    policy = _joint_policy(
        monkeypatch,
        stats=stats,
        robomimic_checkpoint=str(base_path),
        residual_model_path=str(residual_path),
        cache_residual_inputs=True,
    )
    env = _joint_env()
    observation = {
        "policy": {"left_arm_joint_pos": torch.zeros(2, 7), "right_arm_joint_pos": torch.ones(2, 7)},
        "camera_obs": {
            "robot_head_cam_rgb": torch.zeros(2, 2, 2, 3, dtype=torch.uint8),
            "robot_head_cam_depth": torch.zeros(2, 2, 2, 1),
        },
    }
    action = policy.get_action(env, observation)
    inputs = policy.residual_inputs
    features, base = inputs[:, :28], inputs[:, 28:].reshape(2, 2, 16)
    # Training sees physical units and continuous grippers, with no double unnormalization.
    torch.testing.assert_close(base[:, 0, :2], torch.tensor([[-0.5, 0.1]]).expand(2, 2))
    torch.testing.assert_close(base[:, 0, 2:9], torch.full((2, 7), 0.1))
    torch.testing.assert_close(base[:, 0, 9:], torch.full((2, 7), 2.1))
    expected = residual(features, base)
    torch.testing.assert_close(policy.raw_action, expected[:, 0])
    torch.testing.assert_close(policy._export_action_chunk(expected), expected)
    names = env.unwrapped.scene["robot"].data.joint_names
    torch.testing.assert_close(action[:, [names.index(name) for name in G1_ARM_JOINT_NAMES]], expected[:, 0, 2:])
    assert action[0, names.index("left_hand_index_0_joint")] == pytest.approx(0.0 if correct_grippers else -0.6)
    assert action[0, names.index("right_hand_index_0_joint")] == pytest.approx(0.6 if correct_grippers else 0.0)
    assert all(not parameter.requires_grad for parameter in policy._policy.policy.nets.parameters())

    policy.reset(torch.tensor([0]))
    observation["policy"]["left_arm_joint_pos"] += 0.2
    policy.get_action(env, observation)
    torch.testing.assert_close(policy.raw_action[0, 2:9], expected[0, 0, 2:9] + 0.4)
    torch.testing.assert_close(policy.raw_action[1], expected[1, 1])


def test_eef_export_keeps_continuous_grippers():
    policy = object.__new__(stand.RobomimicStandPolicy)
    policy._base_height = 0.72
    raw = torch.zeros(1, 3, 23)
    raw[..., :2] = torch.tensor([-0.3, -0.2])
    exported = policy._export_action_chunk(raw)
    torch.testing.assert_close(exported[..., :2], raw[..., :2])
    torch.testing.assert_close(policy._assemble_action(raw)[..., :2], torch.tensor([[[-0.5, 0.0]]]).expand(1, 3, 2))


def _onnx_policy(monkeypatch, tmp_path, joint, height, width, with_depth=False):
    def port(name, shape, dtype="tensor(float)"):
        return SimpleNamespace(name=name, shape=shape, type=dtype)

    low_dim = (
        {"left_arm_joint_pos": (7,), "right_arm_joint_pos": (7,)}
        if joint
        else {
            "left_eef_pos": (3,),
            "left_eef_quat": (4,),
            "left_wrist_pose_pelvis_frame": (4, 4),
            "right_eef_pos": (3,),
            "right_eef_quat": (4,),
            "right_wrist_pose_pelvis_frame": (4, 4),
        }
    )
    inputs = [port(name, (1, 2, *shape)) for name, shape in low_dim.items()]
    inputs += [
        port("robot_head_cam_rgb", (1, 2, height, width, 3), "tensor(uint8)"),
        port("diffusion_noise", (1, 16, 16 if joint else 21)),
    ]
    if with_depth:
        inputs.append(port("robot_head_cam_depth", (1, 2, height, width, 1)))
    output = np.zeros((1, 8 if joint else 6, 16 if joint else 23), dtype=np.float32)
    output[..., :2] = (-0.3, 0.2)
    if joint:
        output[..., 2:] = np.arange(14) / 10
    calls = []

    def run(names, feed):
        calls.append(feed)
        return [output]

    session = SimpleNamespace(
        get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        get_provider_options=lambda: {},
        disable_fallback=lambda: None,
        get_inputs=lambda: inputs,
        get_outputs=lambda: [port("action", output.shape)],
        run=run,
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            __version__="test",
            get_available_providers=session.get_providers,
            SessionOptions=SimpleNamespace,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=0),
            InferenceSession=lambda *args, **kwargs: session,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(
            load=lambda *args, **kwargs: SimpleNamespace(
                graph=SimpleNamespace(output=[port("action", output.shape)], initializer=[])
            ),
        ),
    )
    model = tmp_path / "model.onnx"
    model.write_bytes(b"mock graph")
    policy = LeappOnnxStandPolicy(LeappOnnxStandPolicyCfg(onnx_model_path=str(model)))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            num_envs=1,
            device="cpu",
            step_dt=0.02,
            action_manager=SimpleNamespace(total_action_dim=50 if joint else 23),
            scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_names=list(reversed(G1_SIM_JOINT_NAMES))))},
        )
    )
    camera = torch.arange(height * width * 3).remainder(256).to(torch.uint8).reshape(1, height, width, 3)
    obs = {
        "policy": {name: torch.zeros(1, *shape) for name, shape in low_dim.items()},
        "camera_obs": {"robot_head_cam_rgb": camera},
    }
    if with_depth:
        obs["camera_obs"]["robot_head_cam_depth"] = torch.linspace(0, 1, height * width).reshape(1, height, width, 1)
    return policy, env, obs, output, calls


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("height,width", [(240, 320), (480, 640)])
@pytest.mark.parametrize("with_depth", [False, True])
def test_onnx_consumes_each_action_layout_at_both_resolutions(monkeypatch, tmp_path, joint, height, width, with_depth):
    policy, env, obs, output, calls = _onnx_policy(monkeypatch, tmp_path, joint, height, width, with_depth)
    action = policy.get_action(env, obs)
    if joint:
        names = env.unwrapped.scene["robot"].data.joint_names
        assert action.shape == (1, 50)
        assert action[0, names.index("left_hand_index_0_joint")] == pytest.approx(-0.6)
        assert action[0, names.index("right_hand_index_0_joint")] == 0
        np.testing.assert_allclose(action[0, [names.index(name) for name in G1_ARM_JOINT_NAMES]], output[0, 0, 2:])
    else:
        assert action[0, :2].tolist() == pytest.approx([-0.5, 0.0])
    assert calls[0]["robot_head_cam_rgb"].shape == (1, 2, height, width, 3)
    assert calls[0]["diffusion_noise"].shape == (1, 16, 16 if joint else 21)
    if with_depth:
        depth = obs["camera_obs"]["robot_head_cam_depth"].numpy()
        assert calls[0]["robot_head_cam_depth"].dtype == np.float32
        np.testing.assert_array_equal(calls[0]["robot_head_cam_depth"], np.stack((depth, depth), axis=1))
    else:
        assert "robot_head_cam_depth" not in calls[0]
    assert output[0, 0, :2].tolist() == pytest.approx([-0.3, 0.2])
    policy.record_evaluation_step()
    torch.testing.assert_close(policy.get_action(env, obs), action)
    assert len(calls) == 1, "The second action should come from the existing chunk"


@pytest.mark.parametrize("joint", [False, True])
def test_onnx_depth_history_tracks_the_latest_two_frames(monkeypatch, tmp_path, joint):
    policy, env, obs, output, calls = _onnx_policy(monkeypatch, tmp_path, joint, 240, 320, with_depth=True)
    horizon = output.shape[1]
    for step in range(horizon + 1):
        obs["camera_obs"]["robot_head_cam_depth"].fill_(step / 10)
        policy.get_action(env, obs)
        policy.record_evaluation_step()
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1]["robot_head_cam_depth"][:, 0], (horizon - 1) / 10)
    np.testing.assert_allclose(calls[1]["robot_head_cam_depth"][:, 1], horizon / 10)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "nonfinite", "missing"])
def test_onnx_rejects_invalid_depth(monkeypatch, tmp_path, invalid):
    policy, env, obs, _, _ = _onnx_policy(monkeypatch, tmp_path, False, 240, 320, with_depth=True)
    camera = obs["camera_obs"]
    if invalid == "shape":
        camera["robot_head_cam_depth"] = torch.zeros(1, 480, 640, 1)
    elif invalid == "dtype":
        camera["robot_head_cam_depth"] = camera["robot_head_cam_depth"].double()
    elif invalid == "nonfinite":
        camera["robot_head_cam_depth"][0, 0, 0, 0] = float("nan")
    else:
        del camera["robot_head_cam_depth"]
    with pytest.raises(KeyError if invalid == "missing" else AssertionError, match="robot_head_cam_depth"):
        policy.get_action(env, obs)


def test_deployment_timing_samples_rgb_and_depth_together():
    timing = DeploymentTiming("fresh240", 1, 0.02)
    for tick, (camera_tick, joint_tick) in enumerate(((0, 0), (0, 0), (2, 0), (2, 3))):
        held = timing.observe({
            "robot_head_cam_rgb": torch.full((1, 1, 1, 3), tick, dtype=torch.uint8),
            "robot_head_cam_depth": torch.full((1, 1, 1, 1), tick / 10),
            "left_arm_joint_pos": torch.full((1, 7), float(tick)),
        })
        assert (held["robot_head_cam_rgb"] == camera_tick).all()
        torch.testing.assert_close(held["robot_head_cam_depth"], torch.full((1, 1, 1, 1), camera_tick / 10))
        assert (held["left_arm_joint_pos"] == joint_tick).all()
        timing.advance()


def test_onnx_rejects_camera_and_embodiment_mismatches(monkeypatch, tmp_path):
    policy, env, obs, _, _ = _onnx_policy(monkeypatch, tmp_path, True, 480, 640)
    obs["camera_obs"]["robot_head_cam_rgb"] = torch.zeros(1, 240, 320, 3, dtype=torch.uint8)
    with pytest.raises(AssertionError, match="matching ONNX input"):
        policy.get_action(env, obs)
    env.unwrapped.action_manager.total_action_dim = 23
    with pytest.raises(AssertionError, match="embodiment"):
        policy.get_action(env, obs)
