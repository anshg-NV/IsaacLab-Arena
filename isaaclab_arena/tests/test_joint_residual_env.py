# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the joint residual training adapter through the real Isaac Lab wrapper API."""


def _test_joint_residual_env(simulation_app):
    import json
    import sys
    import torch
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    import pytest
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_arena.policy.residual_joint_policy import apply_joint_residual
    from isaaclab_arena.policy.residual_rsl_env import ResidualRslEnv, ResidualRslRunnerCfg
    from isaaclab_arena.tests.test_robomimic_joint_policy import _joint_env, _joint_policy

    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        base_path = Path(directory) / "base.pth"
        base_path.write_bytes(b"test frozen base")
        _joint_policy(patch)  # Install the checkpoint loader fixture; the adapter constructs its own policy.
        env = _joint_env()
        native = env.unwrapped
        native.cfg = SimpleNamespace(
            is_finite_horizon=True,
            residual_base=str(base_path),
            residual_input_mode="blind",
            log_dir=directory,
            sim=SimpleNamespace(use_fabric=False),
        )
        native.scene["robot"].body_names = ["left_hand_palm_link"]
        native.scene["robot"].data.body_pos_w = torch.zeros(2, 1, 3)
        for name in ("apple_01_objaverse_robolab", "clay_plates_hot3d_robolab"):
            native.scene[name] = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3)))
        native.action_manager.get_term = lambda name: SimpleNamespace(hand_target_contract={"profile": "current"})
        native.episode_recorder = SimpleNamespace(set_job_name=lambda name: None, set_output_path=lambda path: None)
        native.episode_length_buf = torch.zeros(2, dtype=torch.long)
        native.obs_buf = {
            "policy": {"left_arm_joint_pos": torch.zeros(2, 7), "right_arm_joint_pos": torch.ones(2, 7)},
            "camera_obs": {
                "robot_head_cam_rgb": torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3),
                "robot_head_cam_depth": torch.zeros(2, 2, 2, 1),
            },
        }
        env.action_space = object()
        commands = []

        def native_step(action):
            commands.append(action.clone())
            done = torch.tensor([len(commands) == 1, False])
            native.episode_length_buf += 1
            native.episode_length_buf[done] = 0
            return native.obs_buf, torch.zeros(2), done, torch.zeros(2, dtype=torch.bool), {}

        env.step = native_step

        def initialize_wrapper(self, env, clip_actions):
            self.env, self.clip_actions = env, clip_actions
            self.num_actions, self.num_envs, self.device = 50, 2, "cpu"

        patch.setattr(RslRlVecEnvWrapper, "__init__", initialize_wrapper)
        wrapper = ResidualRslEnv(env)
        assert wrapper.action_space is env.action_space  # Do not shadow the inherited read-only Gym property.
        assert wrapper.num_actions == 16 and wrapper.get_observations()["policy"].shape == (2, 60)
        torch.testing.assert_close(wrapper.base_action[:, :2], torch.tensor([[-0.3, 0.0]]).expand(2, 2))
        raw = torch.full((2, 16), 2.0)
        raw[:, 1] = -2
        expected = wrapper.policy._assemble_action(apply_joint_residual(wrapper.base_action, raw))
        previous_observation = wrapper.get_observations()["policy"].clone()
        observations, _, done, extras = wrapper.step(raw)
        torch.testing.assert_close(commands[0], expected)
        assert done.tolist() == [1, 0] and extras["residual_episode"]["length"].tolist() == [1]
        assert observations["policy"].shape == previous_observation.shape
        # Repeated steps must correct the unmodified base, not accumulate previous residuals.
        wrapper.step(raw)
        torch.testing.assert_close(commands[1], expected)
        context = json.loads((Path(directory) / "residual_context.json").read_text())
        assert context["action_space"] == "joint" and context["config"]["feature_dim"] == 28
        runner = ResidualRslRunnerCfg()
        runner.actor.action_space = runner.critic.action_space = "joint"
        assert runner.to_dict()["actor"]["action_space"] == runner.to_dict()["critic"]["action_space"] == "joint"

        # Exercise the second deployment consumer with a fake inference backend and real action conversion.
        patch.setitem(sys.modules, "leapp", SimpleNamespace(InferenceManager=object))
        patch.setitem(sys.modules, "leapp.utils.tensor_description", SimpleNamespace(map_to_torch_dtype=lambda x: x))
        module_name = "isaaclab_arena.evaluation.robomimic_stand_leapp_deployment_env"
        patch.setitem(sys.modules, module_name, None)
        del sys.modules[module_name]  # Restore the original module, if any, when the patch context exits.
        from isaaclab_arena.evaluation.robomimic_stand_leapp_deployment_env import RobomimicStandLeappDeploymentEnv
        from isaaclab_arena_g1.g1_env.g1_arm_joints import joint_action_to_wbc

        for action_dim in (16, 23):
            chunk = torch.zeros(2, 3, action_dim)
            chunk[..., :2] = torch.tensor([-0.3, 0.1])
            runner = object.__new__(RobomimicStandLeappDeploymentEnv)
            runner._step_count, runner._chunk_index, runner._base_height = 0, 0, 0.72
            runner._action_chunk = chunk
            runner._input_mapping, runner._last_obs, runner._pending_reset = {}, {}, []
            runner._obs_history = None
            runner.env = SimpleNamespace(
                device="cpu",
                scene=native.scene,
                action_manager=SimpleNamespace(total_action_dim=50 if action_dim == 16 else 23),
                step=lambda action: ({}, None, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}),
            )
            applied = runner.step()
            if action_dim == 16:
                expected = joint_action_to_wbc(chunk[:, 0], native.scene["robot"].data.joint_names, 0.72)
            else:
                expected = chunk[:, 0].clone()
                expected[:, :2] = torch.tensor([-0.5, 0.0])
            torch.testing.assert_close(applied, expected)
            torch.testing.assert_close(chunk[..., :2], torch.tensor([-0.3, 0.1]).expand(2, 3, 2))
    return True


def test_joint_residual_env():
    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    assert run_simulation_app_function(_test_joint_residual_env, headless=True)
