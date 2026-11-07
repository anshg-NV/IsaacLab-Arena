# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check joint/gripper correction bounds and native, RSL, and ONNX checkpoint parity."""

import json
import math
import numpy as np
import torch
from tensordict import TensorDict

import pytest
from robomimic.models.residual_policy_nets import ResidualWristPolicy, checkpoint_sha256

from isaaclab_arena.policy.residual_joint_policy import ResidualJointPolicy, apply_joint_residual
from isaaclab_arena.policy.residual_rsl_model import ResidualRslModel, export_residual_checkpoint


def test_joint_residual_bounds_and_zero_initialization():
    model = ResidualJointPolicy(feature_dim=28, horizon=3, input_mode="blind")
    base = torch.randn(2, 3, 16)
    features = torch.randn(2, 28)
    torch.testing.assert_close(model(features, base), base, rtol=0, atol=0)
    raw = torch.tensor([[100.0, -100.0] + [100.0, -100.0] * 7]).expand(2, -1)
    expected_delta = torch.tensor([[0.5, -0.5] + [0.1, -0.1] * 7]).expand(2, -1)
    corrected = model.combine(base, raw)
    torch.testing.assert_close(corrected - base, expected_delta[:, None].expand_as(base))
    torch.testing.assert_close(apply_joint_residual(base[:, 0], raw), corrected[:, 0])
    distribution, value = model.distribution(features, base)
    assert distribution.sample().shape == (2, 16) and value.shape == (2,)


@pytest.mark.parametrize(
    "action_space,input_mode,feature_dim",
    [
        ("joint", "blind", 28),
        ("joint", "vision", 256),
        ("eef", "vision", 220),
    ],
)
def test_rsl_checkpoint_roundtrip_and_actor_initialization(tmp_path, action_space, input_mode, feature_dim):
    joint = action_space == "joint"
    model = (
        ResidualJointPolicy(feature_dim, horizon=3, input_mode=input_mode)
        if joint
        else ResidualWristPolicy(feature_dim)
    )
    base_path = tmp_path / "base.pth"
    base_path.write_bytes(b"frozen diffusion checkpoint")
    base_sha = checkpoint_sha256(str(base_path))
    horizon = model.config["horizon"]
    base = torch.randn(2, horizon, 16 if joint else 23)
    features = torch.randn(2, feature_dim)
    obs = TensorDict({"policy": torch.cat((features, base.flatten(1)), -1)}, batch_size=[2])
    kwargs = dict(obs=obs, obs_groups={"actor": ["policy"], "critic": ["policy"]}, action_space=action_space)
    distribution_cfg = dict(class_name="GaussianDistribution", init_std=math.exp(-1), std_type="log")
    actor = ResidualRslModel(
        **kwargs, obs_set="actor", output_dim=16 if joint else 6, distribution_cfg=distribution_cfg.copy()
    )
    critic = ResidualRslModel(**kwargs, obs_set="critic", output_dim=1)
    # A small optimization step exercises the actual RSL Gaussian actor and critic parameters.
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=0.01)
    loss = (actor.mlp(obs["policy"]) - 1).square().mean() + critic.mlp(obs["policy"]).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    checkpoint, output = tmp_path / "model_1.pt", tmp_path / "residual.pt"
    torch.save(
        {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(), "iter": 1}, checkpoint
    )
    (tmp_path / "residual_context.json").write_text(
        json.dumps({
            "format_version": 1,
            "config": model.config,
            "base_sha256": base_sha,
            "action_space": action_space,
            "input_mode": input_mode,
        })
    )
    exported = export_residual_checkpoint(str(checkpoint), str(output), str(base_path))
    loaded = type(model).load(str(output), "cpu", base_sha)
    expected = model.combine(base, actor.mlp(obs["policy"]))
    torch.testing.assert_close(exported(features, base), expected)
    torch.testing.assert_close(loaded(features, base), expected)
    torch.testing.assert_close(loaded.critic(obs["policy"]), critic.mlp(obs["policy"]))
    initialized = ResidualRslModel(
        **kwargs,
        obs_set="actor",
        output_dim=16 if joint else 6,
        distribution_cfg=distribution_cfg.copy(),
        actor_init_checkpoint=str(output),
        actor_init_sha256=checkpoint_sha256(str(output)),
        actor_init_base_sha256=base_sha,
    )
    torch.testing.assert_close(initialized.mlp(obs["policy"]), actor.mlp(obs["policy"]))
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        type(model).load(str(output), "cpu", "wrong base")


def test_joint_residual_onnx_matches_native_with_continuous_grippers(tmp_path):
    import onnxruntime as ort

    model = ResidualJointPolicy(feature_dim=28, horizon=3, input_mode="blind").eval()
    with torch.no_grad():
        model.actor[-1].bias.fill_(0.8)
        model.actor[-1].bias[1] = -0.8
    features = torch.randn(1, 28)
    base = torch.randn(1, 3, 16)
    base[..., :2] = torch.tensor([-0.3, 0.0])
    destination = tmp_path / "joint_residual.onnx"
    torch.onnx.export(
        model,
        (features, base),
        str(destination),
        input_names=["features", "base"],
        output_names=["action"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    result = session.run(["action"], {"features": features.numpy(), "base": base.numpy()})[0]
    np.testing.assert_allclose(result, model(features, base).detach().numpy(), atol=1e-6, rtol=1e-5)
    assert result.shape == (1, 3, 16)
    assert np.all(result[..., 0] > -0.25) and np.all(result[..., 1] < -0.25)
    assert np.all(result[..., :2] != 0.0) and np.all(result[..., :2] != -0.5)
