# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for camera relocation and G1 Mimic hand labels."""

import ast
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


ARENA = Path(__file__).resolve().parents[2]
SCRIPT = ARENA / "isaaclab_arena/scripts/imitation_learning/move_camera_obs.py"
spec = importlib.util.spec_from_file_location("move_camera_obs", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestMimicHandLabels(unittest.TestCase):
    def test_default_canonicalization_and_opt_out(self):
        for enabled in (False, True):
            with self.subTest(canonicalize=enabled), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "source.hdf5"
                actions = np.arange(8 * 23, dtype=np.float32).reshape(8, 23)
                actions[:, 0] = [-0.5, -0.2978, -0.25, -0.0944, -0.027, -0.00415, -1e-8, 0.0]
                actions[:, 1] = 0.0
                with h5py.File(source, "w") as f:
                    demo = f.create_group("data/demo_0")
                    demo.create_dataset("actions", data=actions)
                    demo.create_dataset("obs/actions", data=actions)
                    demo.create_dataset("camera_obs/robot_head_cam_rgb", data=np.ones((8, 2, 3, 3), np.uint8))
                before = hashlib.sha256(source.read_bytes()).digest()
                kwargs = {} if enabled else {"canonicalize_mimic_grippers": False}
                output = module.preprocess(str(source), "robot_head_cam_rgb", **kwargs)
                self.assertEqual(before, hashlib.sha256(source.read_bytes()).digest())
                with h5py.File(output, "r") as f:
                    demo = f["data/demo_0"]
                    actual = demo["actions"][:]
                    np.testing.assert_array_equal(actual[:, 2:], actions[:, 2:])
                    np.testing.assert_array_equal(demo["obs/actions"][:], actions)
                    self.assertNotIn("camera_obs", demo)
                    self.assertEqual(demo["obs/robot_head_cam_rgb"].shape, (8, 2, 3, 3))
                    expected = np.where(actions[:, :2] != 0, -0.5, 0.0) if enabled else actions[:, :2]
                    np.testing.assert_array_equal(actual[:, :2], expected)
                    if enabled:
                        self.assertEqual(f["data"].attrs["mimic_gripper_labels"], "native_nonzero_closed_to_binary_v1")
                        path = ARENA / "isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/g1_wbc_upperbody_ik/g1_wbc_upperbody_controller.py"
                        fn = next(n for n in ast.walk(ast.parse(path.read_text()))
                                  if isinstance(n, ast.FunctionDef) and n.name == "get_hand_joint_pos")
                        namespace = {"np": np}
                        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
                        for raw, label in zip(actions[:, 0], actual[:, 0]):
                            np.testing.assert_array_equal(namespace["get_hand_joint_pos"](None, raw),
                                                          namespace["get_hand_joint_pos"](None, label))

    def test_invalid_contract_rejected(self):
        for data in (np.zeros((2, 21), np.float32), np.zeros((2, 23), np.int32),
                     np.full((2, 23), np.nan), np.ones((2, 23)), np.full((2, 23), -0.6)):
            with self.subTest(shape=data.shape, dtype=data.dtype), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.hdf5"
                with h5py.File(path, "w") as f:
                    f.create_dataset("data/demo_0/actions", data=data)
                with self.assertRaises(AssertionError):
                    module.canonicalize_mimic_gripper_labels(str(path))


if __name__ == "__main__":
    unittest.main()
