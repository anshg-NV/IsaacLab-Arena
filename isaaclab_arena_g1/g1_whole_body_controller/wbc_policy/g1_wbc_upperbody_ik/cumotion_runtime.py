# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Locate and import the cuMotion runtime used by the G1 upper-body RMPFlow controller.

Two different cuMotion builds are reachable from a typical Isaac Sim install, and only one of
them can drive a bimanual controller:

* Isaac Sim prebundles **cumotion 1.1.0a2** under
  ``isaacsim/exts/isaacsim.robot_motion.cumotion/pip_prebundle``. Its ``RmpFlow`` binds a single
  end-effector frame at config time and exposes only the deprecated
  ``set_end_effector_*_attractor`` calls, so it cannot express a two-arm target set.
* The G1 deployment stack links **cumotion 1.1.0**, which adds the multi-target-frame API
  (:meth:`add_target_frame`, :meth:`set_pose_target`, :meth:`clear_pose_target`) that
  ``BimanualIkController`` is built on.

This module resolves the 1.1.0 runtime and puts it ahead of any prebundled copy on ``sys.path``,
then verifies the multi-target API is actually present so a silent fall-back to 1.1.0a2 surfaces
as an explicit error rather than a mid-episode ``AttributeError``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

# Vendored runtime location, relative to the repository root. Matched by the repo's ``/__*``
# ignore rule, so the 71 MB unpacked wheel stays out of git.
_VENDORED_DIR_NAME = "__cumotion_runtime"

# Overrides the vendored location; useful once the runtime is installed properly.
_RUNTIME_DIR_ENV_VAR = "CUMOTION_RUNTIME_DIR"

# Where the matching wheel ships in the deployment checkout, quoted in the error message below.
_WHEEL_HINT = (
    "~/workspaces/isaac/ros_ws/src/isaac_ros_nitros/isaac_ros_nitros/lib/cumotion/"
    "x86_64_cuda_13_0/python_wheels/cumotion-1.1.0-cp312-cp312-linux_x86_64.whl"
)

# Calls that only exist on cumotion >= 1.1.0 and that the bimanual controller requires.
_REQUIRED_RMPFLOW_METHODS = ("add_target_frame", "set_pose_target", "clear_pose_target")


def _repository_root() -> Path:
    """Return the IsaacLab-Arena checkout that contains this file."""
    return Path(__file__).resolve().parents[4]


def runtime_dir() -> Path:
    """Resolve the directory holding the ``cumotion`` package.

    Returns:
        Directory to place on ``sys.path``. Honours ``$CUMOTION_RUNTIME_DIR`` when set, otherwise
        falls back to the vendored copy at the repository root.

    Raises:
        FileNotFoundError: If neither location contains a ``cumotion`` package.
    """
    override = os.environ.get(_RUNTIME_DIR_ENV_VAR)
    candidates = [Path(override).expanduser()] if override else []
    candidates.append(_repository_root() / _VENDORED_DIR_NAME)

    for candidate in candidates:
        if (candidate / "cumotion").is_dir():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "cuMotion 1.1.0 runtime not found. Searched:\n"
        f"{searched}\n\n"
        "Unpack the wheel that ships with the G1 deployment stack:\n"
        f"  mkdir -p {_repository_root() / _VENDORED_DIR_NAME} && \\\n"
        f"    unzip -q -o {_WHEEL_HINT} \\\n"
        f"    -d {_repository_root() / _VENDORED_DIR_NAME}\n\n"
        f"Or point {_RUNTIME_DIR_ENV_VAR} at an existing install."
    )


@lru_cache(maxsize=1)
def import_cumotion() -> ModuleType:
    """Import and return the ``cumotion._cumotion`` extension module.

    The resolved runtime is prepended to ``sys.path`` so it wins over any copy Isaac Sim has
    already contributed. Cached, so repeated controller construction costs one lookup.

    Returns:
        The ``cumotion._cumotion`` module, exposing ``load_robot_from_file``, ``create_world``,
        ``create_rmpflow_config_from_file``, ``create_rmpflow`` and the ``Pose3`` / ``Rotation3``
        value types.

    Raises:
        FileNotFoundError: If no cuMotion runtime could be located.
        RuntimeError: If the runtime that loaded predates the multi-target-frame API.
    """
    directory = str(runtime_dir())
    if directory not in sys.path:
        sys.path.insert(0, directory)

    from cumotion import _cumotion

    missing = [name for name in _REQUIRED_RMPFLOW_METHODS if not hasattr(_cumotion.RmpFlow, name)]
    if missing:
        raise RuntimeError(
            f"cuMotion at '{getattr(_cumotion, '__file__', '<unknown>')}' is missing "
            f"{', '.join(missing)}, so it cannot drive a bimanual RMPFlow policy. This is the "
            "signature of Isaac Sim's prebundled cumotion 1.1.0a2 shadowing the 1.1.0 runtime — "
            "import this module before Isaac Sim's cumotion extension loads, or set "
            f"{_RUNTIME_DIR_ENV_VAR} to a 1.1.0 install."
        )
    return _cumotion
