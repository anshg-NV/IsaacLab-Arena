# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for the former Isaac RTX first-frame render patch.

Isaac Lab's stock ``ensure_isaac_rtx_render_update`` no longer exposes the
``_ensure_streaming_subscription`` attribute that the old patch relied on.
This function is kept as a no-op for callers such as ``record_demos.py``.
"""


def patch_isaac_rtx_renderer() -> None:
    """No-op: the former RTX first-frame patch is no longer compatible with the current IsaacLab."""
