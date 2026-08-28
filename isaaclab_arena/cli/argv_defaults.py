# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Default command-line arguments for the local run scripts, applied to ``sys.argv``.

These scripts parse repeatedly (once to launch the simulation app, again once the environment
subcommand and policy flags exist), and the simulation app reads ``sys.argv`` directly at launch.
Filling defaults into ``sys.argv`` therefore makes them behave exactly as if they had been typed,
which parser-level defaults do not: a later parse resets those.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

DEFAULT_ENVIRONMENT = "hubble_g1_static_pick_and_place"
"""Environment the local run scripts fall back to when no environment is selected."""


def apply_argv_defaults(
    valued_args: Sequence[tuple[str, str]] = (),
    flags: Sequence[str] = (),
    *,
    label: str,
) -> None:
    """Fill default arguments into ``sys.argv``, leaving anything already given alone.

    Args:
        valued_args: ``(flag, value)`` pairs to append when the flag is absent.
        flags: Boolean flags to append when absent.
        label: Script name used to attribute the printed notice.
    """
    applied = []
    for flag, value in valued_args:
        if not any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:]):
            sys.argv += [flag, value]
            applied.append(f"{flag} {value}")
    for flag in flags:
        if flag not in sys.argv[1:]:
            sys.argv.append(flag)
            applied.append(flag)

    if applied:
        print(f"[{label}] Applying default arguments: {' '.join(applied)}")


def apply_default_environment(
    args_cli: argparse.Namespace,
    default_environment: str = DEFAULT_ENVIRONMENT,
    *,
    label: str,
) -> bool:
    """Append the default environment subcommand to ``sys.argv`` when no environment was selected.

    The environment is a subparser, so the default can only be applied once parsing shows that
    neither an environment subcommand nor ``--env_graph_spec_yaml`` was given. Callers re-parse when
    this returns True, so the subparser contributes its own arguments.

    Args:
        args_cli: Namespace from a parse that included the environment subparsers.
        default_environment: Environment subcommand to fall back to.
        label: Script name used to attribute the printed notice.
    """
    environment_given = getattr(args_cli, "example_environment", None) is not None
    graph_spec_given = getattr(args_cli, "env_graph_spec_yaml", None) is not None
    if environment_given or graph_spec_given:
        return False

    print(f"[{label}] Applying default environment: {default_environment}")
    sys.argv.append(default_environment)
    return True
