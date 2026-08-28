# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scaffold a new local experiment under ``experiments/<name>``.

Creates the directory tree and writes a ``train.bash`` / ``eval.bash`` pair wired to the
new layout, so the only thing left to do is drop datasets into ``dataset/`` and training
configs into ``train/configs/``.

Runs on the host, outside the container (no Isaac Sim imports):

    python isaaclab_arena/scripts/experiments/new_experiment.py --experiment my_exp \\
        --num_runs 4 --from_config path/to/dp_0.json
"""

from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path

from isaaclab_arena.utils.experiment_paths import ExperimentPaths

TRAIN_BASH_TEMPLATE = """#!/usr/bin/env bash
# Training sweep for the "{experiment}" experiment.
#
#   conda activate isaaclab_robomimic
#   bash experiments/{experiment}/train/train.bash
#
# Each run writes to experiments/{experiment}/eval/<run>.
set -euo pipefail

EXPERIMENT={experiment}
ARENA_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.." && pwd)"
EXPERIMENT_DIR="${{ARENA_ROOT}}/experiments/${{EXPERIMENT}}"
DATASET="${{EXPERIMENT_DIR}}/dataset/generated_preproc.hdf5"

cd "${{ARENA_ROOT}}/submodules/IsaacLab"

# The training config is taken from train/configs/dp_<run>.json.
for i in {{0..{last_run}}}; do
    ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \\
        --dataset "${{DATASET}}" \\
        --experiment "${{EXPERIMENT}}" \\
        --run "${{i}}"
    echo "--------------------------------"
    echo "Completed training for dp_${{i}}"
    echo "--------------------------------"
done
"""

EVAL_BASH_TEMPLATE = """#!/usr/bin/env bash
# Evaluation sweep for the "{experiment}" experiment.
#
#   conda activate isaaclab_robomimic_3_0
#   bash experiments/{experiment}/train/eval.bash
#
# Each evaluation writes to experiments/{experiment}/eval/<run>/eval/eval_<n> -- the name is
# allocated automatically, so evaluating several checkpoints of one run does not overwrite --
# and appends one record to experiments/{experiment}/eval/results.jsonl.
set -euo pipefail

EXPERIMENT={experiment}
ENVIRONMENT={environment}
CHECKPOINT_NAME={checkpoint_name}
ARENA_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.." && pwd)"
EXPERIMENT_DIR="${{ARENA_ROOT}}/experiments/${{EXPERIMENT}}"

cd "${{ARENA_ROOT}}"

for i in {{0..{last_run}}}; do
    python isaaclab_arena/evaluation/policy_runner.py \\
        --viz kit \\
        --policy_type robomimic_stand \\
        --checkpoint "models/${{CHECKPOINT_NAME}}" \\
        --num_episodes 50 \\
        --num_envs 5 \\
        --enable_cameras \\
        --experiment "${{EXPERIMENT}}" \\
        --run "${{i}}" \\
        "${{ENVIRONMENT}}"
    echo "--------------------------------"
    echo "Completed evaluation for dp_${{i}}"
    echo "--------------------------------"
done
"""


def write_bash(path: Path, contents: str) -> None:
    """Write an executable bash script."""
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to create.")
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs the generated train/eval sweeps loop over."
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="hubble_g1_static_pick_and_place",
        help="Arena environment the generated eval sweep runs against.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="model_epoch_600.pth",
        help="Checkpoint file name under a run's models directory that the generated eval sweep loads.",
    )
    parser.add_argument(
        "--from_config", type=str, default=None, help="Training config JSON to seed train/configs/dp_0.json with."
    )
    args = parser.parse_args()

    paths = ExperimentPaths(args.experiment)
    assert not paths.root.exists(), f"Experiment {args.experiment} already exists at {paths.root}"
    assert args.num_runs > 0, "--num_runs must be positive"

    for directory in (paths.dataset_dir, paths.configs_dir, paths.eval_root):
        directory.mkdir(parents=True)

    template_args = {
        "experiment": args.experiment,
        "environment": args.environment,
        "checkpoint_name": args.checkpoint_name,
        "last_run": args.num_runs - 1,
    }
    write_bash(paths.train_dir / "train.bash", TRAIN_BASH_TEMPLATE.format(**template_args))
    write_bash(paths.train_dir / "eval.bash", EVAL_BASH_TEMPLATE.format(**template_args))

    if args.from_config is not None:
        shutil.copyfile(args.from_config, paths.configs_dir / "dp_0.json")

    print(f"Created experiment {args.experiment} at {paths.root}")
    print(f"  datasets  -> {paths.dataset_dir}")
    print(f"  configs   -> {paths.configs_dir}")
    print(f"  train     -> {paths.train_dir / 'train.bash'}")
    print(f"  eval      -> {paths.train_dir / 'eval.bash'}")
    print(f"  results   -> {paths.results_jsonl}")


if __name__ == "__main__":
    main()
