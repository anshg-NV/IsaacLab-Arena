# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Directory layout for local experiments under ``<repo root>/experiments``.

A single experiment holds its datasets, training configs and every training/evaluation
run it produced::

    experiments/<name>/
      dataset/                      teleop.hdf5, annotated.hdf5, generated.hdf5, ...
      train/
        train.bash  eval.bash
        configs/                    dp_0.json, dp_1.json, ...
      eval/
        <run>/                      logs/  models/  config.json  last.pth
          eval/<eval name>/         per-episode results, report, videos
        results.jsonl               one record appended per completed evaluation

Runs are named by the caller after the training config they came from (``0``, ``1``, ...),
and each holds the evaluations of its own checkpoints under ``eval/``.

This module is imported by entry points outside the ``isaaclab_arena`` package (the
robomimic ``train.py`` in the Isaac Lab submodule), so it must stay free of Isaac Sim
imports and must not depend on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "experiments"
"""Root directory holding every local experiment."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Paths making up a single experiment.

    Args:
        name: Experiment name, i.e. the directory name under the experiments root.
    """

    name: str

    @property
    def root(self) -> Path:
        return EXPERIMENTS_ROOT / self.name

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def train_dir(self) -> Path:
        return self.root / "train"

    @property
    def configs_dir(self) -> Path:
        return self.train_dir / "configs"

    @property
    def eval_root(self) -> Path:
        return self.root / "eval"

    @property
    def results_jsonl(self) -> Path:
        """File that one record is appended to per completed evaluation."""
        return self.eval_root / "results.jsonl"

    def run_dir(self, run: str) -> Path:
        """Training outputs for ``run``: ``logs/``, ``models/``, ``config.json``, ``last.pth``."""
        return self.eval_root / run

    def evals_dir(self, run: str) -> Path:
        """Directory holding every evaluation of ``run``."""
        return self.run_dir(run) / "eval"

    def leapp_dir(self, run: str) -> Path:
        """Scratch outputs of ``run``'s LEAPP export runs, kept out of the evaluation record."""
        return self.run_dir(run) / "leapp"

    def eval_dir(self, run: str, eval_name: str) -> Path:
        """Outputs of one evaluation of ``run``: per-episode results, report and videos."""
        return self.evals_dir(run) / eval_name

    def next_eval_name(self, run: str) -> str:
        """First unused ``eval_<n>`` name for ``run``, so repeated evaluations do not overwrite."""
        evals_dir = self.evals_dir(run)
        index = 0
        while (evals_dir / f"eval_{index}").exists():
            index += 1
        return f"eval_{index}"
