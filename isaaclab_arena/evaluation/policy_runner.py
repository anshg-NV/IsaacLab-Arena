# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime
import os
import json
import tempfile

import torch
import tqdm
from importlib import import_module
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.cli.argv_defaults import apply_argv_defaults, apply_default_environment
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import (
    add_policy_cli_args,
    add_policy_runner_arguments,
    apply_experiment_checkpoint,
    build_policy_from_cli,
)
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.experiment_paths import ExperimentPaths
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir, wrap_env_for_video
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase


DEFAULT_VALUED_ARGS = (("--viz", "kit"), ("--policy_type", "robomimic_stand"))
"""Valued flags this script fills in when they are absent from the command line."""

DEFAULT_FLAGS = ("--enable_cameras",)
"""Boolean flags this script sets when they are absent from the command line."""


def get_policy_cls(policy_type: str) -> type[PolicyBase]:
    """Get the policy class for the given policy type name.

    Note that this function:
    - first: checks for a registered policy type in the PolicyRegistry
    - if not found, it tries to dynamically import the policy class, treating
      the policy_type argument as a string representing the module path and class name.

    """
    policy_registry = PolicyRegistry()
    if policy_registry.is_registered(policy_type):
        return policy_registry.get_policy(policy_type)
    else:
        print(f"Policy {policy_type} is not registered. Dynamically importing from path: {policy_type}")
        assert "." in policy_type, (
            "policy_type must be a dotted Python import path of the form 'module.submodule.ClassName', got:"
            f" {policy_type}"
        )
        # Dynamically import the class from the string path
        module_path, class_name = policy_type.rsplit(".", 1)
        module = import_module(module_path)
        policy_cls = getattr(module, class_name)
        return policy_cls


def is_distributed(args_cli: argparse.Namespace) -> bool:
    return (
        "cuda" in args_cli.device and hasattr(args_cli, "distributed") and args_cli.distributed and get_world_size() > 1
    )


def _to_jsonable(value: Any) -> Any:
    """Convert common metric and CLI values to JSON-serializable types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def is_anonymous_run(args_cli: argparse.Namespace) -> bool:
    """Whether this run only exists to produce a LEAPP export, and should leave no evaluation record."""
    return bool(getattr(args_cli, "leapp_export", False))


def resolve_output_dir(args_cli: argparse.Namespace) -> str:
    """Directory this evaluation writes its outputs to.

    With ``--experiment``, that is ``eval/<run>/eval/<eval name>`` inside the experiment, where the
    name defaults to the next unused ``eval_<n>``; otherwise a reverse-dated subdirectory of
    ``--output_base_dir``.

    An anonymous run writes to ``eval/<run>/leapp`` instead, so any recorded video sits with the run
    without landing in its evaluation record. Nothing creates the directory unless a recorder writes.
    """
    if is_anonymous_run(args_cli):
        if args_cli.experiment is not None and args_cli.run is not None:
            return str(ExperimentPaths(args_cli.experiment).leapp_dir(args_cli.run))
        return os.path.join(tempfile.gettempdir(), "policy_runner_leapp_export")
    if args_cli.experiment is None:
        return timestamped_run_dir(args_cli.output_base_dir)
    assert args_cli.run is not None, "--run is required when --experiment is given"
    paths = ExperimentPaths(args_cli.experiment)
    eval_name = args_cli.eval_name or paths.next_eval_name(args_cli.run)
    return str(paths.eval_dir(args_cli.run, eval_name))


def emit_final_metrics(
    metrics: dict[str, Any],
    args_cli: argparse.Namespace,
    local_rank: int,
    world_size: int,
    output_dir: str,
) -> None:
    """Emit final metrics as parseable JSON and, with ``--experiment``, append them to the experiment's JSONL."""
    record = _to_jsonable(
        {
            "experiment": args_cli.experiment,
            "run": args_cli.run,
            "eval_name": os.path.basename(output_dir) if args_cli.experiment is not None else None,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "rank": local_rank,
            "world_size": world_size,
            "checkpoint_path": getattr(args_cli, "robomimic_checkpoint", None),
            "num_episodes": args_cli.num_episodes,
            "output_dir": output_dir,
            "metrics": metrics,
        }
    )
    metrics_json = json.dumps(record, sort_keys=True)
    print(f"POLICY_RUNNER_FINAL_METRICS_JSON: {metrics_json}", flush=True)

    if args_cli.experiment is None or is_anonymous_run(args_cli):
        return

    results_path = ExperimentPaths(args_cli.experiment).results_jsonl
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(metrics_json + "\n")


def rollout_policy(
    env,
    policy: PolicyBase,
    num_steps: int | None,
    num_episodes: int | None,
) -> MetricsDataCollection | None:
    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    pbar = None
    try:
        obs, _ = env.reset()
        policy.reset()
        policy.set_task_description(env.unwrapped.get_language_instruction())

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        num_episodes_completed = 0
        num_steps_completed = 0

        while True:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    # Only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                    # Break if number of episodes is reached
                    completed_episodes = env_ids.shape[0]
                    num_episodes_completed += completed_episodes
                    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                        metrics = env.unwrapped.compute_metrics()
                        tqdm.tqdm.write(
                            f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                            f" {metrics_to_plain_python_types(metrics)}"
                        )
                    if num_episodes is not None:
                        pbar.update(completed_episodes)
                        if num_episodes_completed >= num_episodes:
                            break
                # Break if number of steps is reached
                num_steps_completed += 1
                if num_steps is not None:
                    pbar.update(1)
                    if num_steps_completed >= num_steps:
                        break

        pbar.close()

    except Exception as e:
        if pbar is not None:
            pbar.close()
        raise RuntimeError(f"Error rolling out policy: {e}")

    else:

        # Only compute metrics if env has non-None metrics.
        # Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing)
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None


def main():
    """Run an IsaacLab Arena environment with a policy.
    Use --distributed with torchrun command for one process per GPU on multi-GPU machines. AppLauncher uses LOCAL_RANK for device.
    """
    apply_argv_defaults(DEFAULT_VALUED_ARGS, DEFAULT_FLAGS, label="policy_runner")

    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    local_rank = get_local_rank()
    world_size = get_world_size()
    # Setting device to local rank before SimulationAppContext
    if is_distributed(args_cli):
        args_cli.device = f"cuda:{local_rank}"
        print(f"[Rank {local_rank}/{world_size}] One Isaac Lab instance per process on cuda:{local_rank}")

    # --record_camera_video requires cameras to be enabled at sim startup, before SimulationAppContext.
    if "--record_camera_video" in unknown:
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli):

        # Get the policy-type flag before proceeding to other arguments
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()

        # Get the policy class from the policy type
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(
            f"[Rank {local_rank}/{world_size}] Requested policy type: {args_cli.policy_type} -> Policy class:"
            f" {policy_cls}"
        )

        # Add the example environment arguments and config-derived policy arguments.
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = add_policy_cli_args(args_parser, policy_cls)
        apply_experiment_checkpoint(args_parser, args_cli)
        args_cli, hydra_overrides = args_parser.parse_known_args()

        if apply_default_environment(args_cli, label="policy_runner"):
            args_cli, hydra_overrides = args_parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser)
        # Re-apply per-rank device after parse preventing device got overwritten by the default value
        if is_distributed(args_cli):
            args_cli.distributed = True
            args_cli.device = f"cuda:{local_rank}"
            # Per-rank seed when distributed so each process has a different seed
            if args_cli.seed is not None:
                args_cli.seed += local_rank

        # Re-apply enable_cameras: the full parse resets it to default False.
        if args_cli.record_camera_video:
            args_cli.enable_cameras = True

        # Build scene. Use rgb_array render mode when recording so RecordVideo can grab frames.
        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)

        if args_cli.list_variations:
            print(arena_builder.get_variations_catalogue_as_string())
            return

        # A LEAPP export run leaves no evaluation record: no per-episode results, metrics or report.
        # Requested videos are still recorded, so the exported policy can be watched running.
        anonymous_run = is_anonymous_run(args_cli)

        output_dir = resolve_output_dir(args_cli)
        if anonymous_run and (args_cli.record_viewport_video or args_cli.record_camera_video):
            print(f"[LEAPP export] Recording video to {output_dir}")
        video_cfg = VideoRecordingCfg(
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            video_base_dir=output_dir,
        )
        env = arena_builder.make_registered(render_mode=video_cfg.render_mode)

        # Write per-episode results to disk. Without an output path the recorder keeps them in memory only.
        if not anonymous_run:
            results_path = os.path.join(output_dir, f"episode_results_rank{local_rank}.jsonl")
            env.unwrapped.episode_recorder.set_job_name("policy_runner")
            env.unwrapped.episode_recorder.set_output_path(results_path)

        # Create the policy through the typed config compatibility adapter.
        policy = build_policy_from_cli(policy_cls, args_cli)

        # Simulation length.
        if policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_episodes} episodes")
            else:
                raise ValueError(f"[Rank {local_rank}/{world_size}] Either num_steps or num_episodes must be provided")

        # Optionally wrap with the viewport/camera video recorders (both independent).
        env = wrap_env_for_video(env, video_cfg, num_steps, num_episodes)

        steps_str = f"{num_steps} steps" if num_steps is not None else f"{num_episodes} episodes"
        print(f"[Rank {local_rank}/{world_size}] Starting rollout ({steps_str})")
        metrics = rollout_policy(env, policy, num_steps, num_episodes)

        if metrics is not None:
            metrics_plain = metrics_to_plain_python_types(metrics)
            print(f"[Rank {local_rank}/{world_size}] Metrics: {metrics_plain}")
            if not anonymous_run:
                emit_final_metrics(metrics_plain, args_cli, local_rank, world_size, output_dir)

        # NOTE(huikang, 2025-12-30)Explicitly clean up the remote policy client / server.
        # Do NOT rely on a __del__ destructor in policy for this, since destructors are
        # triggered implicitly and their execution time (or even whether they run)
        # is not guaranteed, which makes resource cleanup unreliable.
        if policy.is_remote:
            policy.shutdown_remote(kill_server=args_cli.remote_kill_on_exit)

        # Close the environment.
        env.close()

        # Write and serve the evaluation report.
        # Only the local rank 0 writes/serves it, to avoid races on a shared output dir.
        if get_local_rank() == 0 and not anonymous_run:
            report_path = build_report(output_dir)
            if args_cli.serve_evaluation_report:
                serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)


if __name__ == "__main__":
    main()
