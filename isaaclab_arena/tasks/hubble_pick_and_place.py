# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import object_on_destination
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class HubblePickAndPlaceTask(TaskBase):
    """Hubble Pick-and-place task. Success fires when the pick-up object contacts the destination
    with low velocity. Failure (object_dropped) fires when the object falls below the
    background's ``object_min_z``.

    The default Mimic cfg is ``HubblePickAndPlaceMimicEnvCfg``. Pass
    ``mimic_env_cfg_factory`` to inject a custom ``MimicEnvCfg`` instead:

        HubblePickAndPlaceTask(..., mimic_env_cfg_factory=lambda arm_mode: MyCustomMimicEnvCfg(arm_mode=arm_mode, ...))

    The factory receives ``arm_mode`` from the env builder and returns a constructed cfg.
    """

    def __init__(
        self,
        pick_up_object: Asset,
        destination_location: Asset,
        background_scene: Asset,
        destination_object: Asset | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        force_threshold: float = 1.0,
        velocity_threshold: float = 0.1,
        mimic_env_cfg_factory: Callable[[ArmMode], MimicEnvCfg] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.pick_up_object = pick_up_object
        self.destination_object = destination_object
        self.background_scene = background_scene
        self.destination_location = destination_location
        self.scene_config = SceneCfg(
            pick_up_object_contact_sensor=self.pick_up_object.get_contact_sensor_cfg(
                contact_against_object=self.destination_location,
            ),
        )
        self.force_threshold = force_threshold
        self.velocity_threshold = velocity_threshold
        self.mimic_env_cfg_factory = mimic_env_cfg_factory
        self.events_cfg = None
        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Pick up the {pick_up_object.name}, and place it into the {destination_location.name}"
            if task_description is None
            else task_description
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        success = TerminationTermCfg(
            func=object_on_destination,
            params={
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "contact_sensor_cfg": SceneEntityCfg("pick_up_object_contact_sensor"),
                "force_threshold": self.force_threshold,
                "velocity_threshold": self.velocity_threshold,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Build the Mimic env cfg for this task.

        If ``mimic_env_cfg_factory`` was passed at construction, invoke it with
        ``arm_mode`` and return its result. Otherwise build the default
        ``HubblePickAndPlaceMimicEnvCfg``.
        """
        if self.mimic_env_cfg_factory is not None:
            return self.mimic_env_cfg_factory(arm_mode)
        return HubblePickAndPlaceMimicEnvCfg(
            arm_mode=arm_mode,
            pick_up_object_name=self.pick_up_object.name,
            destination_location_name=self.destination_location.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.pick_up_object)]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class SceneCfg:
    """Scene configuration for the pick and place task."""

    pick_up_object_contact_sensor: ContactSensorCfg = MISSING


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING



@configclass
class HubblePickAndPlaceMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Hubble pick-and-place tasks.

    Left arm performs the full pick-and-place sequence (2 subtasks: grasp + place),
    matching PickPlaceMimicEnvCfg. Right arm is passive with a single final subtask.
    """

    pick_up_object_name: str = MISSING
    destination_location_name: str = MISSING
    arm_mode: ArmMode = ArmMode.DUAL_ARM

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        if self.arm_mode != ArmMode.DUAL_ARM:
            raise ValueError(f"HubblePickAndPlaceMimicEnvCfg only supports ArmMode.DUAL_ARM; got {self.arm_mode}")

        self.datagen_config.name = (
            f"locomanip_pick_and_place_{self.pick_up_object_name}_to_{self.destination_location_name}_D0"
        )
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1
        self.datagen_config.use_navigation_controller = False

        # Right arm subtasks
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref=self.destination_location_name,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = subtask_configs

        # Left arm subtasks
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=self.destination_location_name,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = subtask_configs

    # Body subtasks
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref=self.destination_location_name,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["body"] = subtask_configs
