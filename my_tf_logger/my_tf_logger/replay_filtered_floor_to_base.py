#!/usr/bin/env python3
"""
Replay a recorded floor -> base trajectory from the filtered CSV and move the
existing Unitree Go2 URDF model in RViz.

Expected live TF subtree supplied by the robot/URDF nodes:

    map -> base_link -> robot links

This replay node reads recorded floor_T_base poses and publishes the bridge:

    floor -> map

For every recorded pose:

    floor_T_map = floor_T_base_recorded @ inverse(map_T_base_live)

Therefore RViz reconstructs:

    floor_T_map @ map_T_base_live = floor_T_base_recorded

IMPORTANT:
Do not run another node that also publishes floor -> map while this replay is
active. Stop the AprilTag TF logger (or disable its floor -> map broadcaster)
before starting this replay node.
"""

import csv
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import JointState
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class FilteredFloorToRobotReplay(Node):
    """Replay recorded floor-relative base poses through floor -> map."""

    FLOOR_TO_BASE_COLUMNS = [
        f"T{row}{column}"
        for row in range(4)
        for column in range(4)
    ]

    CAMERA_TO_TAG_COLUMNS = [
        f"C_T1_T{row}{column}"
        for row in range(4)
        for column in range(4)
    ]

    FLOOR_TO_LEG_COLUMNS = [
        f"F_T{row}{column}"
        for row in range(4)
        for column in range(4)
    ]

    LEG_FRAME_NAMES = [
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FL_calflower",
        "FL_calflower1",
        "FL_foot",

        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FR_calflower",
        "FR_calflower1",
        "FR_foot",

        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RL_calflower",
        "RL_calflower1",
        "RL_foot",

        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RR_calflower",
        "RR_calflower1",
        "RR_foot",
    ]

    REPLAY_JOINT_NAMES = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",

        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",

        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",

        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]


    # ============================================================
    # URDF RECONSTRUCTION VALIDATION
    # ============================================================
    VALIDATION_FRAME_NAMES = [
        "FL_hip",
        "FL_thigh",
        "FL_calf",

        "FR_hip",
        "FR_thigh",
        "FR_calf",

        "RL_hip",
        "RL_thigh",
        "RL_calf",

        "RR_hip",
        "RR_thigh",
        "RR_calf",
    ]

    # ------------------------------------------------------------
    # URDF joint-origin translations.
    #
    # Because the hip/thigh/calf joint origins have rpy="0 0 0",
    # the parent->child joint transform is:
    #
    #     T_parent_child(q)
    #       =
    #     Translation(xyz) @ Rotation_about_joint_axis(q)
    # ------------------------------------------------------------

    URDF_HIP_ORIGIN_XYZ = {
        "FL": ( 0.1934,  0.0465, 0.0),
        "FR": ( 0.1934, -0.0465, 0.0),
        "RL": (-0.1934,  0.0465, 0.0),
        "RR": (-0.1934, -0.0465, 0.0),
    }

    URDF_THIGH_ORIGIN_XYZ = {
        "FL": (0.0,  0.0955, 0.0),
        "FR": (0.0, -0.0955, 0.0),
        "RL": (0.0,  0.0955, 0.0),
        "RR": (0.0, -0.0955, 0.0),
    }

    URDF_CALF_ORIGIN_XYZ = (
        0.0,
        0.0,
        -0.213,
    )


    P_star_Rc_to_T1 = np.array ([
    [0.997492, 0.059323, -0.038611, 0.002513],
    [-0.034889, -0.062547, -0.997432, -0.041843],
    [-0.061586, 0.996277, -0.060320, -0.112739],
    [0.000000, 0.000000, 0.000000, 1.000000],
    ],
    dtype=float,
    )

    def __init__(self) -> None:
        super().__init__("filtered_floor_to_robot_replay")

        self.declare_parameter(
            "csv_file",
            os.path.expanduser(
                "~/Go2_Skill_Base_Data_Sensor"
                "/HandStand/Trial_39/Tf_stationary_HandStand_Trial_39.csv"
            ),
        )

        self.declare_parameter(
            "leg_csv_file",
            os.path.expanduser(
                "~/Go2_Skill_Base_Data_Sensor"
                "/HandStand/Trial_39/Tf_leg_stationary_HandStand_Trial_39.csv"
            ),
        )
        self.declare_parameter("floor_frame", "floor")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("playback_speed", 1.0)
        self.declare_parameter("timer_hz", 100.0)
        self.declare_parameter("loop_playback", False)
        self.declare_parameter("tf_lookup_timeout_s", 0.2)

        ##---adding new parameters--------------------#####
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("tag_frame", "object_1")

        self.csv_file = os.path.expanduser(
            self.get_parameter("csv_file")
            .get_parameter_value()
            .string_value
        )

        # --------------------------------------------------
        # SYNCHRONIZED LEG CSV
        # --------------------------------------------------
        self.leg_csv_file = os.path.expanduser(
            self.get_parameter("leg_csv_file")
            .get_parameter_value()
            .string_value
        )

        self.floor_frame = (
            self.get_parameter("floor_frame")
            .get_parameter_value()
            .string_value
        )
        self.map_frame = (
            self.get_parameter("map_frame")
            .get_parameter_value()
            .string_value
        )
        self.base_frame = (
            self.get_parameter("base_frame")
            .get_parameter_value()
            .string_value
        )
        self.playback_speed = (
            self.get_parameter("playback_speed")
            .get_parameter_value()
            .double_value
        )
        self.timer_hz = (
            self.get_parameter("timer_hz")
            .get_parameter_value()
            .double_value
        )
        self.loop_playback = (
            self.get_parameter("loop_playback")
            .get_parameter_value()
            .bool_value
        )
        self.tf_lookup_timeout_s = (
            self.get_parameter("tf_lookup_timeout_s")
            .get_parameter_value()
            .double_value
        )

        ##----Reading Camera and Tag frame names--------------------#####
        self.camera_frame = (
            self.get_parameter("camera_frame")
            .get_parameter_value()
            .string_value
        )

        self.tag_frame = (
            self.get_parameter("tag_frame")
            .get_parameter_value()
            .string_value
        )

        if self.playback_speed <= 0.0:
            raise ValueError("playback_speed must be greater than zero.")
        if self.timer_hz <= 0.0:
            raise ValueError("timer_hz must be greater than zero.")
        if self.tf_lookup_timeout_s <= 0.0:
            raise ValueError("tf_lookup_timeout_s must be greater than zero.")
        if len({self.floor_frame, self.map_frame, self.base_frame}) != 3:
            raise ValueError("floor, map, and base frame names must differ.")

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=10.0)
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ============================================================
        # REPLAY JOINT-STATE PUBLISHER
        # ============================================================
        self.joint_state_pub = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )
        self.marker_pub = self.create_publisher(
            Marker,
            "floor_to_robot_replay_markers",
            10,
        )

        self.samples = self.load_filtered_csv(self.csv_file)
        self.leg_samples = (self.load_leg_stationary_csv(self.leg_csv_file))
        self.sample_index = 0
        self.playback_start_monotonic_ns = time.monotonic_ns()
        self.playback_finished = False
        self.last_floor_to_base: Optional[np.ndarray] = None
        self.last_camera_to_tag: Optional[np.ndarray] = None
        self.path_marker = self.create_path_marker()
        self.path_completed_once = False
        self.last_tf_warning_ns = 0
        self.last_base_common_time_ns = None

        # ============================================================
        # URDF RECONSTRUCTION VALIDATION STORAGE
        # ============================================================
        self.validation_errors = {
            frame_name: {
                "position_error_mm": [],
                "rotation_error_deg": [],
            }
            for frame_name in self.VALIDATION_FRAME_NAMES
        }

        self.validation_sample_count = 0

        # Validation is accumulated only during the first replay pass.
        # This prevents loop_playback=True from adding the same samples
        # repeatedly forever.
        self.validation_first_pass_complete = False

        base_common_times = [
            int(
                Decimal(
                    sample[3][
                        "common_time_ns"
                    ].strip()
                )
            )
            for sample in self.samples
        ]

        leg_common_times = set(
            self.leg_samples.keys()
        )

        missing_leg_times = (
            set(base_common_times)
            - leg_common_times
        )

        extra_leg_times = (
            leg_common_times
            - set(base_common_times)
        )

        if missing_leg_times:
            raise RuntimeError(
                "[LEG REPLAY] Leg CSV is missing "
                "base timestamps: "
                f"{sorted(missing_leg_times)}"
            )

        if extra_leg_times:
            self.get_logger().warn(
                "[LEG REPLAY] Leg CSV contains "
                "additional timestamps not used by "
                "the base replay: "
                f"{sorted(extra_leg_times)}"
            )

        self.get_logger().info(
            "[LEG REPLAY] Loaded "
            f"{len(self.leg_samples)} synchronized "
            "leg-pose groups."
        )

        self.get_logger().info(
            "[LEG REPLAY] Each base sample "
            f"contains {len(self.LEG_FRAME_NAMES)} "
            "leg frames."
        )

        self.timer = self.create_timer(
            1.0 / self.timer_hz,
            self.tick,
        )

        recorded_duration = (
            self.samples[-1][0] - self.samples[0][0]
        ) / 1_000_000_000.0

        self.get_logger().info(
            f"Loaded {len(self.samples)} filtered floor -> base poses."
        )
        self.get_logger().info(
            f"Recorded trajectory duration: {recorded_duration:.9f} s."
        )
        self.get_logger().info(
            f"Publishing replay bridge {self.floor_frame} -> "
            f"{self.map_frame}; existing {self.map_frame} -> "
            f"{self.base_frame} will carry the URDF robot."
        )
        self.get_logger().warn(
            "Make sure the AprilTag TF logger is not simultaneously "
            f"publishing {self.floor_frame} -> {self.map_frame}."
        )


        self.P_base_to_Head_Upper = np.eye(4, dtype=float)
        self.P_base_to_Head_Upper[:3, 3] = np.array(
            [0.285, 0.0, 0.01],
            dtype=float,
        )

        self.P_Head_Upper_to_Rc = np.eye(4, dtype=float)

        self.P_Head_Upper_to_Rc[:3, :3] = R.from_quat(
            [-0.5, 0.500002, -0.5, 0.499998]
        ).as_matrix()

        self.P_Head_Upper_to_Rc[:3, 3] = np.array(
            [0.045, 0.0, 0.03],
            dtype=float,
        )

        self.P_base_to_T1 = (
            self.P_base_to_Head_Upper
            @ self.P_Head_Upper_to_Rc
            @ self.P_star_Rc_to_T1
        )


    @staticmethod
    def transform_stamped_to_matrix(
        transform_stamped: TransformStamped,
    ) -> np.ndarray:
        translation = transform_stamped.transform.translation
        quaternion = transform_stamped.transform.rotation

        rotation = R.from_quat(
            [
                quaternion.x,
                quaternion.y,
                quaternion.z,
                quaternion.w,
            ]
        ).as_matrix()

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [
            translation.x,
            translation.y,
            translation.z,
        ]
        return matrix

    # ------------------------------------------------------------------
    # CSV LOADING
    # ------------------------------------------------------------------
    def load_filtered_csv(
            self,
            filepath: str,
        ) -> List[Tuple[
                int,
                np.ndarray,
                np.ndarray,
                Dict[str, str],
            ]
        ]:
        # Load a filtered floor->base trajectory CSV for replay.
        # Each valid row becomes:
        # (
        #     wall_time_ns,
        #     floor_to_base matrix,
        #     camera_to_tag matrix,
        #     original CSV row dictionary,
        # )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Filtered trajectory CSV does not exist: {filepath}"
            )

        samples: List[Tuple[int, np.ndarray, np.ndarray, Dict[str, str]]] = []
        skipped_rows = 0

        try:
            with open(
                filepath,
                "r",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                reader = csv.DictReader(csv_file)

                if reader.fieldnames is None:
                    # The CSV needs a header row so we can access named
                    # columns like wall_time_ns and the transform matrix values.
                    raise RuntimeError("CSV file has no header.")

                required_columns = {
                    "wall_time_ns",
                    "common_time_ns",
                    *self.FLOOR_TO_BASE_COLUMNS,
                    *self.CAMERA_TO_TAG_COLUMNS,
                }
                missing_columns = sorted(
                    required_columns - set(reader.fieldnames)
                )
                if missing_columns:
                    # Fail early if the CSV does not contain a full 4x4
                    # floor->base pose matrix.
                    raise RuntimeError(
                        "CSV is missing required columns: "
                        + ", ".join(missing_columns)
                    )

                for row_number, row in enumerate(reader, start=2):
                    try:
                        wall_time_ns = int(
                            Decimal(row["wall_time_ns"].strip())
                        )
                        floor_to_base = np.array(
                            [
                                float(row[column])
                                for column in self.FLOOR_TO_BASE_COLUMNS
                            ],
                            dtype=float,
                        ).reshape(4, 4)

                        camera_to_tag = np.array(
                            [
                                float(row[column])
                                for column in self.CAMERA_TO_TAG_COLUMNS
                            ],
                            dtype=float,
                        ).reshape(4, 4)

                        self.validate_transform_matrix(
                            floor_to_base,
                            row_number,
                        )

                        self.validate_transform_matrix(
                            camera_to_tag,
                            row_number,
                        )

                        
                    except (TypeError, ValueError, KeyError, InvalidOperation) as error:
                        skipped_rows += 1
                        self.get_logger().warn(
                            f"Skipping invalid CSV row {row_number}: {error}"
                        )
                        continue

                    samples.append((wall_time_ns,floor_to_base, camera_to_tag,row,))
        except OSError as error:
            raise RuntimeError(
                f"Could not read filtered trajectory CSV: {error}"
            ) from error

        if not samples:
            raise RuntimeError(
                "No valid floor -> base poses were found in the CSV."
            )

        # Sort the samples by recorded wall time so replay is chronological.
        samples.sort(key=lambda item: item[0])
        if skipped_rows:
            self.get_logger().warn(
                f"Skipped {skipped_rows} invalid CSV rows."
            )
        return samples

    def load_leg_stationary_csv(
            self,
            filepath: str,
        ) -> Dict[int, Dict[str, Tuple[np.ndarray, int, float]]]:
            """
            Load synchronized floor-relative leg poses.

            Returned structure:

                leg_samples[base_common_time_ns][frame_name]
                    =
                    (
                        floor_to_leg_matrix,
                        large_gap_warning,
                        interpolation_gap_ms,
                    )
            """

            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    f"Leg trajectory CSV does not exist: {filepath}"
                )

            leg_samples = {}

            required_columns = {
                "base_common_time_ns",
                "frame_name",
                "interpolation_gap_ms",
                "large_gap_warning",
                *self.FLOOR_TO_LEG_COLUMNS,
            }

            with open(
                filepath,
                "r",
                newline="",
                encoding="utf-8",
            ) as csv_file:

                reader = csv.DictReader(csv_file)

                if reader.fieldnames is None:
                    raise RuntimeError(
                        "Leg CSV has no header."
                    )

                missing_columns = (
                    required_columns
                    - set(reader.fieldnames)
                )

                if missing_columns:
                    raise RuntimeError(
                        "Leg CSV is missing required columns: "
                        + ", ".join(sorted(missing_columns))
                    )

                for row_number, row in enumerate(
                    reader,
                    start=2,
                ):
                    try:
                        base_time_ns = int(
                            Decimal(
                                row[
                                    "base_common_time_ns"
                                ].strip()
                            )
                        )

                        frame_name = (
                            row["frame_name"].strip()
                        )

                        if frame_name not in self.LEG_FRAME_NAMES:
                            self.get_logger().warn(
                                "[LEG REPLAY] Ignoring unexpected "
                                f"frame '{frame_name}' "
                                f"at CSV row {row_number}."
                            )
                            continue

                        floor_to_leg = np.array(
                            [
                                float(row[column])
                                for column
                                in self.FLOOR_TO_LEG_COLUMNS
                            ],
                            dtype=float,
                        ).reshape(4, 4)

                        self.validate_transform_matrix(
                            floor_to_leg,
                            row_number,
                        )

                        large_gap_warning = int(
                            row["large_gap_warning"]
                        )

                        interpolation_gap_ms = float(
                            row["interpolation_gap_ms"]
                        )

                    except (
                        TypeError,
                        ValueError,
                        KeyError,
                        InvalidOperation,
                    ) as error:
                        raise RuntimeError(
                            "[LEG REPLAY] Invalid leg CSV "
                            f"row {row_number}: {error}"
                        ) from error

                    if base_time_ns not in leg_samples:
                        leg_samples[
                            base_time_ns
                        ] = {}

                    if (
                        frame_name
                        in leg_samples[base_time_ns]
                    ):
                        raise RuntimeError(
                            "[LEG REPLAY] Duplicate frame "
                            f"{frame_name} for timestamp "
                            f"{base_time_ns}."
                        )

                    leg_samples[
                        base_time_ns
                    ][
                        frame_name
                    ] = (
                        floor_to_leg,
                        large_gap_warning,
                        interpolation_gap_ms,
                    )

            if not leg_samples:
                raise RuntimeError(
                    "No valid synchronized leg poses "
                    "were found."
                )

            # Every base timestamp should have all 24 frames.
            expected_frames = set(
                self.LEG_FRAME_NAMES
            )

            for (
                base_time_ns,
                frame_group,
            ) in leg_samples.items():

                actual_frames = set(
                    frame_group.keys()
                )

                missing_frames = (
                    expected_frames
                    - actual_frames
                )

                if missing_frames:
                    raise RuntimeError(
                        "[LEG REPLAY] Timestamp "
                        f"{base_time_ns} is missing frames: "
                        f"{sorted(missing_frames)}"
                    )

            return leg_samples



    # ------------------------------------------------------------------
    # TRANSFORM VALIDATION
    # ------------------------------------------------------------------
    @staticmethod
    def validate_transform_matrix(
        matrix: np.ndarray,
        row_number: int,
    ) -> None:
        # Ensure the CSV row contains a proper 4x4 homogeneous transform.
        if matrix.shape != (4, 4):
            raise ValueError(
                f"row {row_number}: matrix shape is not 4x4"
            )
        # Reject NaN or infinite entries.
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"row {row_number}: matrix contains non-finite values"
            )
        # The bottom row of a homogeneous transform must be [0, 0, 0, 1].
        if not np.allclose(
            matrix[3, :],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-5,
        ):
            raise ValueError(
                f"row {row_number}: invalid homogeneous bottom row"
            )

        rotation = matrix[:3, :3]
        # The top-left 3x3 block must be a valid rotation matrix.
        # Check orthonormality: R^T R should equal identity.
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=5e-3,
        ):
            raise ValueError(
                f"row {row_number}: rotation matrix is not orthonormal"
            )
        # Check determinant to ensure this is a proper rotation (no mirror).
        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=5e-3,
        ):
            raise ValueError(
                f"row {row_number}: rotation determinant is invalid"
            )

    def tick(self) -> None:
        if self.playback_finished:
            if (
                self.last_floor_to_base is not None
                and self.last_camera_to_tag is not None
            ):
                self.publish_bridge_for_recorded_pose(
                    self.last_floor_to_base
                )

                self.publish_camera_tag_branch(
                    self.last_floor_to_base,
                    self.last_camera_to_tag,
                )

            if (
                self.last_base_common_time_ns
                is not None
            ):
                self.publish_leg_frames_for_base_time(
                    self.last_base_common_time_ns,
                    report_quality=False,
                )


            if (
                self.last_floor_to_base is not None
                and self.last_base_common_time_ns
                is not None
            ):
                self.publish_replay_joint_state(
                    self.last_floor_to_base,
                    self.last_base_common_time_ns,
                )

            return

        elapsed_real_ns = (
            time.monotonic_ns()
            - self.playback_start_monotonic_ns
        )
        elapsed_recorded_ns = int(
            elapsed_real_ns * self.playback_speed
        )
        first_wall_time_ns = self.samples[0][0]

        while self.sample_index < len(self.samples):
            (
                sample_wall_time_ns,
                floor_to_base,
                camera_to_tag,
                row,
            ) = self.samples[self.sample_index]
            sample_relative_ns = (
                sample_wall_time_ns - first_wall_time_ns
            )

            if sample_relative_ns > elapsed_recorded_ns:
                break

            if not self.publish_sample(
                floor_to_base,
                camera_to_tag,
                row,
                self.sample_index,
            ):
                # Wait until map -> base_link becomes available.
                break

            self.last_floor_to_base = floor_to_base.copy()
            self.last_camera_to_tag = camera_to_tag.copy()
            self.sample_index += 1

            self.last_base_common_time_ns = int(
                Decimal(
                    row[
                        "common_time_ns"
                    ].strip()
                )
            )


        if self.sample_index >= len(self.samples):

            # ========================================================
            # PRINT VALIDATION ONCE AFTER THE FIRST COMPLETE PASS
            # ========================================================
            if not self.validation_first_pass_complete:

                self.print_urdf_validation_summary()

                self.validation_first_pass_complete = True

            self.path_completed_once = True

            if self.loop_playback:
                self.get_logger().info(
                    "Replay completed. Restarting because "
                    "loop_playback=true."
                )

                self.restart_playback()

            else:
                self.playback_finished = True

                self.get_logger().info(
                    "Replay completed. The final robot pose will remain "
                    "visible until Ctrl+C."
                )

    def publish_sample(
        self,
        floor_to_base: np.ndarray,
        camera_to_tag: np.ndarray,
        row: Dict[str, str],
        sample_index: int,
    ) -> bool:
        if not self.publish_bridge_for_recorded_pose(
            floor_to_base
        ):
            return False

        try:
            base_common_time_ns = int(
                Decimal(
                    row[
                        "common_time_ns"
                    ].strip()
                )
            )
        except (
            KeyError,
            ValueError,
            InvalidOperation,
        ) as error:
            self.get_logger().error(
                "[LEG REPLAY] Invalid "
                f"common_time_ns: {error}"
            )
            return False

        if not self.publish_leg_frames_for_base_time(
            base_common_time_ns,
            report_quality=True,
        ):
            return False

        # ============================================================
        # DRIVE THE ACTUAL GO2 URDF JOINTS
        # ============================================================
        if not self.publish_replay_joint_state(
            floor_to_base,
            base_common_time_ns,
        ):
            return False


        # ============================================================
        # NUMERICAL URDF RECONSTRUCTION VALIDATION
        # ============================================================
        if not self.validation_first_pass_complete:
            self.accumulate_urdf_validation(
                floor_to_base,
                base_common_time_ns,
            )


        self.publish_camera_tag_branch(
            floor_to_base,
            camera_to_tag,
        )

        self.publish_current_pose_marker(
            floor_to_base
        )

        if not self.path_completed_once:
            self.append_path_point(
                floor_to_base
            )
        else:
            # Republish the already completed path so RViz retains it.
            self.path_marker.header.stamp = (
                self.get_clock().now().to_msg()
            )
            self.marker_pub.publish(
                self.path_marker
            )

        self.get_logger().info(
            "[REPLAY] "
            f"sample={sample_index + 1}/{len(self.samples)}, "
            f"timestamp_iso={row.get('timestamp_iso', 'unknown')}, "
            f"base_position=({floor_to_base[0, 3]:.4f}, "
            f"{floor_to_base[1, 3]:.4f}, "
            f"{floor_to_base[2, 3]:.4f})"
        )

        return True

    def publish_camera_tag_branch(
        self,
        floor_to_base: np.ndarray,
        camera_to_tag: np.ndarray,
    ) -> None:
        """
        Reconstruct and publish:

            floor -> camera -> object_1
        """

        floor_to_camera = (
            floor_to_base
            @ self.P_base_to_T1
            @ np.linalg.inv(camera_to_tag)
        )

        self.publish_tf(
            floor_to_camera,
            self.floor_frame,
            self.camera_frame,
        )

        self.publish_tf(
            camera_to_tag,
            self.camera_frame,
            self.tag_frame,
        )

    def extract_replay_joint_positions(
        self,
        floor_to_base: np.ndarray,
        base_common_time_ns: int,
    ) -> Dict[str, float]:
        """
        Recover the 12 actuated Go2 leg-joint angles from the
        synchronized recorded floor-relative frame poses.

        Recorded data provides:

            floor -> base
            floor -> hip
            floor -> thigh
            floor -> calf

        From these we calculate:

            base  -> hip
            hip   -> thigh
            thigh -> calf

        The Go2 URDF uses:
            hip joint   : rotation about X
            thigh joint : rotation about Y
            calf joint  : rotation about Y
        """

        frame_group = self.leg_samples.get(
            base_common_time_ns
        )

        if frame_group is None:
            raise RuntimeError(
                "[JOINT REPLAY] No synchronized leg poses "
                f"for timestamp {base_common_time_ns}."
            )

        joint_positions = {}

        for leg_prefix in (
            "FL",
            "FR",
            "RL",
            "RR",
        ):

            # --------------------------------------------------
            # Recorded absolute floor-relative transforms
            # --------------------------------------------------
            floor_to_hip = (
                frame_group[
                    f"{leg_prefix}_hip"
                ][0]
            )

            floor_to_thigh = (
                frame_group[
                    f"{leg_prefix}_thigh"
                ][0]
            )

            floor_to_calf = (
                frame_group[
                    f"{leg_prefix}_calf"
                ][0]
            )

            # --------------------------------------------------
            # Convert absolute poses into URDF parent->child
            # relative transforms.
            #
            # base_T_hip =
            #     inv(floor_T_base) @ floor_T_hip
            #
            # hip_T_thigh =
            #     inv(floor_T_hip) @ floor_T_thigh
            #
            # thigh_T_calf =
            #     inv(floor_T_thigh) @ floor_T_calf
            # --------------------------------------------------
            base_to_hip = (
                np.linalg.inv(floor_to_base)
                @ floor_to_hip
            )

            hip_to_thigh = (
                np.linalg.inv(floor_to_hip)
                @ floor_to_thigh
            )

            thigh_to_calf = (
                np.linalg.inv(floor_to_thigh)
                @ floor_to_calf
            )

            R_hip = base_to_hip[:3, :3]
            R_thigh = hip_to_thigh[:3, :3]
            R_calf = thigh_to_calf[:3, :3]

            # --------------------------------------------------
            # HIP:
            #
            # URDF axis = [1, 0, 0]
            #
            # Rx(q) =
            #
            # [ 1    0       0   ]
            # [ 0  cos(q) -sin(q)]
            # [ 0  sin(q)  cos(q)]
            #
            # therefore:
            #
            # q = atan2(R[2,1], R[1,1])
            # --------------------------------------------------
            q_hip = np.arctan2(
                R_hip[2, 1],
                R_hip[1, 1],
            )

            # --------------------------------------------------
            # THIGH:
            #
            # URDF axis = [0, 1, 0]
            #
            # Ry(q) =
            #
            # [ cos(q)  0  sin(q)]
            # [   0     1    0   ]
            # [-sin(q)  0  cos(q)]
            #
            # therefore:
            #
            # q = atan2(R[0,2], R[0,0])
            # --------------------------------------------------
            q_thigh = np.arctan2(
                R_thigh[0, 2],
                R_thigh[0, 0],
            )

            # Calf is also a Y-axis revolute joint.
            q_calf = np.arctan2(
                R_calf[0, 2],
                R_calf[0, 0],
            )

            joint_positions[
                f"{leg_prefix}_hip_joint"
            ] = float(q_hip)

            joint_positions[
                f"{leg_prefix}_thigh_joint"
            ] = float(q_thigh)

            joint_positions[
                f"{leg_prefix}_calf_joint"
            ] = float(q_calf)

        return joint_positions

    def publish_replay_joint_state(
        self,
        floor_to_base: np.ndarray,
        base_common_time_ns: int,
    ) -> bool:
        """
        Convert the recorded leg-frame poses into the 12 actuated
        Go2 joint angles and publish them for robot_state_publisher.
        """

        try:
            joint_positions = (
                self.extract_replay_joint_positions(
                    floor_to_base,
                    base_common_time_ns,
                )
            )

        except Exception as error:
            self.get_logger().error(
                "[JOINT REPLAY] Failed to recover "
                f"joint angles: {error}"
            )
            return False

        message = JointState()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        message.name = list(
            self.REPLAY_JOINT_NAMES
        )

        message.position = [
            joint_positions[joint_name]
            for joint_name
            in self.REPLAY_JOINT_NAMES
        ]

        # These are not required for RobotModel replay.
        message.velocity = []
        message.effort = []

        self.joint_state_pub.publish(message)

        return True

    @staticmethod
    def make_translation_transform(
        xyz: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Construct a 4x4 homogeneous pure-translation transform.
        """

        transform = np.eye(4, dtype=float)

        transform[:3, 3] = np.array(
            xyz,
            dtype=float,
        )

        return transform


    @staticmethod
    def make_rotation_x_transform(
        angle_rad: float,
    ) -> np.ndarray:
        """
        Construct a 4x4 homogeneous rotation about X.
        """

        cosine = np.cos(angle_rad)
        sine = np.sin(angle_rad)

        transform = np.eye(4, dtype=float)

        transform[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine, -sine],
                [0.0, sine, cosine],
            ],
            dtype=float,
        )

        return transform



    @staticmethod
    def make_rotation_y_transform(
        angle_rad: float,
    ) -> np.ndarray:
        """
        Construct a 4x4 homogeneous rotation about Y.
        """

        cosine = np.cos(angle_rad)
        sine = np.sin(angle_rad)

        transform = np.eye(4, dtype=float)

        transform[:3, :3] = np.array(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ],
            dtype=float,
        )

        return transform

    def reconstruct_urdf_leg_transforms(
            self,
            floor_to_base: np.ndarray,
            joint_positions: Dict[str, float],
        ) -> Dict[str, np.ndarray]:
            """
            Reconstruct floor-relative hip/thigh/calf poses using
            the Go2 URDF forward kinematics.

            Returns:

                urdf_transforms["FL_hip"]   = floor_T_FL_hip
                urdf_transforms["FL_thigh"] = floor_T_FL_thigh
                urdf_transforms["FL_calf"]  = floor_T_FL_calf

            and likewise for FR, RL, RR.
            """

            urdf_transforms = {}

            for leg_prefix in (
                "FL",
                "FR",
                "RL",
                "RR",
            ):

                q_hip = joint_positions[
                    f"{leg_prefix}_hip_joint"
                ]

                q_thigh = joint_positions[
                    f"{leg_prefix}_thigh_joint"
                ]

                q_calf = joint_positions[
                    f"{leg_prefix}_calf_joint"
                ]

                # ====================================================
                # base -> hip
                #
                # URDF:
                #
                # Translation(hip origin)
                #       @
                # Rotation_X(q_hip)
                # ====================================================
                base_to_hip = (
                    self.make_translation_transform(
                        self.URDF_HIP_ORIGIN_XYZ[
                            leg_prefix
                        ]
                    )
                    @ self.make_rotation_x_transform(
                        q_hip
                    )
                )

                # ====================================================
                # hip -> thigh
                #
                # Translation(thigh origin)
                #       @
                # Rotation_Y(q_thigh)
                # ====================================================
                hip_to_thigh = (
                    self.make_translation_transform(
                        self.URDF_THIGH_ORIGIN_XYZ[
                            leg_prefix
                        ]
                    )
                    @ self.make_rotation_y_transform(
                        q_thigh
                    )
                )

                # ====================================================
                # thigh -> calf
                #
                # Translation(calf origin)
                #       @
                # Rotation_Y(q_calf)
                # ====================================================
                thigh_to_calf = (
                    self.make_translation_transform(
                        self.URDF_CALF_ORIGIN_XYZ
                    )
                    @ self.make_rotation_y_transform(
                        q_calf
                    )
                )

                # ====================================================
                # Forward kinematics in the floor frame.
                # ====================================================

                floor_to_hip = (
                    floor_to_base
                    @ base_to_hip
                )

                floor_to_thigh = (
                    floor_to_hip
                    @ hip_to_thigh
                )

                floor_to_calf = (
                    floor_to_thigh
                    @ thigh_to_calf
                )

                urdf_transforms[
                    f"{leg_prefix}_hip"
                ] = floor_to_hip

                urdf_transforms[
                    f"{leg_prefix}_thigh"
                ] = floor_to_thigh

                urdf_transforms[
                    f"{leg_prefix}_calf"
                ] = floor_to_calf

            return urdf_transforms

    @staticmethod
    def compute_transform_validation_error(
        urdf_transform: np.ndarray,
        recorded_transform: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compare one URDF-reconstructed transform against the
        corresponding recorded transform.

        Returns:

            position_error_mm
            rotation_error_deg
        """

        # ========================================================
        # POSITION ERROR
        #
        # ep = || p_URDF - p_recorded ||
        # ========================================================

        p_urdf = urdf_transform[:3, 3]

        p_recorded = recorded_transform[:3, 3]

        position_error_m = np.linalg.norm(
            p_urdf - p_recorded
        )

        position_error_mm = (
            1000.0 * position_error_m
        )

        # ========================================================
        # ROTATION ERROR
        #
        # R_error =
        #     R_URDF^T @ R_recorded
        #
        # theta =
        # acos(
        #     (trace(R_error) - 1) / 2
        # )
        # ========================================================

        R_urdf = urdf_transform[:3, :3]

        R_recorded = (
            recorded_transform[:3, :3]
        )

        R_error = (
            R_urdf.T
            @ R_recorded
        )

        cosine_angle = (
            np.trace(R_error) - 1.0
        ) / 2.0

        # Numerical floating-point protection.
        #
        # acos() requires [-1, +1].
        cosine_angle = np.clip(
            cosine_angle,
            -1.0,
            1.0,
        )

        rotation_error_rad = np.arccos(
            cosine_angle
        )

        rotation_error_deg = np.degrees(
            rotation_error_rad
        )

        return (
            float(position_error_mm),
            float(rotation_error_deg),
        )

    def accumulate_urdf_validation(
        self,
        floor_to_base: np.ndarray,
        base_common_time_ns: int,
    ) -> None:
        """
        For one replay sample:

        1. Recover the 12 joint angles.
        2. Reconstruct URDF hip/thigh/calf poses using FK.
        3. Compare them against the recorded synchronized poses.
        4. Store position and rotation errors.
        """

        if self.validation_first_pass_complete:
            return

        frame_group = self.leg_samples.get(
            base_common_time_ns
        )

        if frame_group is None:
            raise RuntimeError(
                "[URDF VALIDATION] Missing leg group "
                f"for timestamp {base_common_time_ns}."
            )

        # --------------------------------------------------------
        # Recover q from the recorded spatial poses.
        # --------------------------------------------------------
        joint_positions = (
            self.extract_replay_joint_positions(
                floor_to_base,
                base_common_time_ns,
            )
        )

        # --------------------------------------------------------
        # Apply URDF forward kinematics using those q values.
        # --------------------------------------------------------
        urdf_transforms = (
            self.reconstruct_urdf_leg_transforms(
                floor_to_base,
                joint_positions,
            )
        )

        # --------------------------------------------------------
        # Compare the reconstructed URDF transform against the
        # original synchronized recorded transform.
        # --------------------------------------------------------
        for frame_name in self.VALIDATION_FRAME_NAMES:

            urdf_transform = (
                urdf_transforms[frame_name]
            )

            recorded_transform = (
                frame_group[frame_name][0]
            )

            (
                position_error_mm,
                rotation_error_deg,
            ) = self.compute_transform_validation_error(
                urdf_transform,
                recorded_transform,
            )

            self.validation_errors[
                frame_name
            ][
                "position_error_mm"
            ].append(
                position_error_mm
            )

            self.validation_errors[
                frame_name
            ][
                "rotation_error_deg"
            ].append(
                rotation_error_deg
            )

        self.validation_sample_count += 1


    def print_urdf_validation_summary(
        self,
    ) -> None:
        """
        Print mean/max reconstruction errors for every
        hip, thigh, and calf frame.
        """

        if self.validation_sample_count == 0:
            self.get_logger().warn(
                "[URDF VALIDATION] No validation "
                "samples were collected."
            )
            return

        self.get_logger().info(
            "============================================================"
        )

        self.get_logger().info(
            "[URDF VALIDATION] "
            f"Results over {self.validation_sample_count} "
            "replay samples"
        )

        self.get_logger().info(
            "============================================================"
        )

        all_position_errors = []
        all_rotation_errors = []

        for frame_name in self.VALIDATION_FRAME_NAMES:

            position_errors = np.asarray(
                self.validation_errors[
                    frame_name
                ][
                    "position_error_mm"
                ],
                dtype=float,
            )

            rotation_errors = np.asarray(
                self.validation_errors[
                    frame_name
                ][
                    "rotation_error_deg"
                ],
                dtype=float,
            )

            if position_errors.size == 0:
                continue

            mean_position_error = float(
                np.mean(position_errors)
            )

            max_position_error = float(
                np.max(position_errors)
            )

            mean_rotation_error = float(
                np.mean(rotation_errors)
            )

            max_rotation_error = float(
                np.max(rotation_errors)
            )

            all_position_errors.extend(
                position_errors.tolist()
            )

            all_rotation_errors.extend(
                rotation_errors.tolist()
            )

            self.get_logger().info(
                "[URDF VALIDATION] "
                f"{frame_name}: "
                f"mean_position={mean_position_error:.4f} mm, "
                f"max_position={max_position_error:.4f} mm, "
                f"mean_rotation={mean_rotation_error:.6f} deg, "
                f"max_rotation={max_rotation_error:.6f} deg"
            )

        # --------------------------------------------------------
        # Overall result across all 12 reconstructed frames.
        # --------------------------------------------------------

        if all_position_errors:

            overall_mean_position = float(
                np.mean(all_position_errors)
            )

            overall_max_position = float(
                np.max(all_position_errors)
            )

            overall_mean_rotation = float(
                np.mean(all_rotation_errors)
            )

            overall_max_rotation = float(
                np.max(all_rotation_errors)
            )

            self.get_logger().info(
                "------------------------------------------------------------"
            )

            self.get_logger().info(
                "[URDF VALIDATION] OVERALL: "
                f"mean_position={overall_mean_position:.4f} mm, "
                f"max_position={overall_max_position:.4f} mm, "
                f"mean_rotation={overall_mean_rotation:.6f} deg, "
                f"max_rotation={overall_max_rotation:.6f} deg"
            )

        self.get_logger().info(
            "============================================================"
        )























    


    def publish_leg_frames_for_base_time(
            self,
            base_common_time_ns: int,
            report_quality: bool = True,
        ) -> bool:
            """
            Publish the 24 synchronized recorded leg frames
            corresponding to one base timestamp.

            The stored matrices are already:

                floor -> leg_frame

            so NO forward kinematics is required here.
            """

            frame_group = self.leg_samples.get(
                base_common_time_ns
            )

            if frame_group is None:
                self.get_logger().error(
                    "[LEG REPLAY] No synchronized "
                    "leg poses for base timestamp "
                    f"{base_common_time_ns}."
                )
                return False

            warning_frames = []
            maximum_gap_ms = 0.0

            for frame_name in self.LEG_FRAME_NAMES:

                (
                    floor_to_leg,
                    large_gap_warning,
                    interpolation_gap_ms,
                ) = frame_group[frame_name]

                # IMPORTANT:
                # Do not publish the original URDF frame name.
                #
                # robot_state_publisher already owns:
                #
                #     FL_hip
                #     FL_thigh
                #     ...
                #
                # Therefore use a unique replay name.
                replay_frame_name = (
                    f"replay_{frame_name}"
                )

                self.publish_tf(
                    floor_to_leg,
                    self.floor_frame,
                    replay_frame_name,
                )

                if large_gap_warning == 1:
                    warning_frames.append(
                        frame_name
                    )

                    maximum_gap_ms = max(
                        maximum_gap_ms,
                        interpolation_gap_ms,
                    )

            if (
                report_quality
                and warning_frames
            ):
                self.get_logger().warn(
                    "[LEG REPLAY] "
                    f"{len(warning_frames)}/24 leg frames "
                    "at this base sample were generated "
                    "from an interpolation gap >100 ms. "
                    f"Maximum gap={maximum_gap_ms:.3f} ms. "
                    "They are still being replayed."
                )

            return True




    def publish_bridge_for_recorded_pose(
        self,
        floor_to_base: np.ndarray,
    ) -> bool:
        """
        Publish floor -> map so that the existing map -> base_link subtree
        appears at the recorded floor -> base pose.
        """
        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except Exception as error:
            now_ns = time.monotonic_ns()
            if now_ns - self.last_tf_warning_ns >= 1_000_000_000:
                self.get_logger().warn(
                    f"Waiting for TF {self.map_frame} -> "
                    f"{self.base_frame}: {error}"
                )
                self.last_tf_warning_ns = now_ns
            return False

        map_to_base = self.transform_stamped_to_matrix(
            map_to_base_tf
        )
        floor_to_map = (
            floor_to_base
            @ np.linalg.inv(map_to_base)
        )
        self.publish_tf(
            floor_to_map,
            self.floor_frame,
            self.map_frame,
        )
        return True

    def publish_tf(
        self,
        matrix: np.ndarray,
        parent: str,
        child: str,
    ) -> None:
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = parent
        message.child_frame_id = child

        message.transform.translation.x = float(matrix[0, 3])
        message.transform.translation.y = float(matrix[1, 3])
        message.transform.translation.z = float(matrix[2, 3])

        quaternion = R.from_matrix(
            matrix[:3, :3]
        ).as_quat()
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])

        self.tf_broadcaster.sendTransform(message)

    def restart_playback(self) -> None:
        self.sample_index = 0
        self.playback_start_monotonic_ns = time.monotonic_ns()
        self.playback_finished = False
        self.last_floor_to_base = None
        self.last_camera_to_tag = None
        self.last_base_common_time_ns = None
       

    def create_path_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.floor_frame
        marker.ns = "floor_to_robot_replay_path"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.015
        marker.color = ColorRGBA(
            r=0.1,
            g=0.8,
            b=0.1,
            a=1.0,
        )
        marker.pose.orientation.w = 1.0
        marker.points = []
        return marker

    def append_path_point(
        self,
        floor_to_base: np.ndarray,
    ) -> None:
        self.path_marker.points.append(
            Point(
                x=float(floor_to_base[0, 3]),
                y=float(floor_to_base[1, 3]),
                z=float(floor_to_base[2, 3]),
            )
        )
        self.path_marker.header.stamp = (
            self.get_clock().now().to_msg()
        )
        self.marker_pub.publish(self.path_marker)

    def publish_current_pose_marker(
        self,
        floor_to_base: np.ndarray,
    ) -> None:
        marker = Marker()
        marker.header.frame_id = self.floor_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "floor_to_robot_replay_current_pose"
        marker.id = 2
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color = ColorRGBA(
            r=1.0,
            g=0.25,
            b=0.1,
            a=1.0,
        )

        marker.pose.position.x = float(floor_to_base[0, 3])
        marker.pose.position.y = float(floor_to_base[1, 3])
        marker.pose.position.z = float(floor_to_base[2, 3])

        quaternion = R.from_matrix(
            floor_to_base[:3, :3]
        ).as_quat()
        marker.pose.orientation.x = float(quaternion[0])
        marker.pose.orientation.y = float(quaternion[1])
        marker.pose.orientation.z = float(quaternion[2])
        marker.pose.orientation.w = float(quaternion[3])

        self.marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = FilteredFloorToRobotReplay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Replay interrupted by user.")
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f"Fatal replay error: {error}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


