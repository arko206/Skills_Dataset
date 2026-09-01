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

    # ============================================================
    # FIXED CALIBRATION MATRIX P*(Rc -> T1)
    # ============================================================
    P_star_Rc_to_T1 = np.array(
        [
            [0.998833, 0.039814, -0.027331, 0.008661],
            [-0.025861, -0.036982, -0.998981, -0.049915],
            [-0.040784, 0.998523, -0.035909, -0.107167],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ],
        dtype=float,
    )

    def __init__(self) -> None:
        super().__init__("filtered_floor_to_robot_replay")

        self.declare_parameter(
            "csv_file",
            os.path.expanduser(
                "~/Go2_Walk_Base_Data_Sensor/"
                "/Parameters_RS/Fd_1.2_0.0_0.0/Tf_stationary_1.2_0.0_0.0.csv"
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
        self.marker_pub = self.create_publisher(
            Marker,
            "floor_to_robot_replay_markers",
            10,
        )

        self.samples = self.load_filtered_csv(self.csv_file)
        self.sample_index = 0
        self.playback_start_monotonic_ns = time.monotonic_ns()
        self.playback_finished = False
        self.last_floor_to_base: Optional[np.ndarray] = None
        self.last_camera_to_tag: Optional[np.ndarray] = None
        self.path_marker = self.create_path_marker()
        self.path_completed_once = False
        self.last_tf_warning_ns = 0

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

        if self.sample_index >= len(self.samples):
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


